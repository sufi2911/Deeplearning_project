from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm

try:
    from .model import build_model
    from .utils import (
        FaceCropImageFolder,
        build_transforms,
        ensure_dir,
        get_binary_class_counts,
        get_device,
        get_sample_weights,
        set_seed,
        validate_split_folder,
    )
except ImportError:
    from model import build_model
    from utils import (
        FaceCropImageFolder,
        build_transforms,
        ensure_dir,
        get_binary_class_counts,
        get_device,
        get_sample_weights,
        set_seed,
        validate_split_folder,
    )


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits.squeeze(1))
    predictions = (probabilities >= 0.5).float()
    return (predictions == targets).float().mean().item()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss = 0.0
    running_accuracy = 0.0
    total_samples = 0

    progress = tqdm(dataloader, desc="Train" if is_training else "Validate", leave=False)
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits.squeeze(1), targets)

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_accuracy += binary_accuracy(logits, targets) * batch_size
            total_samples += batch_size

            progress.set_postfix(
                loss=running_loss / total_samples,
                acc=running_accuracy / total_samples,
            )

    return running_loss / total_samples, running_accuracy / total_samples


def collect_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities.append(torch.sigmoid(logits.squeeze(1)).cpu())
            labels.append(targets.cpu())

    return torch.cat(probabilities), torch.cat(labels)


def parse_thresholds(raw_thresholds: str) -> list[float]:
    return [float(value.strip()) for value in raw_thresholds.split(",") if value.strip()]


def tune_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    thresholds: list[float],
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    best_accuracy = -1.0

    actual = labels.int().numpy()
    for threshold in thresholds:
        predictions = (probabilities >= threshold).int().numpy()
        macro_f1 = f1_score(actual, predictions, average="macro")
        accuracy = float((predictions == actual).mean())

        if (macro_f1, accuracy) > (best_f1, best_accuracy):
            best_threshold = threshold
            best_f1 = macro_f1
            best_accuracy = accuracy

    return best_threshold, best_f1, best_accuracy


def replace_classifier_head(model: nn.Module, dropout: float = 0.4) -> None:
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Sequential):
        in_features = classifier[-1].in_features
    else:
        in_features = classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))


def set_classifier_only_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def set_last_blocks_trainable(model: nn.Module, blocks_to_unfreeze: int) -> None:
    set_classifier_only_trainable(model)
    if blocks_to_unfreeze <= 0:
        return
    if not hasattr(model, "blocks"):
        raise ValueError("This EfficientNet-B0 model does not expose blocks for staged fine-tuning.")
    for block in model.blocks[-blocks_to_unfreeze:]:
        for parameter in block.parameters():
            parameter.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_float32_matmul_precision("high")

    dataset_dir = Path(args.dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    validate_split_folder(train_dir)
    validate_split_folder(val_dir)

    train_dataset = FaceCropImageFolder(train_dir, transform=build_transforms(train=True))
    val_dataset = FaceCropImageFolder(val_dir, transform=build_transforms(train=False))
    train_counts = get_binary_class_counts(train_dataset)
    val_counts = get_binary_class_counts(val_dataset)

    sampler = None
    shuffle = True
    if args.balance_strategy in {"sampler", "both"}:
        sample_weights = torch.DoubleTensor(get_sample_weights(train_dataset))
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    use_persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
    )

    model = build_model(pretrained=True, freeze_features=True, unfreeze_last_blocks=0)
    replace_classifier_head(model, dropout=0.4)
    set_classifier_only_trainable(model)
    model = model.to(device)

    if args.balance_strategy in {"loss", "both"}:
        pos_weight_value = train_counts["real"] / max(train_counts["fake"], 1)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    threshold_candidates = parse_thresholds(args.thresholds)
    model_dir = ensure_dir(args.model_dir)
    best_model_path = model_dir / "best_model.pth"

    print(f"Device: {device}")
    print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")
    print(f"Train class counts: real={train_counts['real']} fake={train_counts['fake']}")
    print(f"Val class counts: real={val_counts['real']} fake={val_counts['fake']}")
    print(f"Balance strategy: {args.balance_strategy}")
    print(f"Stage 1: classifier only, epochs={args.stage1_epochs}, lr={args.lr}")
    print(
        f"Stage 2: last {args.unfreeze_last_blocks} EfficientNet blocks, "
        f"epochs={max(args.epochs - args.stage1_epochs, 0)}, lr={args.stage2_lr}"
    )
    print(f"Saving best checkpoint to: {best_model_path}")

    stages = [
        ("Stage 1", args.stage1_epochs, args.lr, 0),
        ("Stage 2", max(args.epochs - args.stage1_epochs, 0), args.stage2_lr, args.unfreeze_last_blocks),
    ]
    global_epoch = 0

    for stage_name, stage_epochs, learning_rate, blocks_to_unfreeze in stages:
        if stage_epochs <= 0:
            continue

        if blocks_to_unfreeze > 0:
            set_last_blocks_trainable(model, blocks_to_unfreeze)
        else:
            set_classifier_only_trainable(model)

        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        print(f"\n{stage_name}: trainable parameters={count_trainable_parameters(model):,}")

        for _ in range(stage_epochs):
            global_epoch += 1
            print(f"\nEpoch {global_epoch}/{args.epochs}")
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            print(
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                val_probabilities, val_labels = collect_probabilities(model, val_loader, device)
                threshold, best_f1, threshold_acc = tune_threshold(
                    val_probabilities,
                    val_labels,
                    threshold_candidates,
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_mapping": {"real": 0, "fake": 1},
                        "image_size": 224,
                        "threshold": threshold,
                        "val_macro_f1_at_threshold": best_f1,
                        "val_accuracy_at_threshold": threshold_acc,
                        "val_loss": best_val_loss,
                        "epoch": global_epoch,
                        "stage": stage_name,
                        "train_class_counts": train_counts,
                        "val_class_counts": val_counts,
                    },
                    best_model_path,
                )
                print(
                    "Saved new best model "
                    f"(threshold={threshold:.2f}, val_macro_f1={best_f1:.4f}, "
                    f"val_acc={threshold_acc:.4f})."
                )
            else:
                patience_counter += 1
                print(f"No validation improvement. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 deepfake detector.")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to dataset root.")
    parser.add_argument("--model-dir", default="models", help="Directory for model checkpoints.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=25, help="Total epochs across both stages.")
    parser.add_argument("--stage1-epochs", type=int, default=3, help="Classifier-only warmup epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Stage 1 Adam learning rate.")
    parser.add_argument("--stage2-lr", type=float, default=1e-5, help="Stage 2 Adam learning rate.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--cpu-threads", type=int, default=12, help="CPU threads for torch ops.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--balance-strategy",
        choices=["sampler", "loss", "both", "none"],
        default="none",
        help="Optional imbalance handling. Keep none for balanced datasets.",
    )
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=4,
        help="Fine-tune the last N EfficientNet blocks in stage 2.",
    )
    parser.add_argument("--thresholds", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6", help="Threshold candidates for F1 tuning.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
