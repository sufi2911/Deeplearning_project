from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

try:
    from .model import load_model
    from .utils import build_transforms, ensure_dir, get_device, validate_split_folder
except ImportError:
    from model import load_model
    from utils import build_transforms, ensure_dir, get_device, validate_split_folder


def parse_thresholds(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def choose_best_threshold(y_true, y_prob, thresholds):
    best_t, best_f1 = thresholds[0], -1
    for t in thresholds:
        preds = [1 if p >= t else 0 for p in y_prob]
        f1 = f1_score(y_true, preds, average="macro")
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def evaluate(args):
    device = get_device()
    torch.set_num_threads(max(1, args.cpu_threads))

    test_dir = Path(args.dataset_dir) / "val"
    validate_split_folder(test_dir)

    #  Load dataset
    dataset = ImageFolder(test_dir, transform=build_transforms(train=False))
    print("Class mapping:", dataset.class_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    #  Load model
    model = load_model(args.checkpoint, device=device, pretrained=False)
    model.eval()

    y_true, y_prob = [], []

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluate"):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(1)).cpu()

            #  FIX: map labels using folder names (bulletproof)
            for i, label in enumerate(targets):
                class_name = dataset.classes[label]
                if "fake" in class_name.lower():
                    y_true.append(1)
                else:
                    y_true.append(0)

            y_prob.extend(probs.tolist())

    #  Threshold tuning
    thresholds = parse_thresholds(args.thresholds)
    threshold, best_f1 = choose_best_threshold(y_true, y_prob, thresholds)

    if args.threshold is not None:
        threshold = args.threshold

    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    roc_auc = roc_auc_score(y_true, y_prob)

    report = classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4)
    matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Best Threshold: {threshold:.2f}")
    print(f"Macro F1: {best_f1:.4f}")
    print(report)

    print("Confusion Matrix")
    print(matrix)

    output_dir = ensure_dir(args.output_dir)

    # Save report
    (output_dir / "classification_report.txt").write_text(report)

    # Save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--output-dir", default="models/evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--thresholds", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())