from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VALID_REAL_NAMES = {"real", "real_image"}
VALID_FAKE_NAMES = {"fake", "fake_image"}
VALID_CLASS_NAMES = VALID_REAL_NAMES | VALID_FAKE_NAMES


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def crop_largest_face(image: Image.Image, margin: float = 0.20) -> Image.Image:
    """Compatibility helper: dataset images are already cropped faces."""
    return image.convert("RGB")


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomRotation(10),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _label_name_to_binary(label_name: str) -> float:
    normalized = label_name.lower()
    if normalized in VALID_FAKE_NAMES or "fake" in normalized:
        return 1.0
    if normalized in VALID_REAL_NAMES or "real" in normalized:
        return 0.0
    raise ValueError(
        f"Unsupported class folder '{label_name}'. Expected real/fake folders "
        "such as real, fake, real_image, or fake_image."
    )


class FaceCropImageFolder(Dataset):
    """ImageFolder dataset for already-cropped face images.

    Labels are normalized to binary targets:
    real -> 0.0, fake -> 1.0.
    """

    def __init__(self, root: str | Path, transform: transforms.Compose | None = None):
        self.root = Path(root)
        self.dataset = datasets.ImageFolder(str(self.root))
        self.transform = transform
        self.class_to_idx = self.dataset.class_to_idx
        self.classes = self.dataset.classes
        self.binary_labels = {
            class_idx: _label_name_to_binary(class_name)
            for class_name, class_idx in self.class_to_idx.items()
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, class_idx = self.dataset[index]
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.binary_labels[class_idx], dtype=torch.float32)
        return image, label


def validate_split_folder(split_dir: str | Path) -> None:
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_path}")

    class_dirs = {path.name.lower() for path in split_path.iterdir() if path.is_dir()}
    has_real = any(name in VALID_REAL_NAMES or "real" in name for name in class_dirs)
    has_fake = any(name in VALID_FAKE_NAMES or "fake" in name for name in class_dirs)
    if not (has_real and has_fake):
        raise ValueError(
            f"{split_path} must contain real/fake class folders for ImageFolder. "
            "Expected folders like real/ and fake/ or real_image/ and fake_image/."
        )


def get_binary_class_counts(dataset: FaceCropImageFolder) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for _, class_idx in dataset.dataset.samples:
        label = int(dataset.binary_labels[class_idx])
        if label == 1:
            counts["fake"] += 1
        else:
            counts["real"] += 1
    return counts


def get_sample_weights(dataset: FaceCropImageFolder) -> list[float]:
    counts = get_binary_class_counts(dataset)
    label_counts = {0: counts["real"], 1: counts["fake"]}
    return [
        1.0 / max(label_counts[int(dataset.binary_labels[class_idx])], 1)
        for _, class_idx in dataset.dataset.samples
    ]
