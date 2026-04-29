from __future__ import annotations

from pathlib import Path

try:
    from .utils import FaceCropImageFolder, get_binary_class_counts, validate_split_folder
except ImportError:
    from utils import FaceCropImageFolder, get_binary_class_counts, validate_split_folder


def main() -> None:
    dataset_dir = Path("dataset")
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        validate_split_folder(split_dir)
        dataset = FaceCropImageFolder(split_dir)
        counts = get_binary_class_counts(dataset)
        total = counts["real"] + counts["fake"]
        real_pct = counts["real"] / max(total, 1) * 100
        fake_pct = counts["fake"] / max(total, 1) * 100
        ratio = counts["fake"] / max(counts["real"], 1)
        print(
            f"{split}: real={counts['real']} ({real_pct:.2f}%) "
            f"fake={counts['fake']} ({fake_pct:.2f}%) fake:real={ratio:.2f}:1"
        )


if __name__ == "__main__":
    main()
