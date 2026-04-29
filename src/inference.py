from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .model import load_model
    from .utils import build_transforms, crop_largest_face, get_device
except ImportError:
    from model import load_model
    from utils import build_transforms, crop_largest_face, get_device


@lru_cache(maxsize=1)
def get_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade from {cascade_path}")
    return cascade


def detect_largest_face_box(
    image: Image.Image,
    margin: float = 0.20,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    rgb_image = image.convert("RGB")
    np_image = np.array(rgb_image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

    faces = get_face_cascade().detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None, False

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    pad_w = int(w * margin)
    pad_h = int(h * margin)

    left = max(x - pad_w, 0)
    top = max(y - pad_h, 0)
    right = min(x + w + pad_w, np_image.shape[1])
    bottom = min(y + h + pad_h, np_image.shape[0])
    return (left, top, right, bottom), True


def detect_and_crop_face(image: Image.Image, margin: float = 0.20) -> tuple[Image.Image, bool]:
    rgb_image = image.convert("RGB")
    face_box, detected = detect_largest_face_box(rgb_image, margin=margin)
    if not detected or face_box is None:
        return rgb_image, False

    left, top, right, bottom = face_box
    return rgb_image.crop((left, top, right, bottom)), True


def preprocess_image(image: Image.Image, assume_cropped: bool = False) -> torch.Tensor:
    prepared = image.convert("RGB") if assume_cropped else detect_and_crop_face(image)[0]
    transform = build_transforms(train=False)
    return transform(prepared).unsqueeze(0)


def predict_image(
    image: Image.Image | str | Path,
    model: torch.nn.Module,
    device: torch.device,
    threshold: float | None = None,
    assume_cropped: bool = False,
) -> dict[str, Any]:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    processed_image, face_detected = (image, True) if assume_cropped else detect_and_crop_face(image)
    tensor = preprocess_image(processed_image, assume_cropped=True).to(device)
    model.eval()

    with torch.no_grad():
        logit = model(tensor).view(-1)[0]
        fake_probability = torch.sigmoid(logit).item()

    threshold = getattr(model, "threshold", 0.5) if threshold is None else threshold
    is_fake = fake_probability >= threshold
    prediction = "Fake" if is_fake else "Real"
    confidence = fake_probability if is_fake else 1.0 - fake_probability

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "fake_probability": round(float(fake_probability), 4),
        "threshold": round(float(threshold), 4),
        "processed_image": processed_image,
        "face_detected": face_detected,
    }


def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    raise ValueError("Could not find a suitable convolution layer for Grad-CAM.")


def generate_gradcam(
    image: Image.Image | str | Path,
    model: torch.nn.Module,
    device: torch.device,
    target_label: int | None = None,
    assume_cropped: bool = False,
) -> dict[str, np.ndarray]:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    processed_image = image if assume_cropped else detect_and_crop_face(image)[0]
    tensor = preprocess_image(processed_image, assume_cropped=True).to(device)
    target_layer = _get_target_layer(model)
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        if not output.requires_grad:
            output.requires_grad_(True)
        activations = output.detach()
        output.register_hook(_capture_gradients)

    def _capture_gradients(grad: torch.Tensor):
        nonlocal gradients
        gradients = grad.detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        logit = output.view(-1)[0]

        if target_label is None:
            fake_probability = torch.sigmoid(logit).item()
            threshold = getattr(model, "threshold", 0.5)
            target_label = 1 if fake_probability >= threshold else 0

        score = logit if target_label == 1 else -logit
        score.backward()

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(processed_image.height, processed_image.width),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        base_image = np.array(processed_image, dtype=np.float32)
        overlay = np.clip(0.6 * base_image + 0.4 * heatmap.astype(np.float32), 0, 255).astype(np.uint8)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
        }
    finally:
        forward_handle.remove()
        model.zero_grad(set_to_none=True)


def load_detector(checkpoint_path: str = "models/best_model.pth") -> tuple[torch.nn.Module, torch.device]:
    device = get_device()
    model = load_model(checkpoint_path, device=device, pretrained=False)
    return model, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("image", help="Image path.")
    parser.add_argument("--checkpoint", default="models/best_model.pth", help="Model checkpoint path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector, detector_device = load_detector(args.checkpoint)
    print(predict_image(args.image, detector, detector_device))
