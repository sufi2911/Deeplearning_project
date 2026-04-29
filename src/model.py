from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError as exc:
    raise ImportError("Install timm before creating the model: pip install timm") from exc


def build_model(
    pretrained: bool = True,
    freeze_features: bool = True,
    unfreeze_last_blocks: int = 0,
) -> nn.Module:
    """Create EfficientNet-B0 with a lightweight binary classifier head."""
    model = timm.create_model("efficientnet_b0", pretrained=pretrained)

    if freeze_features:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if unfreeze_last_blocks > 0:
        if not hasattr(model, "blocks"):
            raise ValueError("This EfficientNet implementation does not expose blocks to unfreeze.")
        for block in model.blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    in_features = model.get_classifier().in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1),
    )
    return model


def load_model(
    checkpoint_path: str,
    device: torch.device,
    pretrained: bool = False,
) -> nn.Module:
    model = build_model(pretrained=pretrained, freeze_features=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.threshold = float(checkpoint.get("threshold", 0.5))
    model.to(device)
    model.eval()
    return model
