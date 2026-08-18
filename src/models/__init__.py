from src.models.deeplab import build_deeplabv3
from src.models.unet import UNet

import torch.nn as nn


def build_model(architecture: str, num_classes: int = 2, encoder: str = "resnet34") -> nn.Module:
    """Construit un modèle à partir du nom d'architecture (cf. configs/config.yaml : model.architecture)."""
    if architecture == "unet":
        return UNet(num_classes=num_classes)
    if architecture == "deeplabv3":
        return build_deeplabv3(num_classes=num_classes, encoder=encoder)
    raise ValueError(f"Architecture inconnue : {architecture!r} (attendu : 'unet' ou 'deeplabv3')")
