"""
Wrapper DeepLabV3 (Chen et al., 2017) via segmentation_models_pytorch.

Contrairement à U-Net (implémenté from scratch), DeepLabV3 est réutilisé
depuis une bibliothèque établie : c'est un choix assumé (cf. README), pas un
raccourci — l'effort d'implémentation "maison" porte sur U-Net, DeepLabV3
sert de point de comparaison architectural fidèle à l'article de référence.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_deeplabv3(
    num_classes: int = 2,
    encoder: str = "resnet34",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    return smp.DeepLabV3(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
