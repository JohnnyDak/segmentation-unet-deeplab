"""
DeepLabV3 (from scratch) — Chen et al., 2017.
https://arxiv.org/abs/1706.05587

Composé de :
1. ResNetBackbone (voir resnet_backbone.py) : extrait des features à
   output_stride=16 (résolution H/16 x W/16).
2. ASPP (Atrous Spatial Pyramid Pooling) : capture le contexte à
   plusieurs échelles en parallèle, via des convolutions dilatées de
   taux différents — voir la fiche théorique, section 6.
3. Une tête de classification simple : projette vers `num_classes`,
   puis remonte à la résolution de l'image d'entrée par interpolation
   bilinéaire (pas de décodeur symétrique avec skip connections comme
   U-Net — c'est la différence architecturale centrale entre les deux).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet_backbone import ResNetBackbone


class ASPPConv(nn.Sequential):
    """Une branche de l'ASPP : une convolution 3x3 dilatée avec un taux donné."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    """Branche de contexte global de l'ASPP : moyenne globale de l'image
    entière (pooling adaptatif vers 1x1), puis projection, puis on
    réétale ("broadcast") à la résolution d'origine. Capture le contexte
    le plus large possible (toute l'image), en complément des branches
    dilatées qui capturent des contextes locaux à moyens."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.relu(self.bn(self.conv(x)))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.

    Applique en parallèle : une conv 1x1, trois convs 3x3 dilatées
    (taux 6, 12, 18 — valeurs du papier original), et une branche de
    pooling global. Les 5 sorties sont concaténées puis fusionnées par
    une conv 1x1 finale.
    """

    def __init__(self, in_channels: int = 512, out_channels: int = 256, atrous_rates=(6, 12, 18)):
        super().__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        r1, r2, r3 = atrous_rates
        self.aspp_conv1 = ASPPConv(in_channels, out_channels, dilation=r1)
        self.aspp_conv2 = ASPPConv(in_channels, out_channels, dilation=r2)
        self.aspp_conv3 = ASPPConv(in_channels, out_channels, dilation=r3)
        self.pooling = ASPPPooling(in_channels, out_channels)

        # Fusionne les 5 branches (5 x out_channels canaux après concat)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [
            self.conv_1x1(x),
            self.aspp_conv1(x),
            self.aspp_conv2(x),
            self.aspp_conv3(x),
            self.pooling(x),
        ]
        x = torch.cat(branches, dim=1)
        return self.project(x)


class DeepLabV3(nn.Module):
    """
    DeepLabV3, from scratch : ResNetBackbone -> ASPP -> conv 1x1 finale
    -> upsampling bilinéaire vers la résolution d'entrée.

    Sortie : logits de forme (batch, num_classes, H, W), comme UNet
    (voir src/models/unet.py) — les deux modèles sont donc interchangeables
    pour l'entraînement et l'évaluation.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels=in_channels)
        self.aspp = ASPP(in_channels=512, out_channels=256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        features = self.backbone(x)   # (B, 512, H/16, W/16)
        x = self.aspp(features)       # (B, 256, H/16, W/16)
        x = self.classifier(x)        # (B, num_classes, H/16, W/16)

        # Remonte à la résolution de l'image d'entrée. Contrairement à
        # U-Net, ce n'est pas appris (pas de transposed convolution) :
        # une simple interpolation bilinéaire suffit ici, car l'ASPP a
        # déjà consolidé l'information à basse résolution.
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    # Test rapide de cohérence des formes (à exécuter sur Colab).
    model = DeepLabV3(in_channels=3, num_classes=2)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("Entrée :", dummy.shape)
    print("Sortie :", out.shape)  # attendu : torch.Size([2, 2, 256, 256])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")