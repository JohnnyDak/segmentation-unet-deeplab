"""
U-Net (from scratch), version allégée — Ronneberger et al., 2015.
https://arxiv.org/abs/1505.04597

Version allégée (32->64->128->256->512 filtres, au lieu de 64->...->1024
dans le papier original) : choisie car on entraîne from scratch (aucun
poids pré-entraîné autorisé) sur un dataset encore modeste (~1400 images
d'entraînement, ISIC 2017) — moins de paramètres réduit le risque de
surapprentissage et accélère l'entraînement sur le GPU T4 gratuit de Colab.

Rappel d'architecture (voir la fiche de révision) :
- Encodeur : réduit la résolution (MaxPool) et augmente le nombre de
  filtres à chaque étage -> agrandit le champ réceptif (contexte global).
- Décodeur : remonte la résolution via des transposed convolutions.
- Skip connections : à chaque étage, la carte de features de l'encodeur
  (avant réduction) est concaténée à la carte correspondante du décodeur,
  pour récupérer les détails fins perdus pendant le downsampling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2 — le bloc de base répété à
    chaque étage de U-Net, aussi bien dans l'encodeur que le décodeur."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Un étage de l'encodeur : MaxPool (divise H,W par 2) puis DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Un étage du décodeur : transposed convolution (remonte H,W x2 et
    divise les canaux par 2), concaténation avec la skip connection de
    l'encodeur, puis DoubleConv pour fusionner l'information."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # in_channels -> in_channels // 2 (= out_channels, par construction
        # des tailles choisies dans UNet ci-dessous)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # après concat : (in_channels // 2) + out_channels canaux = in_channels
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_decoder: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_decoder = self.up(x_decoder)

        # Sécurité si H/W ne tombent pas parfaitement juste (dimensions
        # impaires à un étage intermédiaire) : on recadre par padding.
        diff_h = x_skip.size(2) - x_decoder.size(2)
        diff_w = x_skip.size(3) - x_decoder.size(3)
        x_decoder = F.pad(
            x_decoder,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )

        x = torch.cat([x_skip, x_decoder], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Convolution 1x1 finale : projette vers `num_classes` cartes de
    logits, une par classe, à la résolution de l'image d'entrée."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net allégé, from scratch.

    Args:
        in_channels: canaux de l'image d'entrée (3 pour RGB).
        num_classes: nombre de classes de sortie (2 pour ISIC : fond/lésion).
        base_filters: nombre de filtres du premier étage (32 par défaut ;
            64 pour retrouver l'architecture du papier original).

    Sortie : logits de forme (batch, num_classes, H, W) — pas de softmax
    appliqué ici, car nn.CrossEntropyLoss l'applique en interne, et pour
    Dice loss on appliquera softmax explicitement dans le module de loss.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_filters: int = 32):
        super().__init__()
        f = base_filters  # ex. 32 -> étages : 32, 64, 128, 256, 512

        # Encodeur
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)  # bottleneck

        # Décodeur (avec skip connections)
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)

        self.outc = OutConv(f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodeur : on garde une copie à chaque étage pour les skip connections
        x1 = self.inc(x)      # (B, f,    H,    W)
        x2 = self.down1(x1)   # (B, f*2,  H/2,  W/2)
        x3 = self.down2(x2)   # (B, f*4,  H/4,  W/4)
        x4 = self.down3(x3)   # (B, f*8,  H/8,  W/8)
        x5 = self.down4(x4)   # (B, f*16, H/16, W/16)  <- bottleneck

        # Décodeur : remonte en fusionnant avec chaque skip connection
        x = self.up1(x5, x4)  # (B, f*8,  H/8,  W/8)
        x = self.up2(x, x3)   # (B, f*4,  H/4,  W/4)
        x = self.up3(x, x2)   # (B, f*2,  H/2,  W/2)
        x = self.up4(x, x1)   # (B, f,    H,    W)

        return self.outc(x)   # (B, num_classes, H, W)


if __name__ == "__main__":
    # Test rapide de cohérence des formes (à exécuter sur Colab où torch
    # est disponible) : une image factice 256x256 doit ressortir avec la
    # même résolution spatiale, et num_classes canaux.
    model = UNet(in_channels=3, num_classes=2, base_filters=32)
    dummy = torch.randn(2, 3, 256, 256)  # batch de 2 images 256x256 RGB
    out = model(dummy)
    print("Entrée :", dummy.shape)
    print("Sortie :", out.shape)  # attendu : torch.Size([2, 2, 256, 256])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")