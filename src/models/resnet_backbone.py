"""
Backbone ResNet (style ResNet18), from scratch — utilisé comme encodeur
pour DeepLabV3.

Point important à ne pas confondre avec U-Net : les "skip connections"
d'un ResNet sont INTERNES à chaque bloc résiduel (elles aident à
entraîner des réseaux profonds en facilitant la propagation du gradient,
via y = F(x) + x). Ce n'est PAS la même chose que les skip connections
de U-Net (qui relient l'encodeur au décodeur pour préserver les détails
spatiaux). Un ResNet n'a pas de décodeur du tout — c'est juste un
extracteur de features profond.

Astuce "output stride" : pour que DeepLabV3 garde une résolution de
features suffisante (nécessaire pour l'ASPP), le dernier étage du
backbone n'utilise PAS de stride pour réduire la résolution — il utilise
plutôt une dilation (atrous), comme vu dans la fiche théorique. Résultat :
la résolution spatiale n'est divisée que par 16 par rapport à l'image
d'entrée (au lieu de 32 pour un ResNet classique), tout en gardant un
grand champ réceptif.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Bloc résiduel de base (style ResNet18/34) :
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN, plus la connexion résiduelle
    (identité, ou une projection 1x1 si les dimensions changent), puis ReLU.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection pour aligner les dimensions de la branche résiduelle
        # quand stride != 1 ou que le nombre de canaux change.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # la connexion résiduelle : y = F(x) + x
        return self.relu(out)


class ResNetBackbone(nn.Module):
    """
    Backbone style ResNet18, from scratch, avec output_stride=16
    (via dilation sur le dernier étage plutôt qu'un stride supplémentaire).

    Sortie : carte de features à 512 canaux, résolution H/16 x W/16.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Stem : réduit déjà la résolution par 4 (conv stride2 + maxpool stride2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 étages, 2 blocs résiduels chacun (style ResNet18)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, dilation=1)   # /4
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, dilation=1)  # /8
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, dilation=1) # /16
        # Dernier étage : stride=1 + dilation=2 au lieu de stride=2
        # -> garde la résolution à /16 tout en élargissant le champ réceptif
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=1, dilation=2) # /16

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride, dilation):
        layers = [BasicBlock(in_channels, out_channels, stride=stride, dilation=dilation)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)     # (B, 64,  H/4,  W/4)
        x = self.layer1(x)   # (B, 64,  H/4,  W/4)
        x = self.layer2(x)   # (B, 128, H/8,  W/8)
        x = self.layer3(x)   # (B, 256, H/16, W/16)
        x = self.layer4(x)   # (B, 512, H/16, W/16)  <- champ réceptif élargi via dilation
        return x


if __name__ == "__main__":
    model = ResNetBackbone(in_channels=3)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print("Entrée :", dummy.shape)
    print("Sortie :", out.shape)  # attendu : torch.Size([2, 512, 16, 16])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Nombre de paramètres : {n_params:,}")