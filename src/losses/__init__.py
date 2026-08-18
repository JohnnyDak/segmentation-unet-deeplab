"""
Fonctions de perte pour la segmentation binaire (lésion vs fond) : Dice loss,
Cross-Entropy, et leur combinaison pondérée. C'est le cœur de l'étude
d'ablation du sujet (cf. ROADMAP.md, étape 2) — leur effet sur la classe
minoritaire (lésion, ~18 % des pixels, cf. CHALLENGES.md) est ce qu'on
compare.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss multi-classe (moyenne sur les classes), sur les probabilités softmax."""

    def __init__(self, num_classes: int = 2, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (N, C, H, W)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)  # (N, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        union = probs.sum(dims) + targets_onehot.sum(dims)
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_per_class.mean()


class ComboLoss(nn.Module):
    """Combinaison pondérée Dice + Cross-Entropy (poids configurables, cf. config.yaml)."""

    def __init__(self, num_classes: int = 2, weights: tuple[float, float] = (0.5, 0.5)):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.w_dice, self.w_ce = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, targets) + self.w_ce * self.ce(logits, targets)


def build_loss(loss_cfg: dict, num_classes: int) -> nn.Module:
    """Construit la loss à partir de la section `loss` de configs/config.yaml."""
    loss_type = loss_cfg["type"]
    if loss_type == "dice":
        return DiceLoss(num_classes=num_classes)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_type == "combo":
        return ComboLoss(num_classes=num_classes, weights=tuple(loss_cfg["combo_weights"]))
    raise ValueError(f"Loss inconnue : {loss_type!r} (attendu : 'dice' | 'cross_entropy' | 'combo')")
