"""
Fonctions de perte pour l'étude d'ablation : Dice loss, Cross-Entropy,
et leur combinaison.

Choix méthodologique important : la Cross-Entropy est laissée NON
PONDÉRÉE, volontairement. Le but de l'ablation est de démontrer que la
CE standard souffre face au déséquilibre de classes (28% lésion / 72%
fond), par contraste avec Dice qui gère mieux la classe minoritaire.
Pondérer la CE effacerait ce contraste et fausserait la comparaison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss multi-classe (calculée par classe puis moyennée).

    Le score de Dice mesure le chevauchement entre la prédiction et la
    vérité terrain : Dice = 2*|A∩B| / (|A|+|B|). La loss est 1 - Dice
    (on minimise). Contrairement à la Cross-Entropy (qui traite chaque
    pixel indépendamment), Dice raisonne au niveau de la région entière
    prédite — ça la rend naturellement moins sensible au déséquilibre de
    classes : une petite région bien prédite compte autant qu'une grande.
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth  # évite la division par zéro si une classe est absente

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits : (B, C, H, W) ; targets : (B, H, W) avec des indices de classe
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        dims = (0, 2, 3)  # somme sur batch + spatial, garde la dimension classe
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class ComboLoss(nn.Module):
    """Combinaison pondérée de Dice loss et de Cross-Entropy (non pondérée).
    combo_weights = (poids_dice, poids_ce), doit sommer à 1.0 en général."""

    def __init__(self, num_classes: int = 2, weights=(0.5, 0.5)):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss()  # non pondérée, volontairement (voir docstring du module)
        self.w_dice, self.w_ce = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_dice * self.dice(logits, targets) + self.w_ce * self.ce(logits, targets)


def get_loss(loss_type: str, num_classes: int = 2, combo_weights=(0.5, 0.5)) -> nn.Module:
    """Factory : instancie la loss à partir de la valeur `loss.type` de config.yaml."""
    if loss_type == "dice":
        return DiceLoss(num_classes=num_classes)
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()  # non pondérée
    if loss_type == "combo":
        return ComboLoss(num_classes=num_classes, weights=combo_weights)
    raise ValueError(f"loss_type inconnu : {loss_type!r} (attendu : dice | cross_entropy | combo)")


if __name__ == "__main__":
    # Test rapide : logits/targets factices, vérifie que les 3 losses
    # tournent sans erreur et donnent une valeur scalaire positive.
    logits = torch.randn(2, 2, 64, 64)
    targets = torch.randint(0, 2, (2, 64, 64))

    for loss_type in ["dice", "cross_entropy", "combo"]:
        loss_fn = get_loss(loss_type, num_classes=2)
        value = loss_fn(logits, targets)
        print(f"{loss_type:<15} -> {value.item():.4f}")