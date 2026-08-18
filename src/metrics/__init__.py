"""
Métriques de segmentation : IoU et Dice, calculés PAR CLASSE (pas seulement
la moyenne globale) — exigence explicite du sujet, particulièrement
pertinente ici puisque les classes sont déséquilibrées (~18 % lésion /
82 % fond, cf. CHALLENGES.md).
"""
import torch


@torch.no_grad()
def compute_confusion_stats(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    preds, targets : (N, H, W) int64, indices de classe (pas des logits).

    Retourne intersection/union/aires par classe, sommées sur le batch —
    permet d'accumuler sur plusieurs batches avant de calculer IoU/Dice
    finaux sur un epoch entier plutôt que de moyenner des moyennes de batch.
    """
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    pred_area = torch.zeros(num_classes)
    target_area = torch.zeros(num_classes)

    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        intersection[c] = (pred_c & target_c).sum()
        union[c] = (pred_c | target_c).sum()
        pred_area[c] = pred_c.sum()
        target_area[c] = target_c.sum()

    return intersection, union, pred_area, target_area


def iou_from_stats(intersection: torch.Tensor, union: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return (intersection + eps) / (union + eps)


def dice_from_stats(
    intersection: torch.Tensor, pred_area: torch.Tensor, target_area: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    return (2 * intersection + eps) / (pred_area + target_area + eps)


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """Calcule IoU et Dice par classe sur un batch, à partir des logits du modèle."""
    preds = logits.argmax(dim=1)
    intersection, union, pred_area, target_area = compute_confusion_stats(preds, targets, num_classes)

    iou = iou_from_stats(intersection, union)
    dice = dice_from_stats(intersection, pred_area, target_area)

    return {
        "iou_per_class": iou,
        "dice_per_class": dice,
        "mean_iou": iou.mean().item(),
        "mean_dice": dice.mean().item(),
    }
