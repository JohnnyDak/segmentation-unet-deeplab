"""
Métriques de segmentation : IoU et Dice score, calculés PAR CLASSE
(exigence du cahier des charges — pas seulement une moyenne globale).

Les métriques sont accumulées sur une epoch entière (somme des
intersections/unions batch par batch), puis calculées une seule fois à
la fin. C'est plus correct que de moyenner des métriques calculées batch
par batch, car ça évite qu'un petit batch avec peu de pixels de la
classe minoritaire ne fausse la moyenne (micro-average plutôt que
macro-average naïf).
"""
import torch


class SegmentationMetrics:
    """Accumulateur d'IoU/Dice par classe sur une epoch.

    Usage :
        metrics = SegmentationMetrics(num_classes=2, class_names=["fond", "lésion"])
        for batch in loader:
            logits = model(images)
            metrics.update(logits, masks)
        results = metrics.compute()   # dict avec iou/dice par classe + moyennes
        metrics.reset()               # avant l'epoch suivante
    """

    def __init__(self, num_classes: int = 2, class_names=None, eps: float = 1e-7):
        self.num_classes = num_classes
        self.class_names = class_names or [f"classe_{i}" for i in range(num_classes)]
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.cardinality = torch.zeros(self.num_classes)  # somme(pred) + somme(target), pour Dice

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """logits : (B, C, H, W) — targets : (B, H, W) avec indices de classe."""
        preds = torch.argmax(logits, dim=1)  # (B, H, W)

        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)

            self.intersection[c] += (pred_c & target_c).sum().float().cpu()
            self.union[c] += (pred_c | target_c).sum().float().cpu()
            self.cardinality[c] += pred_c.sum().float().cpu() + target_c.sum().float().cpu()

    def compute(self) -> dict:
        iou_per_class = self.intersection / (self.union + self.eps)
        dice_per_class = (2 * self.intersection) / (self.cardinality + self.eps)

        results = {"mean_iou": iou_per_class.mean().item(), "mean_dice": dice_per_class.mean().item()}
        for i, name in enumerate(self.class_names):
            results[f"iou_{name}"] = iou_per_class[i].item()
            results[f"dice_{name}"] = dice_per_class[i].item()
        return results


if __name__ == "__main__":
    # Test rapide avec des tenseurs factices.
    metrics = SegmentationMetrics(num_classes=2, class_names=["fond", "lesion"])

    logits = torch.randn(4, 2, 32, 32)
    targets = torch.randint(0, 2, (4, 32, 32))
    metrics.update(logits, targets)

    results = metrics.compute()
    for k, v in results.items():
        print(f"{k:<15} : {v:.4f}")