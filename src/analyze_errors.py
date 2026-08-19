"""
Analyse qualitative des erreurs de segmentation, pour chaque architecture.

Démarche :
1. Charge les modèles en "Production" (Model Registry MLflow).
2. Fait l'inférence sur tout le test set (jamais vu à l'entraînement).
3. Calcule un Dice PAR IMAGE (pas une moyenne globale) pour repérer les
   pires cas de chaque architecture.
4. Catégorise chaque échec via une heuristique :
   - "bord imprécis" : la majorité des pixels d'erreur sont proches du
     contour de la vérité terrain (bande dilatée autour du contour).
   - "petites régions fragmentées" : le modèle prédit plus de composantes
     connexes que la vérité terrain (faux positifs isolés, "bruit").
   - "sous-segmentation" : le modèle manque une région entière (aucun de
     ces deux cas ne domine).
5. Sauvegarde des grilles de visualisation (image / vérité terrain /
   prédiction) pour les pires cas de chaque architecture.

Usage :
    python -m src.analyze_errors --config configs/config.yaml --top-k 6
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import ISICDataset
from src.train import MLFLOW_TRACKING_URI

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def unnormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Inverse la normalisation ImageNet pour ré-afficher l'image telle
    qu'elle apparaît réellement (pour la visualisation uniquement)."""
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(pred, target).sum()
    return (2 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)


def classify_failure(gt: np.ndarray, pred: np.ndarray) -> str:
    """Heuristique de catégorisation du type d'échec dominant."""
    error_mask = np.logical_xor(gt, pred).astype(np.uint8)
    if error_mask.sum() == 0:
        return "aucune erreur notable"

    # Bande autour du contour de la vérité terrain (érosion + dilatation
    # pour isoler la frontière, puis élargissement de quelques pixels)
    gt_uint8 = gt.astype(np.uint8)
    contour_band = cv2.dilate(gt_uint8, np.ones((7, 7), np.uint8)) - cv2.erode(
        gt_uint8, np.ones((7, 7), np.uint8)
    )
    boundary_error_fraction = np.logical_and(error_mask, contour_band).sum() / error_mask.sum()

    n_components_gt, _ = cv2.connectedComponents(gt_uint8)
    n_components_pred, _ = cv2.connectedComponents(pred.astype(np.uint8))

    if boundary_error_fraction > 0.6:
        return "bord imprécis (contour décalé)"
    if n_components_pred > n_components_gt:
        return "petites régions fragmentées (faux positifs isolés)"
    return "sous-segmentation (région manquée)"


def load_production_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
    return model.to(device).eval()


def evaluate_all(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[dict]:
    """Calcule le Dice (classe lésion) pour chaque image du test set."""
    results = []
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            gt = masks.numpy()[0]

            results.append({
                "index": idx,
                "dice_lesion": dice_score(preds == 1, gt == 1),
                "pred_mask": preds,
                "gt_mask": gt,
                "image_tensor": images[0].cpu(),
            })
    return results


def save_worst_cases(results: list[dict], architecture: str, top_k: int, out_dir: str) -> None:
    worst = sorted(results, key=lambda r: r["dice_lesion"])[:top_k]

    fig, axes = plt.subplots(top_k, 3, figsize=(9, 3 * top_k))
    for i, case in enumerate(worst):
        img = unnormalize(case["image_tensor"])
        gt, pred = case["gt_mask"], case["pred_mask"]
        failure_type = classify_failure(gt == 1, pred == 1)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image (dice={case['dice_lesion']:.3f})")
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(gt, cmap="Greens", alpha=0.5)
        axes[i, 1].set_title("Vérité terrain")
        axes[i, 2].imshow(img)
        axes[i, 2].imshow(pred, cmap="Reds", alpha=0.5)
        axes[i, 2].set_title(f"Prédiction\n{failure_type}", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"{architecture} — {top_k} pires cas sur le test set", fontsize=14)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"failure_cases_{architecture}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Sauvegardé : {out_path}")

    print(f"\n--- {architecture} : catégories d'échec (top {top_k}) ---")
    for case in worst:
        failure_type = classify_failure(case["gt_mask"] == 1, case["pred_mask"] == 1)
        print(f"  image #{case['index']:<4} dice={case['dice_lesion']:.3f}  -> {failure_type}")


def main(config_path: str, top_k: int) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    data_cfg = cfg["data"]
    test_ds = ISICDataset(data_cfg["images_dir"], data_cfg["masks_dir"], split="test",
                           image_size=data_cfg["image_size"])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    out_dir = "reports/error_analysis"

    for architecture, model_name in [("unet", "unet-lesion-segmentation"),
                                      ("deeplabv3", "deeplabv3-lesion-segmentation")]:
        print(f"\n=== {architecture} : inférence sur {len(test_ds)} images de test ===")
        model = load_production_model(model_name, device)
        results = evaluate_all(model, test_loader, device)

        mean_dice_lesion = np.mean([r["dice_lesion"] for r in results])
        print(f"Dice lésion moyen sur le test set : {mean_dice_lesion:.4f}")

        save_worst_cases(results, architecture, top_k, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    main(args.config, args.top_k)