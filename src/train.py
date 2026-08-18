"""
Boucle d'entraînement générique : lit une config (cf. configs/config.yaml),
entraîne le modèle/loss demandés, journalise métriques et checkpoints dans
experiments/<run_name>/, et met à jour experiments/results_summary.csv.

Reprise automatique : si un checkpoint existe déjà pour ce run, l'entraînement
reprend à l'epoch suivante plutôt que de repartir de zéro — protège contre une
session Colab interrompue (cf. discussion garde-fous, CHALLENGES.md).

Usage :
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --smoke-test   # validation rapide, CPU, sous-ensemble minuscule
"""
import argparse
import csv
import os

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ISICDataset
from src.losses import build_loss
from src.metrics import compute_confusion_stats, dice_from_stats, iou_from_stats
from src.models import build_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg: dict, smoke_test: bool) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_ds = ISICDataset(
        images_dir=data_cfg["images_dir"],
        masks_dir=data_cfg["masks_dir"],
        split="train",
        image_size=data_cfg["image_size"],
    )
    val_ds = ISICDataset(
        images_dir=data_cfg["images_dir"],
        masks_dir=data_cfg["masks_dir"],
        split="val",
        image_size=data_cfg["image_size"],
    )

    if smoke_test:
        # Sous-ensemble minuscule : le but est de valider que le pipeline
        # s'exécute de bout en bout, pas d'obtenir un modèle entraîné —
        # l'entraînement réel se fait sur Colab (GPU).
        train_ds = Subset(train_ds, range(min(8, len(train_ds))))
        val_ds = Subset(val_ds, range(min(4, len(val_ds))))

    batch_size = 2 if smoke_test else cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes: int) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    intersection_sum = torch.zeros(num_classes)
    union_sum = torch.zeros(num_classes)
    pred_area_sum = torch.zeros(num_classes)
    target_area_sum = torch.zeros(num_classes)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        inter, union, pred_area, target_area = compute_confusion_stats(preds.cpu(), masks.cpu(), num_classes)
        intersection_sum += inter
        union_sum += union
        pred_area_sum += pred_area
        target_area_sum += target_area

    iou = iou_from_stats(intersection_sum, union_sum)
    dice = dice_from_stats(intersection_sum, pred_area_sum, target_area_sum)
    return {
        "loss": total_loss / max(n_batches, 1),
        "iou_per_class": iou,
        "dice_per_class": dice,
        "mean_iou": iou.mean().item(),
        "mean_dice": dice.mean().item(),
    }


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer, device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]


def update_results_summary(
    run_name: str, architecture: str, loss_type: str, val_metrics: dict, results_path: str
) -> None:
    """Ajoute (ou remplace) la ligne de experiments/results_summary.csv pour ce run."""
    fieldnames = [
        "run_name", "architecture", "loss",
        "iou_lesion", "iou_background", "mean_iou",
        "dice_lesion", "dice_background", "mean_dice",
    ]

    rows = []
    if os.path.exists(results_path):
        with open(results_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row["run_name"] != run_name]

    rows.append({
        "run_name": run_name,
        "architecture": architecture,
        "loss": loss_type,
        "iou_lesion": f"{val_metrics['iou_per_class'][1].item():.4f}",
        "iou_background": f"{val_metrics['iou_per_class'][0].item():.4f}",
        "mean_iou": f"{val_metrics['mean_iou']:.4f}",
        "dice_lesion": f"{val_metrics['dice_per_class'][1].item():.4f}",
        "dice_background": f"{val_metrics['dice_per_class'][0].item():.4f}",
        "mean_dice": f"{val_metrics['mean_dice']:.4f}",
    })

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(config_path: str, smoke_test: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run_name"]
    num_classes = cfg["model"]["num_classes"]
    device = get_device()
    print(f"Run: {run_name} | device: {device} | smoke_test: {smoke_test}")

    train_loader, val_loader = build_dataloaders(cfg, smoke_test)

    model = build_model(cfg["model"]["architecture"], num_classes=num_classes, encoder=cfg["model"]["encoder"])
    model.to(device)

    loss_fn = build_loss(cfg["loss"], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    experiment_dir = os.path.join(cfg["output"]["experiment_dir"], run_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoint_path = os.path.join(experiment_dir, "checkpoint.pt")
    metrics_log_path = os.path.join(experiment_dir, "metrics.csv")

    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch += 1
        print(f"Reprise depuis le checkpoint : epoch {start_epoch}")

    n_epochs = 2 if smoke_test else cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    epochs_without_improvement = 0

    log_is_new = not os.path.exists(metrics_log_path)
    with open(metrics_log_path, "a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        if log_is_new:
            log_writer.writerow(["epoch", "train_loss", "val_loss", "val_mean_iou", "val_mean_dice"])

        for epoch in range(start_epoch, n_epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / max(n_batches, 1)
            val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes)

            print(
                f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | val_mean_iou {val_metrics['mean_iou']:.4f}"
            )
            log_writer.writerow([
                epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}",
                f"{val_metrics['mean_iou']:.4f}", f"{val_metrics['mean_dice']:.4f}",
            ])
            log_file.flush()

            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(experiment_dir, "best_model.pt"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping : pas d'amélioration depuis {patience} epochs.")
                    break

    final_val_metrics = evaluate(model, val_loader, loss_fn, device, num_classes)
    results_path = os.path.join(cfg["output"]["experiment_dir"], "results_summary.csv")
    update_results_summary(run_name, cfg["model"]["architecture"], cfg["loss"]["type"], final_val_metrics, results_path)
    print(f"Run terminé. Résultats écrits dans {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Test rapide sur un sous-ensemble minuscule, en local/CPU (valide le pipeline, pas la qualité du modèle)",
    )
    args = parser.parse_args()
    main(args.config, smoke_test=args.smoke_test)
