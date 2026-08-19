"""
Reconstruit le tableau récapitulatif détaillé de l'étude d'ablation :
IoU et Dice PAR CLASSE (fond / lésion), pas seulement la moyenne
globale — c'est ce qui permet de visualiser l'effet de chaque loss sur
la classe minoritaire (lésion), exigé par le cahier des charges.

Pour chaque run, va chercher les métriques à l'EPOCH OÙ val_mean_dice
était maximal (pas la dernière epoch loggée, qui n'est pas forcément la
meilleure à cause de l'early stopping — patience=8 veut dire que les 8
dernières epochs loggées sont, par définition, moins bonnes que le pic).

Usage :
    python -m src.build_results_table --config configs/config.yaml
"""
import argparse

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

from src.train import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

ARCHITECTURES = ["unet", "deeplabv3"]
LOSSES = ["dice", "cross_entropy", "combo"]
METRIC_KEYS = ["val_iou_fond", "val_iou_lesion", "val_dice_fond", "val_dice_lesion"]


def get_metrics_at_best_epoch(client: MlflowClient, run_id: str) -> dict:
    """Trouve l'epoch (step) où val_mean_dice était maximal, puis
    récupère les métriques par classe à cette même epoch précise."""
    dice_history = client.get_metric_history(run_id, "val_mean_dice")
    best_entry = max(dice_history, key=lambda m: m.value)
    best_step = best_entry.step

    result = {"best_epoch": best_step + 1, "val_mean_dice": best_entry.value}
    for key in METRIC_KEYS:
        history = client.get_metric_history(run_id, key)
        matching = [m.value for m in history if m.step == best_step]
        result[key] = matching[0] if matching else None
    return result


def main(config_path: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    rows = []
    for architecture in ARCHITECTURES:
        for loss_type in LOSSES:
            run_name = f"{architecture}_{loss_type}"
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}' and status = 'FINISHED'",
                order_by=["start_time DESC"],
            )
            if len(runs) == 0:
                print(f"⚠️  Aucun run trouvé pour {run_name}")
                continue

            run_id = runs.iloc[0]["run_id"]
            metrics = get_metrics_at_best_epoch(client, run_id)
            rows.append({
                "architecture": architecture,
                "loss": loss_type,
                "best_epoch": metrics["best_epoch"],
                "iou_fond": round(metrics["val_iou_fond"], 4),
                "iou_lesion": round(metrics["val_iou_lesion"], 4),
                "dice_fond": round(metrics["val_dice_fond"], 4),
                "dice_lesion": round(metrics["val_dice_lesion"], 4),
                "mean_dice": round(metrics["val_mean_dice"], 4),
            })

    df = pd.DataFrame(rows)
    out_path = "experiments/results_summary_detailed.csv"
    df.to_csv(out_path, index=False)

    print(f"\nTableau détaillé écrit dans {out_path}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)