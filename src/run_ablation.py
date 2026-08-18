"""
Lance automatiquement les 6 runs de l'étude d'ablation :
2 architectures (unet, deeplabv3) x 3 losses (dice, cross_entropy, combo).

Chaque run réutilise train_from_config() de src/train.py, avec seulement
`architecture`, `loss.type` et `run_name` modifiés en mémoire — pas
besoin de dupliquer configs/config.yaml en 6 fichiers séparés.

À la fin, écrit un résumé dans experiments/results_summary.csv (le
tableau demandé dans les livrables), en complément du suivi détaillé
dans MLflow.

Usage :
    python -m src.run_ablation --config configs/config.yaml
"""
import argparse
import copy
import csv
import os

import yaml

from src.train import train_from_config

ARCHITECTURES = ["unet", "deeplabv3"]
LOSSES = ["dice", "cross_entropy", "combo"]


def main(config_path: str) -> None:
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    results = []

    for architecture in ARCHITECTURES:
        for loss_type in LOSSES:
            run_name = f"{architecture}_{loss_type}"
            print(f"\n{'='*60}\nRun : {run_name}\n{'='*60}")

            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["architecture"] = architecture
            cfg["loss"]["type"] = loss_type
            cfg["run_name"] = run_name

            result = train_from_config(cfg)
            results.append(result)

    # --- Écrit le tableau récapitulatif (livrable du projet) ---
    os.makedirs("experiments", exist_ok=True)
    summary_path = "experiments/results_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "architecture", "loss", "best_val_mean_dice", "model_version"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"Les 6 runs sont terminés. Résumé écrit dans {summary_path}")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['run_name']:<25} val_mean_dice={r['best_val_mean_dice']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)