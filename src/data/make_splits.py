"""
Génère une bonne fois pour toutes les listes de fichiers train/val/test,
figées dans des fichiers .txt versionnés sur Git.

Pourquoi ce script existe : l'étude d'ablation compare 6 runs (2 architectures
x 3 losses). Pour que la comparaison soit valide, les 6 runs doivent
s'entraîner et s'évaluer EXACTEMENT sur les mêmes images. Ce script fixe donc
le split une seule fois ; tous les runs le relisent ensuite via
`src/data/dataset.py`.

Usage :
    python -m src.data.make_splits --config configs/config.yaml
"""
import argparse
import glob
import os

import yaml
from sklearn.model_selection import train_test_split


def get_stems(images_dir: str) -> list[str]:
    """Retourne les noms de fichiers (sans extension) de toutes les images."""
    paths = sorted(glob.glob(os.path.join(images_dir, "*")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg["data"]["images_dir"]
    val_split = cfg["data"]["val_split"]
    test_split = cfg["data"]["test_split"]
    seed = cfg["data"]["seed"]

    stems = get_stems(images_dir)
    if not stems:
        raise RuntimeError(
            f"Aucune image trouvée dans {images_dir}. "
            "As-tu bien téléchargé le dataset (voir data/README.md) ?"
        )

    # 1) on sépare d'abord le test set
    train_val_stems, test_stems = train_test_split(
        stems, test_size=test_split, random_state=seed
    )
    # 2) puis le val set, à partir de ce qui reste (proportion recalculée)
    relative_val_size = val_split / (1 - test_split)
    train_stems, val_stems = train_test_split(
        train_val_stems, test_size=relative_val_size, random_state=seed
    )

    splits_dir = "data/processed/splits"
    os.makedirs(splits_dir, exist_ok=True)

    for name, split_stems in [
        ("train", train_stems),
        ("val", val_stems),
        ("test", test_stems),
    ]:
        out_path = os.path.join(splits_dir, f"{name}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(sorted(split_stems)))
        print(f"{name}: {len(split_stems)} images -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
