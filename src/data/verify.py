"""
Vérifie que le dataset ISIC est correctement téléchargé et organisé, et que
le pipeline data (splits figés + ISICDataset) fonctionne de bout en bout.

A exécuter après avoir suivi data/README.md, avant de commencer à coder
dessus — permet aux deux membres de l'équipe de confirmer que leur
environnement local (ou Colab) est prêt sans refaire les vérifications à la main.

Usage :
    python -m src.data.verify --config configs/config.yaml
"""
import argparse
import os

import yaml

from src.data.dataset import ISICDataset


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    images_dir = cfg["data"]["images_dir"]
    masks_dir = cfg["data"]["masks_dir"]
    image_size = cfg["data"]["image_size"]

    n_images = len(os.listdir(images_dir))
    n_masks = len(os.listdir(masks_dir))
    print(f"Images : {n_images} | Masques : {n_masks}")
    if n_images != n_masks:
        raise RuntimeError(
            "Nombre d'images et de masques différent — vérifie data/README.md."
        )

    for split in ["train", "val", "test"]:
        ds = ISICDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            split=split,
            image_size=image_size,
        )
        image, mask = ds[0]
        unique_mask_values = sorted(mask.unique().tolist())
        print(
            f"{split:>5} : {len(ds):4d} échantillons | "
            f"image {tuple(image.shape)} {image.dtype} | "
            f"mask {tuple(mask.shape)} {mask.dtype} | "
            f"valeurs mask {unique_mask_values}"
        )
        if unique_mask_values not in ([0, 1], [0], [1]):
            raise RuntimeError(
                f"Valeurs de masque inattendues pour le split {split} : "
                f"{unique_mask_values} (attendu : sous-ensemble de {{0, 1}})"
            )

    print("\nDataset OK — pipeline data prêt à l'emploi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
