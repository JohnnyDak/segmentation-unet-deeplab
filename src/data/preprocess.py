"""
Pré-traite le dataset une bonne fois pour toutes : redimensionne toutes
les images et masques vers la résolution d'entraînement, et les sauvegarde
dans data/processed/. Le Dataset PyTorch lira ensuite ces fichiers légers
au lieu de décoder des JPEG parfois énormes (jusqu'à 17 Mo) à chaque epoch.

Pourquoi ce script existe : le dataset ISIC 2017 n'est pas standardisé en
résolution (de 576x767 à plus de 4400x6600 pixels selon le dermatoscope
d'origine). Comme le modèle ne voit de toute façon que des images 256x256
(voir get_transforms() dans dataset.py), redimensionner à la volée à
chaque epoch gaspille énormément de temps de calcul et de bande passante
Google Drive, sans aucun bénéfice.

Usage :
    python -m src.data.preprocess --config configs/config.yaml
"""
import argparse
import glob
import os

import cv2
import yaml
from tqdm import tqdm


def find_mask_path(masks_dir: str, stem: str) -> str | None:
    """Recherche le masque par préfixe (le suffixe varie selon l'édition ISIC)."""
    candidates = glob.glob(os.path.join(masks_dir, f"{stem}*.png"))
    return candidates[0] if candidates else None


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Source fixe : toujours data/raw/ (les images originales telles que
    # téléchargées). Volontairement PAS lu depuis config["data"]["images_dir"],
    # qui pointe vers data/processed/ — c'est la destination finale pour
    # l'entraînement, pas la source du pré-traitement.
    raw_images_dir = "data/raw/images"
    raw_masks_dir = "data/raw/masks"
    image_size = cfg["data"]["image_size"]

    out_images_dir = "data/processed/images"
    out_masks_dir = "data/processed/masks"
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_masks_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(raw_images_dir, "*.jpg")))
    if not image_paths:
        raise RuntimeError(f"Aucune image trouvée dans {raw_images_dir}")

    n_ok, n_skipped = 0, 0
    for image_path in tqdm(image_paths, desc="Pré-traitement"):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = find_mask_path(raw_masks_dir, stem)
        if mask_path is None:
            print(f"⚠️  Masque introuvable pour {stem}, image ignorée.")
            n_skipped += 1
            continue

        # Image : interpolation INTER_AREA, la plus adaptée pour réduire
        # la résolution (évite l'aliasing par rapport à un simple bilinéaire).
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_images_dir, f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Masque : interpolation NEAREST obligatoire (jamais bilinéaire)
        # pour ne pas créer de valeurs intermédiaires entre 0 et 255 —
        # un masque de segmentation doit rester strictement binaire.
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_masks_dir, f"{stem}.png"), mask)

        n_ok += 1

    print(f"\nTerminé : {n_ok} paires traitées, {n_skipped} ignorées (masque manquant).")
    print(f"Images -> {out_images_dir}")
    print(f"Masques -> {out_masks_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)