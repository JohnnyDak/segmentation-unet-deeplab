"""
Dataset PyTorch pour ISIC (segmentation binaire lésion/fond) et transforms
albumentations associées.

Point clé : albumentations applique la MEME transformation géométrique
(flip, rotation...) à l'image et à son masque simultanément, via
`transform(image=..., mask=...)`. C'est indispensable en segmentation :
si l'image tourne mais pas le masque, les deux ne correspondent plus.
"""
import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Statistiques ImageNet : utilisées car les encodeurs (ResNet, etc.)
# sont pré-entraînés sur ImageNet et attendent des images normalisées ainsi.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: str, image_size: int = 256) -> A.Compose:
    """
    Retourne les transforms albumentations pour un split donné.

    - "train" : augmentations géométriques/photométriques + resize + normalisation
    - "val" / "test" : uniquement resize + normalisation (pas d'augmentation,
      on veut évaluer sur les images "telles quelles")
    """
    if split == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # Rotation libre légère : une lésion n'a pas d'orientation
                # canonique, mais on limite l'ampleur pour ne pas déformer
                # excessivement sa forme (importante cliniquement).
                A.Affine(rotate=(-15, 15), p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.3
                ),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class ISICDataset(Dataset):
    """
    Dataset ISIC (segmentation binaire).

    Lit les listes de fichiers figées par `src/data/make_splits.py`
    (data/processed/splits/<split>.txt) — jamais de split aléatoire ici,
    pour garantir que les 6 runs de l'ablation utilisent les mêmes images.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        split: str,
        splits_dir: str = "data/processed/splits",
        image_size: int = 256,
        transform: A.Compose | None = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        split_file = os.path.join(splits_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"{split_file} introuvable. "
                "As-tu exécuté `python -m src.data.make_splits` ?"
            )
        with open(split_file) as f:
            self.stems = [line.strip() for line in f if line.strip()]

        self.transform = transform or get_transforms(split, image_size)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        stem = self.stems[idx]

        image_path = os.path.join(self.images_dir, f"{stem}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{stem}_Segmentation.png")

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Masque binaire ISIC : 0 = fond, 255 = lésion -> on ramène à {0, 1}
        mask = (mask > 127).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]  # (3, H, W), float, normalisé
        mask_tensor = transformed["mask"].long()  # (H, W), classes 0/1

        return image_tensor, mask_tensor
