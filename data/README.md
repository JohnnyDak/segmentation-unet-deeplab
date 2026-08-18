# Dataset ISIC — instructions de téléchargement

Le dataset n'est **pas** inclus dans ce repo (trop volumineux). Voici comment
se le procurer directement depuis le portail officiel ISIC Archive.

## Dataset retenu : ISIC 2017 — Task 1 (segmentation)

2000 images d'entraînement, licence **CC-0**. Volume suffisant pour un
entraînement from scratch (sans poids pré-entraînés) en segmentation
binaire — plusieurs études publiées obtiennent de bons résultats avec un
volume comparable, voir la discussion dans le rapport d'avancement.

- 2000 images d'entraînement (JPEG) + 2000 masques binaires (PNG)
- 600 images de test + masques de test (optionnel, pour une évaluation
  finale hors validation)
- ⚠️ Poids important : ~5.8 Go pour les données d'entraînement (le zip
  contient aussi des masques de superpixels qu'on n'utilise pas — seules
  les images .jpg en sont extraites)

### Téléchargement (à exécuter dans le notebook Colab)

```bash
mkdir -p data/raw/images data/raw/masks

# Images d'entraînement (contient aussi des masques de superpixels, ignorés)
wget -q https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Data.zip
# Masques de segmentation (la vraie vérité terrain)
wget -q https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip

unzip -q ISIC-2017_Training_Data.zip -d images_tmp
unzip -q ISIC-2017_Training_Part1_GroundTruth.zip -d masks_tmp

# On ne garde que les .jpg (les superpixels .png du zip d'images sont ignorés)
find images_tmp -name '*.jpg' -exec mv {} data/raw/images/ \;
find masks_tmp -name '*.png' -exec mv {} data/raw/masks/ \;
rm -rf images_tmp masks_tmp *.zip
```

### Test set (optionnel)

```bash
wget -q https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Data.zip
wget -q https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
```

## Pourquoi ISIC 2017 plutôt que 2016 ?

Le dataset ISIC 2016 (900 images) a d'abord été envisagé, mais comme le
projet impose un entraînement **from scratch** (aucun poids pré-entraîné,
y compris pour le backbone ResNet de DeepLabV3), 2000 images offre une
marge de sécurité plus confortable contre le surapprentissage.

## Alternatives

| Édition | Images train | Poids train | Licence |
|---|---|---|---|
| ISIC 2016 Task 1 | 900 | 602 Mo | CC-0 |
| **ISIC 2017 Task 1 (retenu)** | 2000 | 5.8 Go | CC-0 |
| ISIC 2018 Task 1 | 2594 (5 masques/image) | 10.4 Go | CC-0 |

Tous les liens sont sur https://challenge.isic-archive.com/data/

## Citation (à inclure dans le rapport)

> Codella N, Gutman D, Celebi ME, Helba B, Marchetti MA, Dusza S, Kalloo A,
> Liopyris K, Mishra N, Kittler H, Halpern A. "Skin Lesion Analysis Toward
> Melanoma Detection: A Challenge at the 2017 International Symposium on
> Biomedical Imaging (ISBI), Hosted by the International Skin Imaging
> Collaboration (ISIC)". arXiv:1710.05006 [cs.CV]

## Structure attendue après téléchargement

```
data/raw/
  images/        *.jpg — images originales des lésions
  masks/         *.png — masques binaires (0 = peau saine, 255 = lésion)
```

⚠️ **Convention de nommage des masques** : elle varie légèrement selon
l'édition ISIC (ex. `_Segmentation.png` en 2016, `_segmentation.png` en
2017). Le `ISICDataset` (`src/data/dataset.py`) gère cette variation
automatiquement en recherchant le masque par préfixe plutôt que par
suffixe exact.
