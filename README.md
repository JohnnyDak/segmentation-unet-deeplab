# Segmentation sémantique : U-Net vs DeepLabV3 — étude d'ablation

Projet de fin d'année — Master 1 IA & Big Data (ESGIS).

Comparaison de deux architectures de segmentation sémantique (**U-Net** et
**DeepLabV3**) sur le dataset médical **ISIC** (segmentation de lésions
cutanées), avec une étude d'ablation sur le choix de la fonction de perte
(Dice loss vs Cross-Entropy vs combinaison) et son effet sur la classe
minoritaire (lésion).

## Références

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image
  Segmentation* (2015) — https://arxiv.org/abs/1505.04597
- Chen et al., *Rethinking Atrous Convolution for Semantic Image
  Segmentation* (DeepLabV3, 2017) — https://arxiv.org/abs/1706.05587

## Dataset

**ISIC 2016 — Task 1 (segmentation)** — 900 images d'entraînement + masques
binaires, licence CC-0. Portail officiel :
https://challenge.isic-archive.com/data/

Le dataset **n'est pas versionné sur Git** (trop volumineux). Voir
[`data/README.md`](data/README.md) pour les liens de téléchargement directs
et les alternatives (2017, 2018).

## Structure du repo

```
configs/          Fichiers de configuration (hyperparamètres par run)
data/
  raw/             Dataset ISIC brut (non versionné)
  processed/       Données après preprocessing (non versionné)
notebooks/         Exploration, EDA, visualisation, notebooks Colab
src/
  data/            Dataset PyTorch, transforms, augmentation (albumentations)
  models/          U-Net (implémentation) + wrapper DeepLabV3 (segmentation_models_pytorch)
  losses/          Dice loss, Cross-Entropy, loss combinée
  metrics/         IoU / Dice score par classe
  train.py         Script d'entraînement
  evaluate.py       Script d'évaluation
experiments/       Résultats, logs, courbes, poids par run (non versionné,
                   sauf le résumé results_summary.csv)
app/               Application web de démonstration
reports/           Figures, tableaux d'ablation, rapport final
```

## Environnement de travail (Google Colab / Kaggle)

Ce projet est pensé pour tourner sur **Google Colab** (pas de GPU local
requis). Voir le notebook [`notebooks/00_setup_and_first_look.ipynb`](notebooks/00_setup_and_first_look.ipynb)
qui :

1. installe les dépendances,
2. monte Google Drive (pour persister le dataset et les checkpoints entre
   les sessions Colab),
3. télécharge et organise le dataset ISIC 2016 Task 1 directement depuis
   le portail officiel,
4. fait un premier contact visuel avec les images et les masques.

### Installation locale (optionnelle, pour lire/tester le code hors Colab)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Suivi des expériences (étude d'ablation)

Chaque run est nommé selon le schéma `<architecture>_<loss>`, par exemple :

- `unet_dice`, `unet_ce`, `unet_combo`
- `deeplab_dice`, `deeplab_ce`, `deeplab_combo`

Chaque run sauvegarde ses métriques (IoU/Dice par classe), ses courbes de
loss et quelques prédictions visuelles dans `experiments/<nom_du_run>/`.
Le tableau récapitulatif final se trouve dans
`experiments/results_summary.csv`.

## Auteur

Étudiant en Master 1 IA & Big Data — ESGIS.
