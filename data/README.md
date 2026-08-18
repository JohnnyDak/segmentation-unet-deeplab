# Dataset ISIC — instructions de téléchargement

Le dataset n'est **pas** inclus dans ce repo (trop volumineux). Voici comment
se le procurer directement depuis le portail officiel ISIC Archive.

## Dataset recommandé : ISIC 2016 — Task 1 (segmentation)

Le plus léger des challenges de segmentation ISIC, licence **CC-0** (usage
libre, aucune restriction) :

- 900 images d'entraînement (JPEG) + 900 masques binaires (PNG)
- 379 images de test + masques de test
- Poids total raisonnable pour Colab (~840 Mo au total)

### Téléchargement (à exécuter dans le notebook Colab, ou en local)

```bash
mkdir -p data/raw/images data/raw/masks

# Images et masques d'entraînement
wget -q https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_Data.zip
wget -q https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip

unzip -q ISBI2016_ISIC_Part1_Training_Data.zip -d data/raw/images_tmp
unzip -q ISBI2016_ISIC_Part1_Training_GroundTruth.zip -d data/raw/masks_tmp

mv data/raw/images_tmp/*/*.jpg data/raw/images/ 2>/dev/null || mv data/raw/images_tmp/*.jpg data/raw/images/
mv data/raw/masks_tmp/*/*.png data/raw/masks/ 2>/dev/null || mv data/raw/masks_tmp/*.png data/raw/masks/
rm -rf data/raw/images_tmp data/raw/masks_tmp *.zip
```

> **Pas de `wget` en local (ex. Git Bash sur Windows) ?** Remplace les deux
> lignes `wget -q <url>` par `curl -sS -o <nom_du_fichier.zip> <url>` (même
> ordre d'arguments inversé : `-o` avant l'URL). Le reste du script ne change pas.

### Vérifier que tout est en place

Une fois le dataset téléchargé et `python -m src.data.make_splits` exécuté
(génère les splits, déjà versionnés dans `data/processed/splits/` — inutile
de le relancer sauf si tu ajoutes des données), vérifie que le pipeline
fonctionne de bout en bout :

```bash
python -m src.data.verify --config configs/config.yaml
```

Ça confirme : nombre d'images/masques cohérent, et chaque split (train/val/test)
se charge correctement via `ISICDataset` (bonnes dimensions, masque binaire
{0, 1}).

### Test set (optionnel, pour une évaluation finale hors validation)

```bash
wget -q https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_Data.zip
wget -q https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip
```

## Alternatives (si tu veux plus de données)

| Édition | Images train | Poids train | Licence |
|---|---|---|---|
| ISIC 2016 Task 1 (recommandé) | 900 | 602 Mo | CC-0 |
| ISIC 2017 Task 1 | 2000 | 5.8 Go | CC-0 |
| ISIC 2018 Task 1 | 2594 (5 masques/image) | 10.4 Go | CC-0 |

Tous les liens sont sur https://challenge.isic-archive.com/data/

## Citation (à inclure dans ton rapport si tu utilises ISIC 2016)

> Gutman, David; Codella, Noel C. F.; Celebi, Emre; Helba, Brian;
> Marchetti, Michael; Mishra, Nabin; Halpern, Allan. "Skin Lesion Analysis
> toward Melanoma Detection: A Challenge at the International Symposium on
> Biomedical Imaging (ISBI) 2016, hosted by the International Skin Imaging
> Collaboration (ISIC)". arXiv:1605.01397. 2016.

## Structure attendue après téléchargement

```
data/raw/
  images/        *.jpg — images originales des lésions
  masks/         *.png — masques binaires (0 = peau saine, 255 = lésion)
```
