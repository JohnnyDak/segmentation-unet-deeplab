# Roadmap — Segmentation U-Net vs DeepLabV3

Découpage du travail restant en étapes vérifiables. Chaque étape correspond
à une branche (`feat/...`), mergée dans `main` uniquement une fois testée et
fonctionnelle. Voir [GIT_SETUP.md](GIT_SETUP.md) pour la convention de commits.

Pour chaque difficulté rencontrée pendant une étape, ajouter une entrée dans
[CHALLENGES.md](CHALLENGES.md) **au moment où elle survient**, pas a posteriori.

## Étapes

- [x] **0. Dataset** (`feat/dataset-setup`)
      Téléchargement ISIC 2016, splits train/val/test figés (seed=42,
      reproductibilité vérifiée), `ISICDataset` validé sur les 3 splits.
      → Definition of done : `python -m src.data.verify` passe sans erreur.

- [ ] **1. Modèles** (`feat/models`)
      U-Net (implémentation propre, pas juste un import) + wrapper DeepLabV3
      via `segmentation_models_pytorch` (encoder resnet34, cf. config.yaml).
      → Definition of done : forward pass sur un batch factice produit la
      bonne shape de sortie `(N, num_classes, H, W)` ; nombre de paramètres
      de chaque modèle loggé (utile pour le rapport).

- [ ] **2. Losses** (`feat/losses`)
      Dice loss, Cross-Entropy, combo (poids configurables via `config.yaml`).
      → Definition of done : test sur cas synthétiques à valeur attendue
      connue (masque prédit = masque réel → loss proche de 0 ; masque
      totalement faux → loss proche du max).

- [ ] **3. Métriques** (`feat/metrics`)
      IoU / Dice, calculés **par classe** (pas seulement la moyenne).
      → Definition of done : même principe que les losses, cas synthétiques
      à valeur connue.

- [ ] **4. Boucle d'entraînement** (`feat/train-loop`)
      `src/train.py` : lit `configs/config.yaml`, entraîne, checkpoint à
      chaque epoch sur Drive (Colab), reprise automatique si interruption,
      logs métriques par epoch dans `experiments/<run_name>/`.
      → Definition of done : un run complet (`unet_dice`) tourne sans
      crash de bout en bout et écrit une ligne dans
      `experiments/results_summary.csv`.

- [ ] **5. Les 6 runs d'ablation** (pas de code, exécution)
      `unet_dice`, `unet_ce`, `unet_combo`, `deeplab_dice`, `deeplab_ce`,
      `deeplab_combo` — mêmes splits, mêmes seeds, config identique sauf
      architecture/loss.
      → Definition of done : `results_summary.csv` complet (6 lignes).

- [ ] **6. Évaluation** (`feat/evaluate`)
      `src/evaluate.py` : IoU/Dice par classe sur le test set, sélection et
      visualisation des cas d'échec (bords, petites régions).
      → Definition of done : figures + tableau exportés dans `reports/`.

- [ ] **7. Application de démo** (`feat/app-demo`)
      Upload d'image → masques des deux architectures côte à côte.
      → Definition of done : démo fonctionnelle en local (effort allégé,
      pas un produit fini).

- [ ] **8. Rapport** (`docs/rapport`)
      Rédaction à partir de `results_summary.csv`, des figures de l'étape 6
      et de `CHALLENGES.md`.
