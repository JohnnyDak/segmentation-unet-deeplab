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

- [x] **1. Modèles** (`feat/models`)
      U-Net (implémentation propre, pas juste un import) + wrapper DeepLabV3
      via `segmentation_models_pytorch` (encoder resnet34, cf. config.yaml).
      → Forward pass sur batch factice validé : sortie `(N, 2, 256, 256)`
      pour les deux architectures. U-Net : 31M paramètres, DeepLabV3+resnet34 :
      26M paramètres.

- [x] **2. Losses** (`feat/models`)
      Dice loss, Cross-Entropy, combo (poids configurables via `config.yaml`).
      → Validé sur cas synthétiques : masque parfait → loss ~0 (les 3
      losses) ; masque totalement inversé → loss nettement plus élevée
      (dice 0.94, cross_entropy 20.0, combo 10.47).

- [x] **3. Métriques** (`feat/models`)
      IoU / Dice, calculés **par classe** (pas seulement la moyenne).
      → Validé sur cas synthétiques : masque parfait → IoU ~1.0 ; masque
      totalement inversé → IoU ~0.0.

- [x] **4. Boucle d'entraînement** (`feat/models`)
      `src/train.py` : lit `configs/config.yaml`, entraîne, checkpoint à
      chaque epoch, reprise automatique si interruption (protège contre une
      session Colab coupée), logs métriques par epoch dans
      `experiments/<run_name>/`, met à jour `experiments/results_summary.csv`.
      → Validé en conditions réelles : run `unet_dice` en mode
      `--smoke-test` (8 images train / 4 val, CPU local) exécuté de bout en
      bout sans erreur, loss décroît (0.4638 → 0.3596), IoU progresse
      (0.376 → 0.524). Reprise depuis checkpoint vérifiée (relance détecte
      l'epoch suivante, ne retraite rien).
      Étapes 1 à 4 développées et testées ensemble sur une seule branche
      (`feat/models`) plutôt que 4 branches séparées — elles se valident
      mutuellement (le smoke-test du train-loop exerce modèles/losses/
      métriques en conditions réelles), les séparer aurait été artificiel.

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
