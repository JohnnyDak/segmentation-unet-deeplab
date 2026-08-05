# Initialiser le repo et le pousser sur GitHub

1. Crée un repo vide sur GitHub (sans README, sans .gitignore — on a déjà
   les nôtres), par exemple `segmentation-unet-deeplab`.

2. Depuis ce dossier en local :

```bash
git init
git add .
git commit -m "Initial commit: structure du projet, config, notebook de setup"
git branch -M main
git remote add origin https://github.com/<ton-username>/segmentation-unet-deeplab.git
git push -u origin main
```

3. Ajoute ton professeur comme collaborateur : sur GitHub,
   *Settings → Collaborators → Add people*, puis renseigne son
   pseudo ou son email GitHub.

## Bonnes pratiques de commit pour ce projet

- Un commit par étape logique (pas un seul gros commit à la fin).
- Exemples de messages clairs :
  - `feat: implémentation U-Net`
  - `feat: wrapper DeepLabV3 via segmentation_models_pytorch`
  - `feat: losses Dice / CE / combo`
  - `exp: run unet_dice — résultats dans experiments/results_summary.csv`
  - `docs: rapport d'ablation`
- Ne jamais committer : `data/raw/`, `data/processed/`, les poids de modèle
  (`*.pth`), `kaggle.json` — déjà exclus via `.gitignore`.
