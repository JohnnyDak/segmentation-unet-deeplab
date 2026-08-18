# Journal des difficultés

Log vivant des problèmes rencontrés, mis à jour au fil de chaque étape (pas
rétrospectivement). Sert de matière première pour la section
"Discussion et limites" du rapport final, et documente les décisions prises
face à un imprévu.

Une entrée par difficulté, format :

```
## [AAAA-MM-JJ] Titre court du problème

**Étape concernée** : (ex. 0. Dataset)
**Contexte** : ce qu'on essayait de faire
**Problème** : ce qui n'a pas marché / ce qui a été surprenant
**Diagnostic** : la cause identifiée
**Solution / décision** : ce qui a été fait, ou le compromis choisi
**Impact** : conséquence sur le protocole, les résultats, ou le calendrier
**Auteur** : qui a traité ça
```

---

## [2026-08-06] Fins de ligne CRLF lors de la régénération des splits sous Windows

**Étape concernée** : 0. Dataset

**Contexte** : vérification que `python -m src.data.make_splits` reproduit
exactement les splits déjà versionnés (900 images, seed=42), pour confirmer
que l'entraînement local/Windows donnera les mêmes runs que ceux faits sur
Colab.

**Problème** : une comparaison naïve (`diff`/`comm`) entre les fichiers
`data/processed/splits/*.txt` régénérés en local et ceux déjà commités sur
GitHub montrait des centaines de lignes "différentes", laissant croire à un
problème de reproductibilité.

**Diagnostic** : Python en mode texte sur Windows écrit des fins de ligne
`\r\n` (CRLF), alors que les fichiers originaux, générés sur Colab (Linux),
utilisent `\n` (LF). Chaque ligne différait donc d'un caractère invisible
`\r`, pas de contenu réel.

**Solution / décision** : après normalisation des fins de ligne, les deux
jeux de splits sont strictement identiques → la reproductibilité du split
est confirmée, pas de bug réel. Par ailleurs `src/data/dataset.py` utilise
déjà `line.strip()` en lisant les fichiers de split, ce qui absorbe `\r`
automatiquement — donc même sans y prêter attention, le chargement du
dataset n'aurait pas été affecté. Pas de changement de code nécessaire ;
les fichiers de splits ont été restaurés à leur version commitée (LF) pour
éviter un diff Git parasite.

**Impact** : aucun sur les résultats. A surveiller si quelqu'un régénère un
jour les splits en local sous Windows et les recommite directement — préférer
committer depuis Colab, ou configurer `.gitattributes` (`*.txt text eol=lf`)
si ça se reproduit.

**Auteur** : Data setup, session du 2026-08-06.

---

## [2026-08-18] Déséquilibre de classes rapporté sur un échantillon de 200 images, pas 900

**Étape concernée** : 0. Dataset

**Contexte** : relecture du document `Etat_avancement_projet_Segmentation`
pour vérifier sa cohérence avec l'état réel du repo avant diffusion. Le
document annonce 28,29 % de pixels "lésion" / 71,71 % de pixels "fond",
chiffre censé justifier l'étude d'ablation sur la loss.

**Problème** : recalcul du ratio sur les 900 masques réellement téléchargés
→ 17,92 % lésion / 82,08 % fond. Écart net (10 points) avec le chiffre du
document, pas une simple imprécision d'arrondi.

**Diagnostic** : la cellule du notebook `00_setup_and_first_look.ipynb`
("5. Vérifier le déséquilibre de classes") itère sur `mask_paths[:200]`
plutôt que sur la totalité de `mask_paths` (900 images), avec le
commentaire "échantillon pour aller vite". Recalculer précisément sur ces
200 mêmes masques reproduit exactement 28,29 % / 71,71 % — confirme que le
document a été rédigé à partir d'un sous-échantillon présenté comme une
statistique globale du dataset, sans que ce soit précisé.

**Solution / décision** : cellule corrigée pour itérer sur `mask_paths` en
entier (voir commit `fix:` sur `feat/dataset-setup`) — le calcul complet
sur 900 masques prend quelques secondes, la justification "pour aller vite"
ne tenait pas. Le chiffre correct à utiliser partout (rapport inclus) est
**17,92 % lésion / 82,08 % fond**.

**Impact** : le déséquilibre réel est plus marqué que ce qui était annoncé
— ça renforce, si besoin, la pertinence de l'ablation Dice vs CE vs combo
sur la classe minoritaire. Aucun impact sur le code d'entraînement (les
losses/métriques n'avaient pas encore été implémentées). Le document
`Etat_avancement_projet_Segmentation` doit être corrigé avant tout envoi
au professeur ou intégration dans le rapport final.

**Auteur** : Vérification de cohérence, session du 2026-08-18.
