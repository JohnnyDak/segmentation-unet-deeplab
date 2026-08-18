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
