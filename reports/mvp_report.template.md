# MVP Report — Raisonnement latent sur l'addition

> Rapport généré automatiquement par `scripts/run_sweep.py`. À compléter
> manuellement pour les sections d'interprétation (rester factuel, ne pas
> surinterpréter).

## Objectif

Tester H1 : ajouter des cycles de calcul latent (`k`) avant la sortie améliore-t-il
le raisonnement et la généralisation OOD sur l'addition multi-chiffres, sans
augmenter le nombre de paramètres (cellule récurrente à poids partagés) ?

## Setup expérimental

- Tâche : addition multi-chiffres, format autorégressif char-level, résultat inversé (LSB d'abord).
- Train : longueurs courtes ; OOD : longueurs strictement plus grandes.
- Positional encoding identique entre tous les modèles (contrôle du confound de longueur).
- Latent scratchpad : backbone + 1 cellule Transformer partagée appliquée `k` fois.

## Modèles comparés

- baseline = latent(k=0).
- latent(k) pour k croissant.
- compute_matched = baseline avec `n_layers + k` couches distinctes (+FLOPs ET +params).

## Datasets

Voir `configs/`. Splits train/val/test (longueurs vues) + ood (longueurs non vues).

## Résultats — Latent depth sweep

<!-- SWEEP_TABLE -->

Figures : `outputs/figures/accuracy_vs_k.png`, `ood_accuracy_vs_k.png`,
`accuracy_per_flop_vs_k.png`, `latency_vs_k.png`.

## Analyse OOD

(À remplir : l'accuracy OOD augmente-t-elle avec k ? plateau ? rendements décroissants ?)

## Analyse des représentations latentes

(À remplir depuis `outputs/metrics/<id>_analysis.json` et `outputs/figures/latent_pca.png` :
probe linéaire de la retenue, score de silhouette.)

## Limites

(À remplir : taille des modèles, nombre de seeds, portée des conclusions.)

## Conclusion

(À remplir factuellement. Validation faible de H1 : latent(k>0) > baseline.
Validation forte : latent(k>0) > compute_matched à compute équivalent.)

## Prochaines étapes

- Robustesse multi-seed (0–4).
- Phase 2 : communication latente multi-agent.
