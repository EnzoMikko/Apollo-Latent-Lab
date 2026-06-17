# Neural Latent Reasoning & Emergent Communication Lab

## 1. Contexte du projet

Ce projet est un laboratoire expérimental visant à étudier le raisonnement latent et la communication émergente entre agents neuronaux.

L’objectif n’est pas de construire un meilleur LLM généraliste, ni de battre l’état de l’art.  
L’objectif est de tester, dans un cadre contrôlé et reproductible, si :

1. Des cycles de calcul internes dans l’espace latent peuvent améliorer le raisonnement.
2. Des agents neuronaux peuvent apprendre à communiquer via des messages latents compressés.
3. Des représentations interprétables peuvent émerger sous contrainte de bande passante.
4. Ces effets sont mesurables, reproductibles et falsifiables.

Le projet doit rester volontairement simple, expérimental et mesurable.

Il faut privilégier :

- petits modèles ;
- datasets synthétiques ;
- expériences rapides ;
- métriques claires ;
- résultats reproductibles ;
- visualisation des représentations latentes.

---

## 2. Principe scientifique

Le projet cherche à répondre à la question suivante :

> Quelles sont les unités minimales de raisonnement et de communication dans un réseau de neurones ?

Plus concrètement :

- Le langage naturel est-il nécessaire pour raisonner ?
- Un modèle peut-il “réfléchir” dans son espace latent avant de produire une sortie ?
- Une communication vectorielle compressée entre agents peut-elle être plus efficace que du texte ?
- Les contraintes informationnelles peuvent-elles forcer l’apparition de structures latentes interprétables ?

---

## 3. Hypothèses à tester

### H1 — Latent Deliberation

Ajouter des cycles de calcul latent avant la sortie améliore la performance sur des tâches de raisonnement, sans augmenter le nombre de tokens générés.

### H2 — Compute-matched Latent Reasoning

Si le modèle latent est meilleur qu’un modèle classique à budget compute équivalent, alors le gain ne vient pas seulement du fait qu’il “calcule plus”, mais de la structure du calcul latent.

### H3 — Latent Communication

Sous contrainte de bande passante, deux agents peuvent apprendre à transmettre l’information utile via un message latent plus compact qu’une communication textuelle.

### H4 — Symbolic Emergence

Sous contrainte, les représentations latentes peuvent former des clusters ou axes interprétables correspondant à des variables de tâche : retenue, profondeur de parenthèse, distance au but, ordre relatif, etc.

---

## 4. Scope de la première version

La première version du repo ne doit pas tout faire.

Le MVP doit se concentrer sur :

1. Un modèle baseline Transformer simple.
2. Un modèle Latent Scratchpad avec `k` cycles de réflexion latente.
3. Un ou deux datasets synthétiques simples.
4. Une boucle d’entraînement propre.
5. Une évaluation reproductible.
6. Un système de logs d’expériences.
7. Une analyse basique des latents.

La communication multi-agent vient ensuite, dans une deuxième phase.

---

## 5. Expérience prioritaire à implémenter

### Expérience 1 — Addition multi-chiffres avec généralisation

Objectif :

> Tester si un modèle avec cycles latents généralise mieux qu’un Transformer classique sur des additions plus longues que celles vues à l’entraînement.

Exemple :

```text
Train:
12 + 45 = 57
123 + 456 = 579

Test OOD:
12345 + 67890 = 80235
````

Le modèle doit être entraîné sur des longueurs courtes et testé sur des longueurs plus grandes.

Pourquoi cette tâche :

* facile à générer ;
* vérité exacte connue ;
* difficulté contrôlable ;
* mesure d’accuracy simple ;
* permet d’analyser des variables latentes comme la retenue.

***

## 6. Architecture générale du repo

La structure attendue du repo est la suivante :

```text
neural-latent-lab/
│
├── README.md
├── SPEC.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── baseline_addition.yaml
│   ├── latent_addition_k1.yaml
│   ├── latent_addition_k2.yaml
│   ├── latent_addition_k4.yaml
│   ├── latent_addition_k8.yaml
│   ├── latent_addition_k16.yaml
│   ├── compute_matched_baseline.yaml
│   └── multi_agent_comm.yaml
│
├── src/
│   ├── neural_latent_lab/
│   │   ├── __init__.py
│   │   │
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── addition.py
│   │   │   ├── boolean_logic.py
│   │   │   ├── parentheses.py
│   │   │   └── pathfinding.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── baseline_transformer.py
│   │   │   ├── latent_scratchpad.py
│   │   │   ├── latent_cell.py
│   │   │   ├── compute_matched_transformer.py
│   │   │   └── multi_agent.py
│   │   │
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── train.py
│   │   │   ├── evaluate.py
│   │   │   ├── losses.py
│   │   │   ├── optimizer.py
│   │   │   └── checkpointing.py
│   │   │
│   │   ├── experiments/
│   │   │   ├── __init__.py
│   │   │   ├── run_experiment.py
│   │   │   ├── sweep_latent_depth.py
│   │   │   ├── sweep_latent_dim.py
│   │   │   ├── compute_matching.py
│   │   │   └── robustness.py
│   │   │
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── latent_probe.py
│   │   │   ├── latent_visualization.py
│   │   │   ├── clustering.py
│   │   │   ├── representation_similarity.py
│   │   │   └── metrics.py
│   │   │
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   ├── experiment_logger.py
│   │   │   └── results_schema.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── seed.py
│   │       ├── config.py
│   │       ├── device.py
│   │       └── flops.py
│
├── scripts/
│   ├── train_baseline.sh
│   ├── train_latent_sweep.sh
│   ├── evaluate_all.sh
│   ├── analyze_latents.sh
│   └── run_full_mvp.sh
│
├── experiments/
│   ├── addition/
│   │   ├── README.md
│   │   ├── baseline/
│   │   ├── latent_k1/
│   │   ├── latent_k2/
│   │   ├── latent_k4/
│   │   ├── latent_k8/
│   │   └── latent_k16/
│   │
│   ├── parentheses/
│   │   └── README.md
│   │
│   └── multi_agent/
│       └── README.md
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── figures/
│   └── latent_dumps/
│
├── reports/
│   ├── mvp_report.md
│   ├── experiment_results.md
│   ├── latent_analysis.md
│   └── figures/
│
└── tests/
    ├── test_data_generation.py
    ├── test_model_shapes.py
    ├── test_training_step.py
    ├── test_metrics.py
    └── test_reproducibility.py
```

***

## 7. Rôle de chaque dossier

### `configs/`

Contient les configurations YAML des expériences.

Chaque config doit définir :

* dataset ;
* taille du modèle ;
* profondeur du modèle ;
* nombre de cycles latents `k` ;
* batch size ;
* learning rate ;
* nombre d’epochs ;
* seed ;
* device ;
* chemin de sortie ;
* paramètres de logging.

Les configs doivent permettre de relancer exactement une expérience.

***

### `src/neural_latent_lab/data/`

Contient les générateurs de datasets synthétiques.

Le projet doit éviter au début les datasets réels complexes.

Datasets à prévoir :

1. `addition.py`
   * génération d’additions multi-chiffres ;
   * split train/test ;
   * test OOD sur longueurs non vues ;
   * possibilité de récupérer des labels internes comme la retenue.

2. `boolean_logic.py`
   * expressions booléennes ;
   * AND, OR, NOT, XOR ;
   * profondeur contrôlée.

3. `parentheses.py`
   * parenthèses équilibrées ;
   * profondeur maximale ;
   * classification valide/invalide.

4. `pathfinding.py`
   * petites grilles ;
   * shortest path ;
   * distance au but.

Pour la v0, `addition.py` suffit.

***

### `src/neural_latent_lab/models/`

Contient les architectures.

#### `baseline_transformer.py`

Modèle Transformer classique.

Flux attendu :

```text
input tokens
    ↓
embedding
    ↓
Transformer
    ↓
output head
    ↓
prediction
```

Ce modèle sert de baseline principale.

***

#### `latent_scratchpad.py`

Modèle principal du projet.

Flux attendu :

```text
input tokens
    ↓
embedding
    ↓
encoder
    ↓
latent state h0
    ↓
latent refinement block répété k fois
    ↓
decoder / output head
    ↓
prediction
```

Le paramètre central est `k`.

Il doit être possible de lancer :

```text
k = 0
k = 1
k = 2
k = 4
k = 8
k = 16
```

`k = 0` doit être équivalent à une baseline sans réflexion latente.

***

#### `latent_cell.py`

Bloc de calcul latent réutilisable.

Il représente une étape de réflexion interne.

Il doit prendre un état latent en entrée et retourner un état latent mis à jour.

Conceptuellement :

```text
h_next = latent_cell(h)
```

La cellule peut être un petit bloc attentionnel ou MLP, mais elle doit rester simple.

***

#### `compute_matched_transformer.py`

Baseline importante.

Elle sert à vérifier que le gain du modèle latent ne vient pas uniquement d’un budget compute plus élevé.

Ce modèle doit être configuré pour avoir un coût proche du modèle latent.

Comparaison attendue :

```text
latent_scratchpad(k=4) vs transformer_compute_matched
latent_scratchpad(k=8) vs transformer_compute_matched
```

***

#### `multi_agent.py`

À implémenter en phase 2.

Architecture prévue :

```text
input
  ↓
Agent A encoder
  ↓
latent message z
  ↓ bottleneck
Agent B decoder
  ↓
output
```

Le message `z` doit avoir une dimension contrôlée :

```text
latent_dim = 2, 4, 8, 16, 32, 64
```

***

### `src/neural_latent_lab/training/`

Contient la logique d’entraînement.

#### `train.py`

Doit gérer :

* chargement config ;
* initialisation seed ;
* chargement dataset ;
* création modèle ;
* boucle d’entraînement ;
* validation ;
* sauvegarde checkpoint ;
* logging métriques.

#### `evaluate.py`

Doit permettre d’évaluer un modèle entraîné sur :

* split train ;
* split validation ;
* test standard ;
* test OOD ;
* test avec longueurs supérieures.

#### `losses.py`

Contient les fonctions de loss.

Pour la v0 :

* cross entropy token-level ;
* exact match accuracy ;
* éventuellement loss auxiliaire si probes ajoutés plus tard.

#### `checkpointing.py`

Doit sauvegarder :

* poids du modèle ;
* config complète ;
* métriques finales ;
* seed ;
* hash ou identifiant d’expérience ;
* epoch ;
* optimizer state si nécessaire.

***

### `src/neural_latent_lab/experiments/`

Contient les scripts Python qui orchestrent les expériences.

#### `run_experiment.py`

Point d’entrée général.

Exemple conceptuel :

```text
run_experiment --config configs/latent_addition_k4.yaml
```

#### `sweep_latent_depth.py`

Lance plusieurs expériences sur `k`.

Valeurs attendues :

```text
k = 0, 1, 2, 4, 8, 16
```

But :

* mesurer l’effet du nombre de cycles latents ;
* voir s’il existe un seuil utile ;
* détecter les rendements décroissants.

#### `sweep_latent_dim.py`

Utilisé pour la communication multi-agent.

Valeurs attendues :

```text
latent_dim = 2, 4, 8, 16, 32, 64
```

#### `compute_matching.py`

Compare les modèles latents à des modèles classiques de compute équivalent.

Objectif :

```text
Vérifier que le latent scratchpad n’est pas meilleur uniquement parce qu’il utilise plus de calcul.
```

#### `robustness.py`

Relance les expériences avec plusieurs seeds.

Seeds recommandées :

```text
0, 1, 2, 3, 4
```

Le résultat n’est intéressant que s’il est stable sur plusieurs seeds.

***

### `src/neural_latent_lab/analysis/`

Contient l’analyse des représentations latentes.

#### `latent_probe.py`

Objectif :

```text
Tester si les états latents contiennent des variables utiles de la tâche.
```

Pour l’addition :

* présence d’une retenue ;
* position du chiffre ;
* longueur des nombres ;
* difficulté de l’exemple ;
* erreur ou succès final.

Méthode :

* entraîner un petit classifieur linéaire sur les états latents ;
* mesurer si une variable interne est décodable.

Si une variable est décodable, cela suggère que le latent encode une structure utile.

***

#### `latent_visualization.py`

Doit produire :

* PCA ;
* UMAP si disponible ;
* t-SNE optionnel.

Objectif :

```text
Observer si les états latents forment des groupes cohérents.
```

Exemples de couleurs :

* succès vs échec ;
* présence de retenue ;
* longueur de l’addition ;
* valeur du résultat ;
* profondeur de raisonnement `k`.

***

#### `clustering.py`

Mesure si les clusters sont stables.

Métriques possibles :

* silhouette score ;
* adjusted rand index ;
* stabilité entre seeds ;
* stabilité entre checkpoints.

***

#### `representation_similarity.py`

Compare les représentations entre :

* différents seeds ;
* différents `k` ;
* baseline vs latent ;
* train vs OOD.

But :

```text
Vérifier si les structures latentes sont reproductibles ou juste du bruit.
```

***

#### `metrics.py`

Centralise les métriques du projet.

Métriques principales :

```text
accuracy
exact_match_accuracy
ood_accuracy
loss
latency_ms
tokens_generated
estimated_flops
accuracy_per_flop
accuracy_per_token
```

Métriques pour le latent :

```text
latent_depth_k
latent_state_norm
latent_state_variance
latent_cluster_score
linear_probe_accuracy
representation_stability
```

Métriques pour la communication :

```text
latent_dim
message_entropy
success_rate
bits_per_success
performance_under_bottleneck
```

***

## 8. Expériences à mener

### Expérience A — Baseline addition

But :

```text
Établir la performance d’un Transformer classique.
```

Config :

```text
model = baseline_transformer
dataset = addition
train_digits = 2 à 4
test_digits = 2 à 4
ood_digits = 5 à 8
```

Mesures :

```text
train_accuracy
test_accuracy
ood_accuracy
latency_ms
estimated_flops
tokens_generated
```

***

### Expérience B — Latent depth sweep

But :

```text
Mesurer l’effet de la profondeur de réflexion latente.
```

Configs :

```text
k = 0
k = 1
k = 2
k = 4
k = 8
k = 16
```

Comparer :

```text
accuracy vs k
ood_accuracy vs k
latency vs k
flops vs k
accuracy_per_flop vs k
```

Résultat attendu si H1 est vraie :

```text
Les modèles avec k > 0 améliorent l’accuracy ou la généralisation OOD par rapport à k = 0.
```

Mais attention :

```text
Si le gain disparaît à compute égal, H1 reste faible.
```

***

### Expérience C — Compute-matched baseline

But :

```text
Contrôler l’effet du compute supplémentaire.
```

Comparer :

```text
latent_scratchpad_k4
latent_scratchpad_k8
baseline_transformer_compute_matched
```

Résultat fort :

```text
Le modèle latent garde un avantage même face à une baseline compute-matched.
```

Résultat faible :

```text
Le modèle latent est meilleur que la baseline simple, mais pas meilleur que la baseline compute-matched.
```

Résultat négatif :

```text
Le modèle latent est moins bon ou plus lent sans gain clair.
```

***

### Expérience D — Robustesse multi-seed

But :

```text
Vérifier que les résultats ne dépendent pas d’une seed chanceuse.
```

Seeds :

```text
0, 1, 2, 3, 4
```

Mesurer :

```text
mean_accuracy
std_accuracy
mean_ood_accuracy
std_ood_accuracy
```

Un résultat intéressant doit être stable.

***

### Expérience E — Analyse des latents

But :

```text
Comprendre ce que le modèle encode dans ses états latents.
```

À faire :

1. Extraire les états latents `h0, h1, ..., hk`.
2. Sauvegarder les latents dans `outputs/latent_dumps/`.
3. Associer chaque latent à des métadonnées de tâche :
   * longueur de l’addition ;
   * présence de retenue ;
   * succès/échec ;
   * position ;
   * difficulté.
4. Lancer PCA / UMAP.
5. Lancer des probes linéaires.
6. Mesurer la stabilité des clusters.

Résultat intéressant :

```text
Les états latents se structurent selon des variables utiles de la tâche.
```

Exemple :

```text
Les exemples avec retenue forment un cluster ou deviennent décodables par probe linéaire.
```

***

### Expérience F — Communication latente multi-agent

À faire après le MVP.

But :

```text
Tester si deux agents apprennent un protocole compressé.
```

Architecture :

```text
Agent A reçoit une partie de l’information.
Agent A produit un message latent z.
z est compressé via un bottleneck.
Agent B reçoit z et doit produire la bonne réponse.
```

Variables :

```text
latent_dim = 2, 4, 8, 16, 32, 64
```

Baselines :

```text
communication textuelle
communication one-hot
communication aléatoire
communication latente non contrainte
```

Mesures :

```text
success_rate
message_entropy
bits_per_success
performance_under_bottleneck
```

Résultat intéressant :

```text
Un faible latent_dim conserve une performance élevée.
```

Résultat très intéressant :

```text
Le message latent est plus compact qu’une communication textuelle à performance égale.
```

***

## 9. Critères de réussite

Le MVP est réussi si :

1. Le repo permet de lancer une expérience de bout en bout.
2. La baseline Transformer fonctionne.
3. Le Latent Scratchpad fonctionne pour plusieurs valeurs de `k`.
4. Les résultats sont sauvegardés proprement.
5. Les métriques sont comparables entre runs.
6. Les latents peuvent être extraits et analysés.
7. Le sweep `k` produit un rapport lisible.
8. Les expériences sont reproductibles avec une seed fixe.

***

## 10. Critères scientifiques de validation

### Validation faible de H1

```text
latent_scratchpad(k > 0) > baseline(k = 0)
```

sur accuracy ou OOD accuracy.

### Validation forte de H1

```text
latent_scratchpad(k > 0) > compute_matched_baseline
```

à budget compute équivalent.

### Validation de H3

```text
latent_communication > text_communication
```

à performance égale ou à coût informationnel inférieur.

### Validation de H4

```text
Les représentations latentes présentent des structures stables, interprétables et reproductibles.
```

Exemples :

* clusters stables ;
* variables de tâche décodables ;
* similarité entre seeds ;
* axes PCA/UMAP cohérents.

***

## 11. Critères de falsification

Le projet doit accepter l’échec.

Les hypothèses sont affaiblies ou rejetées si :

1. Les cycles latents n’améliorent pas la performance.
2. Les gains disparaissent dès qu’on compare à compute équivalent.
3. Les latents sont instables entre seeds.
4. Les clusters observés ne correspondent à aucune variable de tâche.
5. Les probes linéaires ne décodent aucune information utile.
6. La communication latente ne compresse pas mieux que le texte.
7. Les résultats ne sont pas reproductibles.

Un résultat négatif reste un résultat utile.

***

## 12. Métriques obligatoires

Chaque run doit produire un fichier de métriques contenant au minimum :

```text
experiment_id
config_name
model_type
dataset
seed
latent_depth_k
num_parameters
train_accuracy
validation_accuracy
test_accuracy
ood_accuracy
train_loss
validation_loss
latency_ms
estimated_flops
tokens_generated
accuracy_per_flop
accuracy_per_token
checkpoint_path
timestamp
```

Pour les expériences latentes :

```text
latent_state_norm_mean
latent_state_norm_std
latent_state_variance
linear_probe_accuracy
cluster_score
representation_stability_score
```

Pour les expériences multi-agent :

```text
latent_dim
message_entropy
success_rate
bits_per_success
```

***

## 13. Format attendu des outputs

Chaque expérience doit écrire dans :

```text
outputs/
```

Structure recommandée :

```text
outputs/
├── logs/
│   └── experiment_id.log
│
├── metrics/
│   └── experiment_id.json
│
├── checkpoints/
│   └── experiment_id.pt
│
├── figures/
│   ├── accuracy_vs_k.png
│   ├── ood_accuracy_vs_k.png
│   ├── accuracy_per_flop_vs_k.png
│   └── latent_pca.png
│
└── latent_dumps/
    └── experiment_id_latents.npz
```

Chaque run doit être traçable depuis sa config.

***

## 14. Rapport attendu

Le repo doit permettre de générer un rapport dans :

```text
reports/mvp_report.md
```

Le rapport doit contenir :

```text
# MVP Report

## Objectif

## Setup expérimental

## Modèles comparés

## Datasets

## Métriques

## Résultats baseline

## Résultats latent depth sweep

## Résultats compute-matched

## Analyse OOD

## Analyse des représentations latentes

## Limites

## Conclusion

## Prochaines étapes
```

Le rapport doit être factuel.

Ne pas surinterpréter les résultats.

***

## 15. Ce que Claude Code doit faire en priorité

Ordre de travail recommandé :

1. Créer la structure du repo.
2. Créer les fichiers vides avec docstrings ou commentaires explicatifs.
3. Définir les configs YAML.
4. Implémenter le dataset addition.
5. Implémenter la baseline Transformer.
6. Implémenter le Latent Scratchpad.
7. Implémenter la boucle d’entraînement.
8. Implémenter l’évaluation.
9. Implémenter le logging JSON.
10. Implémenter le sweep sur `k`.
11. Implémenter l’extraction des latents.
12. Implémenter les premières visualisations.
13. Générer un rapport MVP.

***

## 16. Priorité absolue

Le projet doit rester :

```text
simple
mesurable
reproductible
falsifiable
```

Ne pas complexifier trop tôt.

Ne pas ajouter de gros LLM.

Ne pas utiliser de dataset massif.

Ne pas ajouter d’architecture sophistiquée avant d’avoir un résultat clair sur le MVP.

***

## 17. Première milestone

La première milestone est terminée quand on peut lancer :

```text
baseline_addition
latent_addition_k1
latent_addition_k2
latent_addition_k4
latent_addition_k8
latent_addition_k16
```

Et obtenir :

```text
accuracy_vs_k
ood_accuracy_vs_k
latency_vs_k
accuracy_per_flop_vs_k
latent_pca
mvp_report.md
```

***

## 18. Résumé du projet en une phrase

Ce repo teste si un réseau de neurones peut mieux raisonner et mieux communiquer en utilisant des états latents compressés plutôt que du texte explicite, dans un cadre expérimental contrôlé, reproductible et falsifiable.

````

Tu peux donner ça directement à Claude Code comme **contexte initial**.

Ma reco : demande-lui ensuite de faire uniquement la **milestone 1**, pas tout le projet d’un coup. Par exemple :

```text
À partir de cette spec, crée uniquement la structure du repo, les fichiers nécessaires, les configs YAML et les stubs documentés. Ne code pas encore les modèles complets. L’objectif est d’obtenir une base propre avant l’implémentation.
````
