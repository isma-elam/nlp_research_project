# Traitement du langage naturel (japonais) — Classification de difficulté JLPT

Projet de **traitement du langage naturel appliqué au japonais** : estimer la difficulté d’un texte et produire une prédiction en 3 classes (`easy` / `medium` / `hard`) à partir de ressources JLPT et de features linguistiques **explicables**.

Le projet est structuré autour de 3 blocs :
1) **Ressources** (vocabulaire / grammaire / kanji, classés par niveau JLPT) → fichiers JSON optimisés
2) **Extraction de features** (ratios kana/kanji, stats de ponctuation, signaux JLPT) → vecteurs numériques
3) **Modèle baseline** (Régression Logistique scikit-learn) + **évaluation** (test interne + validation sur un jeu séparé)

---

## Objectif scientifique (contexte)

Le cœur du projet est une chaîne expérimentale reproductible :

**texte japonais → features linguistiques → classification (easy/medium/hard) → métriques → analyse**

L’approche mise en place privilégie des features interprétables (ex : densité de kanji, ratios de niveaux JLPT) pour :
- relier la prédiction à des signaux compréhensibles,
- déboguer les erreurs,
- documenter les limites et les futures améliorations.

---

La chaîne end-to-end est opérationnelle et **reproductible** via une commande unique.

- Données : corpus manuel dans `data/input/manual_phrases.csv` → standardisé en `data/input/corpus.csv`
- Modèle : régression logistique scikit-learn entraînée sur un mélange de signaux explicables + représentation texte.
- Évaluation :
	- métriques train/test (split interne) dans `data/raw/baseline_metrics.json`
	- validation séparée via `data/input/validation_phrases.csv`, détails dans `data/raw/validation_details.csv`

Sorties principales :
- `data/raw/baseline_model.joblib`
- `data/raw/baseline_metrics.json`
- `data/raw/validation_details.csv`

---

## Dataset, labels et “word-level scoring” (intuition)

### 1) Intuition (niveau mot / kanji / grammaire)
- On associe des éléments (mots, kanji, patterns) à un niveau JLPT (N5 → N1).
- On peut ensuite dériver une notion de difficulté globale à partir des **proportions** observées.

### 2) Labels du projet
Le projet utilise 3 labels :
- `easy`
- `medium`
- `hard`

Ces labels sont présents dans les fichiers :
- entraînement : `data/input/manual_phrases.csv` (source manuelle)
- validation : `data/input/validation_phrases.csv` (jeu séparé)

---

## Données et ressources

### A) Dictionnaires / listes JLPT
Trois ressources sont utilisées :

1) Vocabulaire JLPT (CSV par niveau)
- source (référence utilisée pendant le projet) : https://www.kaggle.com/datasets/robinpourtaud/jlpt-words-by-level/data

2) Kanji (joyo + niveau JLPT)
- source (référence utilisée pendant le projet) : https://github.com/NHV33/joyo-kanji-compilation/tree/master

3) Grammaire JLPT
- construite manuellement (compilation depuis un support d’apprentissage (JLPT Sensei))

Ces fichiers sources se trouvent dans :
- `data/dictionaries/`

Les JSON générés se trouvent dans :
- `data/json/`

### B) Corpus d’entraînement
- source : `data/input/manual_phrases.csv`
- colonnes attendues : `text`, `label` et éventuellement `target_jlpt` pour la documentation

### C) Jeu de validation (séparé)
- source : `data/input/validation_phrases.csv`
- colonnes attendues : `id`, `text`, `label`

Les métriques de validation dépendent directement des labels dans `validation_phrases.csv`.

---

## Installation

### Prérequis

- Windows (PowerShell ou Git Bash) + Python 3.10+ recommandé
- dépendances Python : `pandas`, `scikit-learn`, `joblib`

Optionnel : **tokenisation + lemmatisation** avec SudachiPy (utile pour mieux matcher le vocabulaire JLPT en forme dictionnaire, au lieu d'heuristiques de déconjugaison uniquement) :
- `pip install sudachipy sudachidict_core`

SudachiPy est essentiel car les listes JLPT sont majoritairement en **forme dictionnaire** (lemme), alors que dans un texte réel on observe beaucoup de **formes conjuguées** (ex: `飲みます`, `食べた`, `見て`, etc.). Sans lemmatisation, un lookup “par sous-chaîne” sur les entrées JLPT rate une partie des occurrences → faux négatifs et features JLPT moins fiables.

Concrètement, quand SudachiPy est disponible, on récupère pour chaque token sa `dictionary_form()` (ou à défaut une forme normalisée), ce qui augmente la couverture du matching vocabulaire.

Comment ça marche sans SudachiPy :
- SudachiPy est important pour garder une exécution simple et reproductible.
- Si SudachiPy n’est pas installé, on retombe sur des heuristiques implémentées dans `scripts/feature_extract.py` :
	- inférence de lemmes à partir des formes polies en `〜ます` (`飲みます → 飲む`, `勉強します → 勉強する`, et hypothèse `stem+る` pour l’ichidan),
	- heuristique minimale `〜て/〜た` pour l’ichidan (`食べて → 食べる`, `見た → 見る`),
	- + matching direct par sous-chaîne (avec garde-fous anti faux-positifs / anti double-compte).

Autrement dit : SudachiPy améliore la qualité des features de vocabulaire, mais le pipeline reste fonctionnel sans lui grâce aux heuristiques de dé-conjugaison (et le modèle final peut aussi s’appuyer sur la représentation texte en n-grammes de caractères).

Important : utilise toujours **le même Python** (idéalement la venv `.venv`) pour entraîner et prédire, sinon tu peux voir des `InconsistentVersionWarning`.

### Créer la venv + installer les dépendances

PowerShell :
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install pandas scikit-learn joblib
```

Git Bash :
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
python -m pip install pandas scikit-learn joblib
```

---

## Exécution (le projet au complet)

### Option 1 — Commande unique (pipeline complet + vérification)

```bash
./.venv/Scripts/python scripts/pipeline.py --all --verify
```

Ce que ça fait :
- `--all` : dictionnaires → corpus → entraînement → validation
- `--verify` : exporte un rapport détaillé par phrase dans `data/raw/validation_details.csv`

Sorties attendues :
- modèle : `data/raw/baseline_model.joblib`
- métriques train/test (split interne) : `data/raw/baseline_metrics.json`
- rapport validation : `data/raw/validation_details.csv`

Optionnel (analyse plus poussée) :

- Écrire le rapport validation ailleurs :
```bash
./.venv/Scripts/python scripts/pipeline.py --all --verify --validate-details-out data/raw/validation_details.csv
```

- Lancer des stress-tests (détecter des erreurs “faciles à provoquer”) :
```bash
./.venv/Scripts/python scripts/pipeline.py --verify --stress --stress-terms "無料" --stress-out data/raw/stress_tests.csv
```

### Graphes

Prérequis (une seule fois) :
```bash
./.venv/Scripts/python -m pip install matplotlib
```

Générer les graphes à partir de `data/raw/validation_details.csv` :
```bash
./.venv/Scripts/python scripts/make_graphs.py
```

Sorties : PNG dans `data/raw/figures/`.

### Option 2 — Étape par étape

Dans toutes les commandes ci-dessous, on utilise explicitement le Python de la venv.

#### 1) (Optionnel) Reconstruire les dictionnaires
Les scripts écrivent dans `data/json/`.

```bash
./.venv/Scripts/python scripts/nettoyage_kanji.py
./.venv/Scripts/python scripts/nettoyage_vocab.py
./.venv/Scripts/python scripts/nettoyage_grammaire.py
```

#### 2) Construire le corpus d’entraînement
À partir de `data/input/manual_phrases.csv`.

```bash
./.venv/Scripts/python scripts/corpus.py
```

Sortie : `data/input/corpus.csv`

#### 3) Entraîner le modèle baseline

```bash
./.venv/Scripts/python scripts/train_model.py
```

Par défaut, `train_model.py` choisit automatiquement la stratégie de split la plus “saine” :
- s’il détecte suffisamment de doublons/quasi-doublons, il utilise un split **groupé** anti-fuite ;
- sinon, il revient à un `train_test_split` **stratifié** (répartition parfaitement équilibrée des classes sur train/test).

La stratégie réellement utilisée est enregistrée dans `data/raw/baseline_metrics.json` → `split.method`.

Sorties :
- `data/raw/baseline_model.joblib`
- `data/raw/baseline_metrics.json`

#### 4) Validation (jeu séparé)
Par défaut, le script lit `data/input/validation_phrases.csv`.

Résumé :
```bash
./.venv/Scripts/python scripts/validate.py
```

Rapport détaillé (par phrase, dans le terminal) :
```bash
./.venv/Scripts/python scripts/validate.py --details
```

Rapport détaillé exporté :
```bash
./.venv/Scripts/python scripts/validate.py --details-out data/raw/validation_details.csv
```

---

## Prédire sur une phrase

```bash
./.venv/Scripts/python scripts/predict.py --text "猫が好きです。"
```

Option JSON :
```bash
./.venv/Scripts/python scripts/predict.py --text "猫が好きです。" --json
```

Note anti-fuite : `predict.py` peut avertir si la phrase est déjà présente dans `data/input/corpus.csv`.
Tu peux désactiver ce check :
```bash
./.venv/Scripts/python scripts/predict.py --text "猫が好きです。" --no-leak-check
```

---

## Comment lire les métriques

Il y a 2 types de métriques, et elles ne mesurent pas la même chose.

### 1) `baseline_metrics.json` (test interne)
Fichier : `data/raw/baseline_metrics.json`

Écrit par `scripts/train_model.py`. Il contient les métriques obtenues sur le **test set** issu du `train_test_split` du corpus (`data/input/corpus.csv`).

Contenu typique :
- `split` : taille test, random_state, stratification, distribution des classes
- `accuracy`
- `f1_macro`
- `report` : `classification_report` détaillé

### 2) `validation_details.csv` (jeu séparé)
Fichier : `data/raw/validation_details.csv`

Écrit par `scripts/validate.py` (et par le pipeline avec `--verify`).
Une ligne par phrase de validation, avec :
- `label_true`, `label_pred`, `correct`
- `p_easy`, `p_medium`, `p_hard` (si le modèle expose `predict_proba`)
- `text`

---

## Comprendre la prédiction : label vs score

### 1) `Prediction: ...`
La classe `easy/medium/hard` affichée par `predict.py` correspond au **choix du modèle** : la probabilité la plus haute (argmax).

Mathématiquement :

$$
\hat{y} = \arg\max_k P(y=k\mid x)
$$

Exemple :

- $P(easy)=0.40$
- $P(medium)=0.35$
- $P(hard)=0.25$

`Prediction = easy` (car $0.40$ est la plus grande probabilité)

### 2) `Score (facile): .../100 -> ...`
Le score affiché est un **score continu** dérivé des probabilités, puis converti en bande (facile/moyen/difficile) via une heuristique.

En 3 classes (easy/medium/hard), le projet utilise :

$$
\mathrm{score}_{easy,\,0\text{–}100} = 100 \times (P(easy) + 0.5 \times P(medium))
$$


Intuition :
- `easy` → vaut **1** (facile)
- `medium` → vaut **0.5** (semi-facile)
- `hard` → vaut **0** (pas facile)

Exemple concret (mêmes probabilités que ci-dessus) :

$$
100\times(0.40 + 0.5\times 0.35)=100\times(0.40+0.175)=57.5
$$

`Score = 57.5 / 100`

Puis une bande :
- `score_easy < 33` → difficile
- `33 ≤ score_easy < 66` → moyen
- `score_easy ≥ 66` → facile

Ici : `57.5 → moyen`

Note : il est donc possible d’avoir une **classe** `easy` mais un **score** dans la bande “moyen” quand les probabilités sont serrées.

Point important : `Prediction` et `Score` ne répondent pas au même besoin

- `Prediction` = décision brute (argmax)
- `Score` = nuance / lecture continue (projection sur une échelle 0–100)

Exemple typique d’incertitude (probabilités proches) :

- $P(easy)=0.36$, $P(medium)=0.34$, $P(hard)=0.30$
- argmax = `easy`
- score $\approx 100\times(0.36+0.5\times 0.34)=53$ → bande `moyen`

Insight : le score est une **projection 1D** de la distribution de probabilités

On passe de $(P(easy),P(medium),P(hard))$ (3D) à un nombre unique (1D) : on perd de l’information, mais on gagne en lisibilité.

---

## Comprendre la Régression Logistique (formules)

Le classifieur final est une **régression logistique** (scikit-learn). Le modèle ne “voit” pas le japonais : il voit un vecteur de features $x$.

Intuition : $x$ encode des signaux mesurables (exemples)
- longueur / ponctuation
- ratio de kanji, hiragana, katakana
- ratios JLPT (vocab/grammaire/kanji matchés)

### Cas binaire

On calcule un score linéaire :

$$
z = w^\top x + b
$$

Puis on transforme en probabilité via la sigmoïde :

$$
P(y=1\mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### Cas multi-classes (easy / medium / hard)

En 3 classes, on calcule un score par classe $k$ :

$$
z_k = w_k^\top x + b_k
$$

Puis on obtient une distribution de probabilités par la **softmax** :

$$
P(y=k\mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

La prédiction affichée (`Prediction: ...`) est l’argmax :

$$
\hat{y} = \arg\max_k P(y=k\mid x)
$$

Rappel : en multi-classes, les probabilités somment à 1 :

$$
P(easy) + P(medium) + P(hard) = 1
$$

Le score 0–100 (défini plus haut) est ensuite une manière simple de résumer ces 3 valeurs sur une échelle unique.

---

## Modèle : signaux explicables + représentation texte

Le projet est conçu pour rester **explicable**, mais la classe `medium` est souvent difficile à capturer avec uniquement des ratios “JLPT”.
Le modèle combine donc deux familles d’entrées (fusionnées dans un pipeline scikit-learn) :

1) Features numériques explicables (issues de `scripts/feature_extract.py`) : ratios kana/kanji, ponctuation, ratios JLPT, etc.
2) Représentation du texte via TF‑IDF de **n-grammes de caractères** (utile en japonais, sans tokenisation obligatoire)

Le tout est classé par une régression logistique (`scripts/model_features.py`).

---

## Difficultés rencontrées : la classe `medium`

Un point dur du projet a été la stabilité de `medium` : on a observé des phases où `medium` “s’effondrait” et était massivement prédit comme `easy` ou `hard`.

Causes typiques (et réalistes) :
- `medium` est une classe intrinsèquement “entre-deux” : certaines phrases sont proches de `easy`, d’autres proches de `hard`.
- Les features purement “proportions JLPT” peuvent manquer de signal sur la **forme** (style, tournures, motifs d’écriture), donc la frontière `medium` devient floue.
- Si le dataset est petit ou bruité (doublons, phrases quasi-identiques, overlap train/validation), le modèle généralise mal et la matrice de confusion devient trompeuse.
- Les pondérations de classes (`--class-weight balanced`) peuvent aider, mais doivent rester optionnelles : par défaut on garde un comportement neutre (“réaliste”).

En pratique, `medium` est souvent la classe la plus difficile à distinguer, car elle se situe naturellement entre `easy` et `hard`.

---

## Analyse d’erreurs : commandes utiles (détails)

### 1) Rapport complet phrase-par-phrase

```bash
./.venv/Scripts/python scripts/validate.py --details
```

Export CSV (recommandé pour analyser ensuite avec pandas / graphes) :

```bash
./.venv/Scripts/python scripts/validate.py --details-out data/raw/validation_details.csv
```

### 4) Stress-tests (ex : répétition d’un terme)

```bash
./.venv/Scripts/python scripts/validate.py --stress --stress-terms "無料" --stress-out data/raw/stress_tests.csv
```

---

## Extraction de features (résumé technique)

Le module central est `scripts/feature_extract.py`.

Signaux principaux :
- ratios d’écriture : `ratio_kanji`, `ratio_hira`, `ratio_kata` (ponctuation et espaces exclus du dénominateur)
- ponctuation : `punct_count`, `comma_count`, `ratio_punct`
- structure : `sent_count`, `sent_avg_len`
- signaux JLPT : comptes et ratios `vocab_*`, `kanji_*`, `grammar_*`

Améliorations (robustesse) :
- normalisation Unicode **NFKC** avant extraction (stabilité plein largeur/mi-largeur)
- réduction des faux-positifs (ignore entrées trop courtes, stoplist minimale, etc.)
- anti double-compte : si plusieurs entrées matchent en sous-chaîne, on garde les matches les plus longs
- features de **ratios** par niveau (`*_ratio_N5..N1`) + `*_max_level_num`

---

## Features utilisées par le modèle (important)

Pour mieux coller à l’idée “proportions par niveau”, le modèle privilégie des ratios et évite de sur-pondérer des signaux de volume.

Concrètement : certaines features “volume” restent calculées dans `extract_features` (pour explicabilité), mais sont retirées de l’entrée du classifieur.

Exemples retirés de l’entrée ML :
- longueurs/volume : `len_chars`, `sent_avg_len`
- totaux bruts : `vocab_total`, `grammar_total`, `kanji_total`
- compteurs absolus par niveau : `vocab_N*`, `kanji_N*`, `grammar_N*`

L’implémentation est centralisée dans `scripts/model_features.py` (construction de X identique et dans le même ordre pour `train_model.py`, `validate.py`, `predict.py`).

---

## Unicode (japonais) : normalisation NFKC

Le projet normalise le texte en NFKC pendant l’extraction pour éviter les différences fullwidth/halfwidth (et autres variantes Unicode) qui peuvent casser les matches dictionnaire.

Exemples d’équivalences fréquentes :
- `ｶﾀｶﾅ` ↔ `カタカナ`
- `ＡＢＣ１２３` ↔ `ABC123`

---

## Structure du dépôt

- `scripts/` : pipeline, entraînement, validation, prédiction, nettoyage dictionnaires
- `data/dictionaries/` : sources CSV (JLPT vocab/grammar + joyo)
- `data/json/` : dictionnaires JSON générés
- `data/input/` : corpus d’entraînement + jeu de validation
- `data/raw/` : artefacts générés (modèle, métriques, rapports)

---

## Dépannage

### 1) Lancer depuis la racine
Si tu as des erreurs d’import, lance les scripts depuis la racine du projet (pas depuis `scripts/`).

### 2) Versions scikit-learn / warnings
Utilise `./.venv/Scripts/python` partout (train + predict + validate).

### 3) `????` dans `--text` (Windows)
Si ton terminal remplace les caractères japonais par `?`, la chaîne reçue par Python est cassée → les features deviennent incohérentes.
Solution la plus simple : Git Bash + quotes simples `'...'`.

---

## Idées d’amélioration (pistes de recherche)


---

## Glossaire (termes du README)

- **Artefact** : fichier généré par le pipeline (ex: modèle `.joblib`, métriques `.json`, rapports `.csv`).
- **Argmax** : opération qui choisit la classe ayant la probabilité la plus élevée.
- **Balanced / `class_weight`** : pondération des classes pour compenser un déséquilibre de labels.
- **Corpus** : ensemble des textes utilisés pour entraîner le modèle (ici construit depuis `manual_phrases.csv`).
- **Cross-validation (CV)** : évaluation par plusieurs découpages train/validation pour estimer la robustesse.
- **Dédoublonnage** : suppression/gestion de doublons ou quasi-doublons (souvent après normalisation).
- **F1-macro** : moyenne de F1 par classe en donnant le même poids à chaque classe (utile quand on veut que `medium` compte autant que les autres).
- **Feature (variable)** : nombre calculé à partir du texte (ex: ratio de kanji, ratio JLPT N3, etc.).
- **Faux positif** : un match détecté alors qu’il ne correspond pas vraiment au phénomène ciblé (ex: sous-chaîne trop courte qui “matche partout”).
- **Faux négatif** : un signal réel n’est pas détecté (ex: vocab JLPT présent mais sous une forme conjuguée non reconnue).
- **Fuite de données (data leak)** : situation où le modèle “voit” pendant l’entraînement des exemples identiques/équivalents à ceux du test/validation.
- **Group split / split groupé** : découpage train/test où des exemples liés (même groupe) ne sont jamais séparés, pour limiter la fuite.
- **Heuristique** : règle simple (non apprise) utilisée comme approximation (ex: inférer `飲みます → 飲む`).
- **Holdout / validation séparée** : jeu de données mis de côté et jamais utilisé pour entraîner, pour mesurer la généralisation.
- **Joblib** : format/fichier utilisé pour sauvegarder et recharger le modèle scikit-learn (`.joblib`).
- **JLPT N5→N1** : niveaux de difficulté du Japanese Language Proficiency Test (N5 = débutant, N1 = avancé).
- **Lemmatisation (lemme / forme dictionnaire)** : transformer un mot en sa forme “de base” (ex: `食べた → 食べる`).
- **Logistic Regression (régression logistique)** : classifieur linéaire qui calcule des scores puis des probabilités (sigmoïde en binaire, softmax en multi-classes).
- **Macro avg** : moyenne par classe non pondérée (chaque classe pèse pareil).
- **Matrice de confusion** : tableau “vrai vs prédit” qui montre les types d’erreurs (ex: `medium → hard`).
- **NFKC** : forme de normalisation Unicode qui réduit les variantes (fullwidth/halfwidth) pour stabiliser les lookups.
- **N-grammes de caractères** : sous-chaînes de longueur $n$ extraites du texte (utile en japonais sans espaces).
- **Overfitting** : quand le modèle apprend trop le train et généralise mal sur des données nouvelles.
- **Pipeline** : enchaînement reproductible d’étapes (dictionnaires → corpus → entraînement → validation).
- **Précision (precision)** : parmi les prédictions d’une classe, proportion de correct.
- **Rappel (recall)** : parmi les vrais exemples d’une classe, proportion retrouvée.
- **Softmax** : fonction qui transforme des scores multi-classes en probabilités qui somment à 1.
- **Split train/test** : découpe des données en “train” (apprentissage) et “test” (mesure interne).
- **Stratify / stratification** : découpe qui conserve les proportions de classes dans train et test.
- **Stoplist** : liste d’éléments trop fréquents/ambigus qu’on exclut pour éviter de biaiser les features.
- **Sous-chaîne (substring matching)** : test du type “token dans le texte”, pratique mais peut sur-matcher sans garde-fous.
- **TF-IDF** : pondération de texte qui valorise les n-grammes fréquents dans un document mais rares globalement.
- **Tokenisation** : découpage d’un texte en unités (tokens) ; en japonais, un tokenizer aide à trouver les frontières lexicales.
- **Weighted avg** : moyenne pondérée par le nombre d’exemples (les classes fréquentes comptent plus).