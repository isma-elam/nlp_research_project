# Projet de 3e année de baccalauréat en informatique - 135h


# Dataset et features — MVP (Minimum Viable Product)

## 1) Cible (approche word-level scoring)
- **Analyse au mot** : chaque mot/kanji/grammaire est classé par niveau JLPT (N5 → N1).
- **Coloriation** (inspiré de la cotation de beta crux à l'escalade):
  - Vert = N5
  - Bleu = N4
  - Violet = N3
  - Rouge = N2
  - Rose fluo = N1 (avancé)
- **Score global** : % de mots par niveau → détermine "facile / moyen / difficile".
- Entrée : phrase/paragraphe d'**importe quelle longueur**.

## 2) Sources de données

### A) Dictionnaires/listes par niveau JLPT
Besoin de 3 ressources :

#### 1. Vocabulaire JLPT
Sur kaggle :
   https://www.kaggle.com/datasets/robinpourtaud/jlpt-words-by-level/data

**Recommandé pour MVP** : Deck déjà constitué (N5, N4, N3, N2, N1) au format CSV. Facile à normaliser.

#### 2. Kanji JLPT
Sur GitHub : https://github.com/NHV33/joyo-kanji-compilation/tree/master

#### 3. Grammaire JLPT
Crée moi-même à partir d'un livre d'apprentissage nommé JLPT Sensei.

### B) Corpus de test
Le MVP utilise un petit corpus manuel, maintenu dans un CSV.

- **Source MVP** : `data/input/manual_phrases.csv`
  - Colonnes attendues : `text`, `label` et`target_jlpt` pour documentation
  - `label` : `easy` / `medium` / `hard` (classification 3 classes)
- **Objectif** : valider le pipeline complet (construction corpus → features → modèle → métriques) de façon **reproductible**, sans dépendre de scraping.

### C) Collecte de données (hors scope MVP)
- Hors scope MVP : toute collecte web (scraping) et toutes les sources externes instables.
- Si des scripts de collecte existent encore dans le dépôt (ex. archivés dans `CODE_AMBITIEUX/`), ils ne font pas partie du MVP.

### D) Scripts de transformation
#### 1. Pipeline de Dictionnaires
Les scripts `nettoyage_*.py` transforment les CSV sources en fichiers JSON optimisés pour la recherche rapide :
- `data/json/jlpt_vocab.json` : Vocabulaire classé par niveau.
- `data/json/joyo.json` : Kanjis classés par niveau.
- `data/json/jlpt_grammar.json` : Patterns grammaticaux.
- `data/json/joyo_meta.json` : Kanjis classés par niveaux avec fréquence et nb ligne.

#### 2. Extraction de Features (`scripts/feature_extract.py`)
Le "Cerveau" du projet. Transforme un texte brut en vecteur mathématique :
- **Stats Basiques** : Ratio Kanjis/Hiragana/Katakana, longueur moyenne des phrases et fréquence.
- **Stats JLPT** : Comptage des occurrences de vocabulaire et kanji par niveau (N5..N1).
- Ces features servent d'entrée à l'algo ML.

#### 3. Construction du Corpus (`scripts/corpus.py`)
Ce script est **l'assembleur final** des données brutes. Indispensable pour l'entraînement.
- **Rôle (MVP)** : transforme `data/input/manual_phrases.csv` en `data/input/corpus.csv`.
- **Action** :
  1. Charge le CSV manuel et valide les colonnes (`text`, `label`).
  2. Nettoie (valeurs vides, doublons éventuels, normalisation simple).
  3. Écrit un fichier `data/input/corpus.csv` standardisé.
- **Résultat** : CSV prêt pour l'entraînement

#### 4. Entraînement du Modèle (`scripts/train_model.py`)
C'est le script qui **crée l'IA** à proprement parler.
- **Entrée** : `data/input/corpus.csv`.
- **Processus** :
  1. **Vectorisation** : Convertit chaque texte en chiffres grâce à `feature_extract.py`.
  2. **Split** : Sépare les données : 80% pour apprendre, 20% pour vérifier si ça marche (Test Set).
  3. **Apprentissage** : Utilise une **Régression Logistique** pour apprendre la séparation entre `easy` / `medium` / `hard`.
- **Sortie** :
  - `data/raw/baseline_model.joblib` : Le fichier du modèle sauvegardé (le "cerveau" entraîné).
  - `data/raw/baseline_metrics.json` : Le bulletin de notes du modèle (Accuracy, F1-macro, rapport).

#### 5. Prédiction / Démo (`scripts/predict.py`)
Script “démo” : charge le modèle `data/raw/baseline_model.joblib` et renvoie :
- `label` (`easy`/`medium`/`hard`)
- `proba` (si disponible)
- `score_easy_0_100` (haut = facile)
- `difficulty_0_100` (haut = difficile)
- aperçu de quelques features explicables


## Comprendre comment le modèle “trouve” la difficulté

### 1) Ce que le modèle voit vraiment
Le modèle ne voit pas “du japonais” directement. Il voit un tableau de nombres produit par `scripts/feature_extract.py`, par exemple :
- ratios d’écriture : `ratio_kanji`, `ratio_hira`, `ratio_kata` (**ponctuation et espaces exclus du dénominateur**)
- ponctuation (signaux séparés) : `punct_count`, `comma_count`, `ratio_punct`
- structure : `sent_count`, `sent_avg_len`
- “niveaux” JLPT trouvés dans le texte : `vocab_total`, `grammar_total`, `kanji_total`, + par niveau `N5..N1`
- complexité kanji : `kanji_strokes_avg`, `kanji_freq_avg`, etc. (via `data/json/joyo_meta.json`)

### 2) Ce qu’il apprend
Avec le corpus, il apprend des corrélations du type :
- si `grammar_total` est élevé + phrases longues + beaucoup de kanji → souvent `hard`
- si vocab `N5/N4` dominant + phrases plus courtes → souvent `easy`
- les cas “entre-deux” peuvent tomber en `medium`

La régression logistique apprend des **poids** (un poids par feature), et calcule une probabilité :

P(easy | 𝑥) = sigma(𝑤 𝑥 + 𝑏)

- 𝑥 (features) : Vecteur des variables d’entrée (longueur d’un texte, nombre de mots techniques, fréquence de certains termes)
- Vecteur de paramètres appris pendant l’entraînement.
Chaque poids mesure l’influence d’une feature sur la probabilité que la classe soit easy.
Si 𝑤 i > 0 -> la feature augmente la probabilité
Si 𝑤 i < 0 -> elle la diminue

- 𝑏 : Constante qui décale la frontière de décision.

- 𝑤 𝑥 : Produit scalaire 

- sigma : fonction sigmoide
Elle transforme un score réel en probabilité entre 0 et 1.

Si 𝑧≫0 → probabilité proche de 1

Si 𝑧≪0→ probabilité proche de 0

**Intuition globale**
On calcule un score linéaire 
On l’envoie dans la sigmoïde
On obtient une probabilité que l’exemple soit "easy"
Si la probabilité > 0.5 → on prédit "easy"
Sinon → "not easy"

### 3) Pourquoi tu as besoin des labels
Les labels `easy/hard` sont le “prof” : ils disent au modèle quand il a raison/faux pendant l’entraînement. Sans labels, il ne peut pas apprendre la frontière.


## Score sur 100 et pourcentage de difficulté

On définit un score “facile” (haut = facile) à partir des probabilités du modèle.

En 3 classes (easy/medium/hard) :
- `score_easy_0_100 = 100 × ( P(easy) + 0.5 × P(medium) )`

En 2 classes (easy/hard) :
- `score_easy_0_100 = 100 × P(easy)`

Ensuite on transforme ce score en 3 bandes (heuristique, ajustable) :
- `score_easy < 33` → difficile
- `33 ≤ score_easy < 66` → moyen
- `score_easy ≥ 66` → facile

Le pourcentage de difficulté est simplement :
- `difficulty_0_100 = 100 − score_easy_0_100`


### Pipeline MVP validé (manuel-only)
- Décision : abandon du scraping (NHK/Wikipedia) pour le MVP, car blocages 401/403/WAF → non reproductible.
- Données : corpus manuel stable dans `data/input/manual_phrases.csv`, standardisé en `data/input/corpus.csv`.
- Dataset : 33 phrases par classe (`easy`/`medium`/`hard`), soit 99 lignes au total → entraînement 3 classes réellement actif.
- Features : ajout des métadonnées kanji (traits + fréquence) via `data/json/joyo_meta.json`.
- Artefacts : choix de stocker le modèle et les métriques dans `data/raw/` (simple, compatible avec les scripts).
- Validation : entraînement OK → création de `data/raw/baseline_model.joblib` + `data/raw/baseline_metrics.json`.
- Résultat test : les métriques deviennent interprétables dès que le corpus grossit (ex: avec 99 phrases et `test_size=0.2`, on a 20 exemples en test).

- Objectif : taper une phrase japonaise → obtenir `easy/medium/hard` + probas + score facile sur 100.
- Implémentation : score `score_easy_0_100` basé sur les probas + bande : `<33 difficile`, `33–66 moyen`, `>=66 facile`.


### L'approche de ML dans le projet (on partira sur du ML)

Sans ML (approche basique)
Tu analyses un texte → tu comptes % N5/N4/N3/N2/N1 → tu dis "facile/moyen/difficile" selon des seuils fixes.
Problème : ça ignore la longueur des phrases, la complexité syntaxique, le vocabulaire hors-JLPT.

Avec ML (approche complète)
Tu entraînes un modèle sur des exemples **étiquetés** (dans le MVP : `manual_phrases.csv`).
Le modèle apprend des frontières `easy` / `medium` / `hard` à partir des features (ratios kana/kanji + stats JLPT + etc.).

Résultat concret pour l'utilisateur
Quand on analyse un texte :

Stats JLPT : "45% N5/N4, 30% N3, 25% N2/N1"
Prédiction ML : "difficile" (le modèle voit des patterns complexes)
Score final explicable : "Texte DIFFICILE - raison : 25% vocab avancé + phrases complexes détectées par le modèle"

## Comment lancer le MVP (end-to-end)
Depuis la racine du projet :

1) Construire le corpus :
- `python scripts/corpus.py`

2) Entraîner le modèle baseline :
- `python scripts/train_model.py`

⚠️ Important (Windows) : utilise **le même Python** pour `train_model.py` et `predict.py`.
Sinon tu peux avoir des warnings du type `InconsistentVersionWarning` (modèle picklé avec scikit-learn X, chargé avec scikit-learn Y).

- PowerShell (commande “sûre”, sans dépendre de l’activation) : `./.venv/Scripts/python scripts/train_model.py`
- Git Bash (commande “sûre”, sans dépendre de l’activation) : `./.venv/Scripts/python scripts/train_model.py`

3) Tester une prédiction (une phrase -> easy/medium/hard) :
- `python scripts/predict.py --text '猫が好きです。'`

Note : `scripts/predict.py` affiche maintenant un WARNING si la phrase est déjà dans `data/input/corpus.csv`.
Ça évite de faire une “validation” involontairement sur une phrase déjà vue au train/test.
Tu peux désactiver ça avec `--no-leak-check`.

4) Validation honnête (jeu séparé, hors corpus) :
- `python scripts/validate.py` (par défaut lit `data/input/validation_phrases.csv`)

Pourquoi c'est important ?
- Si tu testes une phrase qui est déjà dans `data/input/corpus.csv`, le modèle l'a potentiellement déjà “vue” pendant l'entraînement.
- Ça donne une fausse impression de performance (fuite de données).

Le script `scripts/validate.py` **préviens** si une phrase de validation est aussi présente dans le corpus (fuite de données).

La commande `python scripts/predict.py --text "..."` est la façon normale d'utiliser le modèle.

- `train_model.py` : entraîne et sauvegarde le modèle (il ne “répond” pas à une phrase).
- `predict.py` : charge le modèle sauvegardé et donne une réponse pour une phrase.

Quand on dit “test”, c'est juste qu'on a utilisé une phrase connue pour vérifier rapidement que tout fonctionne.

### Problème rencontré : `????`
Sur Windows + terminal intégré VS Code, deux problèmes différents peuvent arriver :

1) **`????` dans `--text`**
- Cause : le terminal a remplacé les caractères japonais avant que Python ne reçoive la chaîne.
- Conséquence : le modèle analyse une phrase composée de `?`, donc les features ne veulent plus rien dire.

### Solution utilisé : Git Bash

1) Activer l'environnement virtuel (une fois par terminal) :
```bash
source .venv/Scripts/activate
```

Astuce si tu vois un `InconsistentVersionWarning` : ré-entraîne avec la venv explicitement :
```bash
./.venv/Scripts/python scripts/train_model.py
```

2) Lancer la prédiction
```bash
python scripts/predict.py --text '今日はいい天気です。'
```

### Exemple de résultat (réel)
Commande :
```bash
python scripts/predict.py --text '猫が好きです。'
```

Exemple de sortie :
```text
Model: ...\data\raw\baseline_model.joblib
Prediction: easy
Score (facile): 100.0/100  ->  facile
Difficulté: 0.0/100
Proba: easy=1.000, hard=0.000, medium=0.000
Features (aperçu):
  - len_chars: 11.0
  - ratio_kanji: 0.36363636363636365
  - ratio_hira: 0.45454545454545453
  - ratio_kata: 0.0
  - sent_count: 2.0
  - sent_avg_len: 5.0
  - vocab_total: 9.0
  - kanji_total: 4.0
  - grammar_total: 2.0
```

Interprétation rapide :
- `Prediction` : classe finale (`easy/medium/hard`).
- `Proba` : confiance du modèle par classe.
- `Score (facile)` : score 0–100 (haut = facile).
- `Difficulté` : 100 - score facile.

### Glossaire — `data/raw/baseline_metrics.json`
Ce fichier est écrit par `scripts/train_model.py` après l'entraînement. Il sert de **bulletin de notes** du modèle sur le **test set**.

#### 1) `classes_present`
Liste des classes réellement présentes dans le dataset au moment du train.

#### 2) `split` (comment on a coupé les données)
- `test_size` : part utilisée pour le test. Ici `0.2` = 20% test, donc 80% train.
- `random_state` : graine pour rendre le split reproductible.
- `stratify` : `true` = on garde la proportion des classes dans train/test.
- `n_total` : nombre total d'exemples utilisés.
- `n_train` : nombre d'exemples en train.
- `n_test` : nombre d'exemples en test.
- `label_counts_total` : répartition des labels sur tout le dataset.
- `label_counts_train` : répartition des labels sur le train.
- `label_counts_test` : répartition des labels sur le test.

#### 3) `accuracy`
Pourcentage d'exemples du test set correctement prédits.

#### 4) `f1_macro`
F1 moyen en donnant le même poids à chaque classe (utile en multi-classes pour ne pas “oublier” la classe `medium`).

#### 5) `report` (détails par classe)
Pour chaque classe (`easy`, `medium`, `hard`) :
- `precision` : parmi les prédictions “classe X”, combien sont correctes.
- `recall` : parmi les vrais “classe X”, combien sont retrouvés.
- `f1-score` : compromis precision/recall (plus c'est haut, mieux c'est).
- `support` : nombre d'exemples de la classe dans le test set.

Puis des agrégats :
- `macro avg` : moyenne simple des classes (chaque classe pèse pareil).
- `weighted avg` : moyenne pondérée par `support` (les classes fréquentes pèsent plus).

Important : `baseline_metrics.json` n'est pas utilisé pour faire une prédiction. Il sert uniquement à analyser la qualité du modèle entraîné.




## Notes d'idées à intégrer plus tard
- **Conjugaisons** : gérer les verbes (forme dictionnaire) avant lookup JLPT.
- **Compteurs** : ajouter une liste de compteurs fréquents (N5→N2) et les détecter comme patterns.
- **Formes grammaticales** : règles simples pour 〜ている, 〜られる, etc. améliore la couverture.