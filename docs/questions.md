# Questions / Réponses — Justification du MVP et des choix techniques

Ce document répond aux questions fréquentes sur : pourquoi un MVP, pourquoi un petit dataset manuel, et pourquoi les bibliothèques/outils utilisés dans ce projet.

Date : 2026-02-16

---

## 1) MVP : pourquoi ?

### Q1 — Pourquoi faire un MVP plutôt qu’un produit “complet” tout de suite ?
**R :** Parce qu’un MVP vise d’abord l’**apprentissage validé** (valider une hypothèse avec un effort minimal), ce qui réduit le risque de passer beaucoup de temps sur une solution qui ne marche pas.
- Un MVP est une version avec **assez de fonctionnalités** pour attirer des premiers utilisateurs et **valider l’idée tôt**.
- L’objectif central est de **collecter un maximum d’apprentissage validé avec un minimum d’effort**.

Sources :
- Agile Alliance — *Minimum Viable Product (MVP)* : https://www.agilealliance.org/glossary/mvp/
- ProductPlan — *Minimum Viable Product (MVP)* : https://www.productplan.com/glossary/minimum-viable-product/

### Q2 — En quoi le MVP est adapté à une contrainte de temps (≈135h) ?
**R :** Avec une contrainte de temps, le risque principal est de **ne pas livrer un pipeline complet** (données → features → modèle → évaluation → sauvegarde). Le MVP force à livrer une boucle de bout en bout mesurable, même si la performance n’est pas encore optimale.
- On obtient rapidement : un dataset contrôlé, un modèle baseline, des métriques, des sorties reproductibles.

Source (cadre général MVP) :
- Agile Alliance : https://www.agilealliance.org/glossary/mvp/

### Q3 — Pourquoi votre MVP “interprétable” plutôt qu’un gros modèle profond (Transformer) dès le départ ?
**R :** Le choix “baseline interprétable” est cohérent avec un MVP :
- Moins de dépendances (GPU, gros corpus, coûts).
- Meilleure explicabilité (features lisibles : ratio kanji/kana, occurrences JLPT, etc.).
- Pipeline plus simple à déboguer et à justifier.

Source (cadre MVP) :
- Agile Alliance : https://www.agilealliance.org/glossary/mvp/

---

## 2) Données : pourquoi un mini dataset manuel ?

### Q4 — Pourquoi ne pas scraper NHK / Wikipedia pour construire le corpus ?
**R :** Parce que le scraping est souvent **instable** (anti-bot/WAF, changements HTML, blocages 401/403), donc difficile à rendre reproductible et à garantir dans un projet court. Le MVP privilégie un corpus manuel qui :
- Est 100% reproductible.
- Permet de valider toute la chaîne technique sans dépendre d’une source externe.

Source (principe MVP : apprendre vite avec peu d’effort, réduire le risque) :
- Agile Alliance : https://www.agilealliance.org/glossary/mvp/

### Q5 — Un dataset de 10 phrases, ce n’est pas “trop petit” ?
**R :** Oui pour faire de la performance. Non pour valider un MVP.
- Ce dataset sert à valider le **fonctionnement du pipeline**, pas à produire un estimateur fiable pour production.
- Avec très peu d’exemples, l’évaluation est bruitée : on documente cette limite et on prévoit une montée en taille plus tard.

Référence méthodo (évaluation, prudence sur la généralisation) :
- scikit-learn — *Cross-validation* : https://scikit-learn.org/stable/modules/cross_validation.html

### Q6 — Pourquoi faire un `train_test_split` sur un petit corpus ?
**R :** Même sur petit corpus, séparer entraînement/test force à **mesurer sur des données non vues**. C’est une discipline utile (et standard) pour éviter de se “tromper” en mesurant sur les données d’entraînement.

Source :
- scikit-learn — *Cross-validation* (train/test split, principes) : https://scikit-learn.org/stable/modules/cross_validation.html

---

## 3) Features : pourquoi ces caractéristiques ?

### Q7 — Pourquoi extraire des features “explicables” au lieu d’embedder le texte ?
**R :** Pour un MVP, les features explicables facilitent :
- La justification scientifique (liens avec kanji/kana, JLPT).
- Le débogage (on voit quelles features changent).
- L’interprétation du modèle (poids d’un classifieur linéaire).

Source (cadre général de baseline linéaire) :
- scikit-learn — *Logistic Regression* : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

### Q8 — Pourquoi compter les kanji / hiragana / katakana ?
**R :** C’est une heuristique simple et stable : le ratio kanji/kana et la structure de phrases peuvent corréler avec le niveau perçu, et surtout elles sont faciles à calculer et à expliquer.

(Heuristique de projet ; pas une “preuve” universelle. Le MVP sert à tester cette hypothèse.)

### Q9 — Pourquoi utiliser des dictionnaires JLPT (vocabulaire/grammaire) et une liste de kanji (joyo) ?
**R :** Ces ressources fournissent une base de “difficulté par niveau” permettant de construire des features comptables (ex : nb de mots N5 vs N1). Dans un MVP, c’est une manière pragmatique de rendre le modèle explicable.

---

## 4) Pourquoi Python et ces bibliothèques ?

### Q10 — Pourquoi Python ?
**R :** Python est très utilisé en data/ML, avec un écosystème mature (pandas, NumPy, scikit-learn) qui accélère fortement la mise en place d’un pipeline complet.

### Q11 — Pourquoi `pandas` ?
**R :** `pandas` est adapté aux données tabulaires (CSV) et aux opérations de nettoyage/filtrage (dropna, dedup, etc.).

Source :
- pandas — *Getting started / Overview* : https://pandas.pydata.org/docs/getting_started/overview.html

### Q12 — Pourquoi `NumPy` ?
**R :** NumPy fournit le tableau `ndarray` et les opérations vectorisées ; c’est une base de l’écosystème scientifique Python et un socle utilisé par beaucoup de bibliothèques ML.

Source :
- NumPy — *What is NumPy?* : https://numpy.org/doc/stable/user/whatisnumpy.html

### Q13 — Pourquoi `scikit-learn` ?
**R :** scikit-learn fournit des modèles baselines robustes, des outils de validation (split/CV) et des métriques standards.

Sources :
- Logistic Regression : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Model evaluation (accuracy, F1, reports) : https://scikit-learn.org/stable/modules/model_evaluation.html
- Cross-validation : https://scikit-learn.org/stable/modules/cross_validation.html

### Q14 — Pourquoi une régression logistique (`LogisticRegression`) ?
**R :** C’est un classifieur linéaire standard (baseline) souvent utilisé pour démarrer :
- Rapide, peu coûteux.
- Interprétable via les coefficients.
- Bon point de comparaison avant des modèles plus complexes.

Source :
- scikit-learn — *Logistic Regression* : https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

### Q15 — Pourquoi `accuracy` et `F1-macro` ?
**R :**
- `accuracy` est simple mais peut être trompeuse en cas de classes déséquilibrées.
- `F1-macro` moyenne le F1 **par classe** puis moyenne, ce qui donne un signal plus “équitable” quand les classes sont déséquilibrées.
- `classification_report` donne une vue détaillée précision/rappel/F1.

Source :
- scikit-learn — *Model evaluation* : https://scikit-learn.org/stable/modules/model_evaluation.html

### Q16 — Pourquoi `joblib` pour sauvegarder le modèle ?
**R :** `joblib` est un choix courant pour persister des objets Python (souvent efficaces pour des objets contenant de gros tableaux NumPy). Mais il faut être conscient des limitations et risques.

Sources :
- scikit-learn — *Model persistence* (comparaison et avertissements) : https://scikit-learn.org/stable/model_persistence.html
- joblib — *Persistence* (dump/load + avertissements) : https://joblib.readthedocs.io/en/latest/persistence.html

### Q17 — Quels sont les risques de `joblib` / pickle ?
**R :** Charger un fichier pickle/joblib non fiable peut exécuter du code arbitraire. Il faut donc :
- Ne charger que des fichiers produits localement / de confiance.
- Versionner l’environnement (versions de dépendances).

Sources :
- scikit-learn — *Model persistence* : https://scikit-learn.org/stable/model_persistence.html
- joblib — *Persistence* : https://joblib.readthedocs.io/en/latest/persistence.html

---

## 5) Reproductibilité, qualité et limites

### Q18 — Comment rendre le projet reproductible ?
**R :**
- Dataset local stable (manuel).
- Scripts déterministes quand possible (`random_state=42`).
- Sorties sauvegardées (modèle + métriques) pour comparer les runs.

Source (aléas/validation) :
- scikit-learn — *Cross-validation* : https://scikit-learn.org/stable/modules/cross_validation.html

### Q19 — Pourquoi documenter des limitations “fortes” dans le MVP ?
**R :** Un MVP ne promet pas la performance finale : il prouve la faisabilité et clarifie les risques.
- Sur petit corpus, les métriques sont instables.
- Les features par “présence de motifs” (ex : grammaire/vocab) sont simplifiées.

Source (MVP et apprentissage/itération) :
- Agile Alliance : https://www.agilealliance.org/glossary/mvp/

### Q20 — Pourquoi parler d’alternatives (transformers, tokenizers japonais, etc.) si on ne les implémente pas ?
**R :** Parce qu’un bon dossier justifie aussi ce qui a été volontairement écarté au MVP.
- Alternatives possibles : modèles profonds, segmentation morphologique (MeCab), embeddings.
- Raisons de report : temps, dépendances, besoin d’un corpus plus large.

---

## 6) Questions “d’architecture projet”

### Q21 — Pourquoi un pipeline en scripts (ETL → features → train) ?
**R :** C’est la structure la plus simple pour un MVP reproductible :
- Étape 1 : construire le corpus (`scripts/corpus.py`).
- Étape 2 : extraire les features (`scripts/feature_extract.py`).
- Étape 3 : entraîner + métriques + sauvegarde (`scripts/train_model.py`).

### Q22 — Pourquoi sauvegarder des métriques en JSON ?
**R :** JSON est lisible, versionnable, et facile à comparer entre runs (diff Git) ; scikit-learn expose aussi des rapports sous forme de dict.

Source (report) :
- scikit-learn — *Model evaluation* : https://scikit-learn.org/stable/modules/model_evaluation.html

---

## Bibliographie (liens)

- Agile Alliance. *Minimum Viable Product (MVP).* https://www.agilealliance.org/glossary/mvp/ (consulté le 2026-02-16)
- ProductPlan. *Minimum Viable Product (MVP).* https://www.productplan.com/glossary/minimum-viable-product/ (consulté le 2026-02-16)
- scikit-learn. *Logistic Regression.* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression (consulté le 2026-02-16)
- scikit-learn. *Model evaluation: quantifying the quality of predictions.* https://scikit-learn.org/stable/modules/model_evaluation.html (consulté le 2026-02-16)
- scikit-learn. *Cross-validation: evaluating estimator performance.* https://scikit-learn.org/stable/modules/cross_validation.html (consulté le 2026-02-16)
- scikit-learn. *Model persistence.* https://scikit-learn.org/stable/model_persistence.html (consulté le 2026-02-16)
- pandas. *Getting started — Overview.* https://pandas.pydata.org/docs/getting_started/overview.html (consulté le 2026-02-16)
- NumPy. *What is NumPy?* https://numpy.org/doc/stable/user/whatisnumpy.html (consulté le 2026-02-16)
- joblib. *Persistence.* https://joblib.readthedocs.io/en/latest/persistence.html (consulté le 2026-02-16)
- Wikipedia. *Minimum viable product.* https://en.wikipedia.org/wiki/Minimum_viable_product (consulté le 2026-02-16)
