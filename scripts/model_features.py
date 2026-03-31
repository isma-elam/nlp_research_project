"""Constructeur de matrice de features partagé (train / predict / validate).

Objectifs :
- Garantir que le modèle voit exactement les mêmes colonnes (sélection + ordre)
    à l'entraînement et à l'inférence.
- Pouvoir retirer de l'entrée ML certaines features qui dominent trop souvent
    les prédictions (ex: proxy de longueur), tout en les gardant disponibles pour
    l'explicabilité via `extract_features`.
"""

from __future__ import annotations

import re
from typing import Iterable, Set

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract import extract_features


# Features observées comme trop dominantes (peuvent pousser en `hard` même sur des
# phrases pas si difficiles). On les retire seulement de l'entrée du modèle :
# elles restent disponibles dans `extract_features` pour inspection/debug.
DROP_FEATURES_FOR_MODEL: Set[str] = {
    # Proxys de structure/volume (peuvent dominer). On garde désormais
    # longueur + totaux vocab/grammaire car c'est souvent un signal utile
    # pour distinguer `easy` vs `medium` sur des phrases plus longues.
    "kanji_total",
    # Les signaux de niveaux kanji (mapping joyo/JLPT) peuvent être trompeurs et
    # sur-dominer des phrases simples contenant des kanji courants.
    "kanji_max_level_num",
    "kanji_jlpt_avg_num",
    # On garde ponctuation + nombre de phrases : ça reflète parfois la structure.
}


_DROP_REGEXES = [
    # On retire les compteurs absolus par niveau ; on garde les ratios.
    re.compile(r"^(vocab|kanji)_N[1-5]$"),
    # Les ratios kanji par niveau peuvent aussi dominer sur phrases courtes.
    re.compile(r"^kanji_ratio_N[1-5]$"),
    # Features réservées à l'explicabilité (pas d'entrée ML).
    re.compile(r"^particle_"),
]


def build_feature_df_for_model(texts: Iterable[str]) -> pd.DataFrame:
    rows = [extract_features(t) for t in texts]
    df = pd.DataFrame(rows).fillna(0.0)

    drop_cols = {c for c in DROP_FEATURES_FOR_MODEL if c in df.columns}
    for col in df.columns:
        if any(rx.match(str(col)) for rx in _DROP_REGEXES):
            drop_cols.add(str(col))

    drop_cols = sorted(drop_cols)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ordre de colonnes déterministe (reproductible).
    return df.reindex(sorted(df.columns), axis=1)


def build_feature_row_for_model(text: str) -> pd.DataFrame:
    return build_feature_df_for_model([text])


class TextToNumericFeatures(BaseEstimator, TransformerMixin):
    """Transforme une liste de textes en matrice numérique de features explicables.

    But:
    - Rester compatible scikit-learn (Pipeline/FeatureUnion/GridSearchCV)
    - Produire des colonnes stables (mêmes features et même ordre)
    """

    feature_names_: list[str]

    def fit(self, X, y=None):  # noqa: N803
        _ = y
        # On apprend l'ordre des colonnes sur X d'entraînement.
        df = build_feature_df_for_model(list(X))
        self.feature_names_ = list(df.columns)
        return self

    def transform(self, X):  # noqa: N803
        df = build_feature_df_for_model(list(X))

        # Aligner sur les colonnes vues en fit (sécurité).
        if hasattr(self, "feature_names_"):
            for c in self.feature_names_:
                if c not in df.columns:
                    df[c] = 0.0
            extra = [c for c in df.columns if c not in self.feature_names_]
            if extra:
                df = df.drop(columns=extra)
            df = df[self.feature_names_]

        return df.astype(float).to_numpy()


def build_text_plus_numeric_model(*, class_weight=None) -> Pipeline:
    """Pipeline robuste: features numériques + TF-IDF char n-grams.

    Pourquoi:
    - Les features explicables sont utiles mais parfois insuffisantes pour séparer `medium`.
    - Les char n-grams capturent des motifs de surface (kana/kanji, okurigana, tournures) sans tokenizer.
    """

    numeric = Pipeline(
        steps=[
            ("num", TextToNumericFeatures()),
            # with_mean=False pour compatibilité sparse (FeatureUnion -> hstack)
            ("scaler", StandardScaler(with_mean=False)),
        ],
        memory=None,
    )

    char_ngrams = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=2,
        sublinear_tf=True,
        max_features=50000,
    )

    features = FeatureUnion(
        transformer_list=[
            ("numeric", numeric),
            ("char", char_ngrams),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=8000,
        random_state=42,
        class_weight=class_weight,
    )

    return Pipeline(steps=[("features", features), ("clf", clf)], memory=None)
