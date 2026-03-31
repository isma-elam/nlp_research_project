"""Entraîne un modèle baseline sur le corpus (easy / medium / hard).

Convention (manuelle-only) :
- Entrée : data/input/corpus.csv
- Sorties :
    - data/raw/baseline_model.joblib
    - data/raw/baseline_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Tuple

# Ajoute le dossier courant au PATH pour pouvoir importer les modules de `scripts/`.
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model_features import build_text_plus_numeric_model

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "input" / "corpus.csv"
MODEL_DIR = ROOT / "data" / "raw"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"

ALLOWED_LABELS = {"easy", "medium", "hard"}


def _is_punct(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("P")


def _norm_text_strict(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def _norm_text_loose(text: str) -> str:
    s = _norm_text_strict(text)
    return "".join(ch for ch in s if (not ch.isspace()) and (not _is_punct(ch)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner le modèle baseline de difficulté")
    parser.add_argument(
        "--no-group-split",
        action="store_true",
        help="Désactive le split groupé (anti-fuite). Reviens au train_test_split classique.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Pondération des classes pour LogisticRegression (utile si `medium` est minoritaire).",
    )
    parser.add_argument(
        "--medium-boost",
        type=float,
        default=1.0,
        help="Facteur multiplicatif appliqué au poids de la classe 'medium' (défaut: 1.0 = aucun boost).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Active un petit GridSearchCV (sur train uniquement) pour optimiser F1-macro.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Nombre de folds CV pour --tune (défaut: 5)",
    )
    return parser.parse_args()


def _group_train_test_split(
    X, y, *, groups, test_size: float, random_state: int
) -> Tuple[object, object, object, object, Dict[str, object]]:
    """Split avec groupes pour limiter la fuite via phrases quasi-identiques."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    meta: Dict[str, object] = {
        "method": "group_shuffle_loose",
        "test_size": float(test_size),
        "random_state": int(random_state),
        "n_groups": int(getattr(groups, "nunique", lambda: len(set(groups)))()),
    }
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], meta


def main() -> None:
    args = _parse_args()
    print(f"Python: {sys.executable}")
    print(f"scikit-learn: {sklearn.__version__}")
    if "\\.venv\\" not in str(sys.executable).lower() and "/.venv/" not in str(sys.executable).lower():
        print(
            "NOTE: Tu n'utilises pas le Python de la venv du projet (.venv). "
            "Ça peut provoquer des InconsistentVersionWarning au chargement du modèle. "
            "Utilise plutôt: .venv/Scripts/python scripts/train_model.py (Git Bash) "
            "ou .\\.venv\\Scripts\\python scripts\\train_model.py (PowerShell).",
            file=sys.stderr,
        )

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Fichier corpus introuvable : {CORPUS_PATH}")

    df = pd.read_csv(CORPUS_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("corpus.csv doit contenir les colonnes : text, label")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(_norm_text_strict)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    # Reset index pour garder X/y alignés après filtrage.
    df = df.reset_index(drop=True)

    unknown = sorted(set(df["label"]) - ALLOWED_LABELS)
    if unknown:
        raise ValueError(
            "Labels inconnus dans corpus.csv : "
            + ", ".join(unknown)
            + ". Attendu uniquement : easy, medium, hard"
        )

    present = sorted(set(df["label"]))
    if len(present) < 2:
        raise ValueError(
            "Il faut au moins 2 classes pour entraîner. Labels présents : " + ", ".join(present)
        )

    y = df["label"].astype(str)
    # Nouveau: le modèle consomme les textes bruts et extrait les features dans le Pipeline.
    x_text = df["text"].astype(str)

    split_meta: Dict[str, object]
    if not args.no_group_split:
        groups = df["text"].astype(str).map(_norm_text_loose)
        # On tente le split groupé seulement si ça a du sens.
        # Si presque tous les groupes sont uniques, GroupShuffleSplit n'apporte pas grand-chose
        # mais il casse la stratification (répartition des classes) -> métriques / counts surprenants.
        n_total = int(len(df))
        n_groups = int(groups.nunique())
        dup_fraction = 1.0 - (n_groups / n_total) if n_total else 0.0
        max_group_size = int(groups.value_counts().max()) if n_total else 0
        use_group_split = (
            n_total >= 30
            and n_groups >= 10
            and max_group_size >= 2
            and (dup_fraction >= 0.01 or max_group_size >= 3)
        )

        if use_group_split:
            try:
                X_train, X_test, y_train, y_test, split_meta = _group_train_test_split(
                    x_text, y, groups=groups, test_size=0.2, random_state=42
                )
            except Exception:
                split_meta = {"method": "train_test_split", "test_size": 0.2, "random_state": 42, "stratify": True}
                X_train, X_test, y_train, y_test = train_test_split(
                    x_text, y, test_size=0.2, random_state=42, stratify=y
                )
        else:
            split_meta = {"method": "train_test_split", "test_size": 0.2, "random_state": 42, "stratify": True}
            X_train, X_test, y_train, y_test = train_test_split(
                x_text, y, test_size=0.2, random_state=42, stratify=y
            )
    else:
        split_meta = {"method": "train_test_split", "test_size": 0.2, "random_state": 42}
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                x_text, y, test_size=0.2, random_state=42, stratify=y
            )
            split_meta["stratify"] = True
        except ValueError:
            # Sur petits datasets ou classes rares, stratify peut échouer.
            X_train, X_test, y_train, y_test = train_test_split(
                x_text, y, test_size=0.2, random_state=42, stratify=None
            )
            split_meta["stratify"] = False

    # class_weight: on le calcule après split, pour être cohérent avec y_train.
    class_weight: object | None
    class_weight_details: Dict[str, float] | None = None

    if args.class_weight == "balanced":
        classes = np.array(sorted(set(y_train.astype(str))), dtype=str)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.astype(str))
        class_weight_details = {str(c): float(v) for c, v in zip(classes, w)}
    else:
        class_weight_details = {str(c): 1.0 for c in sorted(set(y_train.astype(str)))}

    # Boost medium si demandé
    medium_boost = float(args.medium_boost)
    if class_weight_details is not None and medium_boost and abs(medium_boost - 1.0) > 1e-9:
        if "medium" in class_weight_details:
            class_weight_details["medium"] = float(class_weight_details["medium"]) * medium_boost

    # Par défaut (réaliste): pas de pondération.
    if args.class_weight == "none" and abs(float(args.medium_boost) - 1.0) < 1e-9:
        class_weight = None
    else:
        class_weight = class_weight_details if class_weight_details else None

    base_model = build_text_plus_numeric_model(class_weight=class_weight)

    tuning: Dict[str, object] = {"enabled": bool(args.tune), "cv": int(args.cv)}

    if args.tune and int(args.cv) >= 2:
        cv = StratifiedKFold(n_splits=int(args.cv), shuffle=True, random_state=42)
        param_grid = {
            "clf__C": [0.3, 1.0, 3.0, 10.0],
        }
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train.tolist(), y_train)
        model = search.best_estimator_
        tuning["best_params"] = dict(search.best_params_)
        tuning["best_score_f1_macro"] = float(search.best_score_)
    else:
        model = base_model
        model.fit(X_train.tolist(), y_train)

    y_pred = model.predict(X_test.tolist())
    metrics: Dict[str, object] = {
        "classes_present": present,
        "split": {
            **split_meta,
            "n_total": int(len(df)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "label_counts_total": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
            "label_counts_train": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
            "label_counts_test": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
        },
        "train": {
            "class_weight": (
                None
                if (args.class_weight == "none" and float(args.medium_boost) == 1.0)
                else (class_weight_details if class_weight_details else args.class_weight)
            ),
            "medium_boost": float(args.medium_boost),
            "tuning": tuning,
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Métriques sauvegardées : {METRICS_PATH}")
    print(f"Exactitude: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
