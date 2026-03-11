"""Entraîne un modèle baseline sur le corpus (easy / medium / hard).

Convention (manuelle-only) :
- Entrée : data/input/corpus.csv
- Sorties :
    - data/raw/baseline_model.joblib
    - data/raw/baseline_metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

# Add current script directory to path to allow importing feature_extract
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract import extract_features

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "input" / "corpus.csv"
MODEL_DIR = ROOT / "data" / "raw"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "baseline_model.joblib"
METRICS_PATH = MODEL_DIR / "baseline_metrics.json"

ALLOWED_LABELS = {"easy", "medium", "hard"}


def build_feature_df(texts: pd.Series) -> pd.DataFrame:
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows).fillna(0.0)


def main() -> None:
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
        raise FileNotFoundError(f"Missing corpus file: {CORPUS_PATH}")

    df = pd.read_csv(CORPUS_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("corpus.csv must contain columns: text, label")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    unknown = sorted(set(df["label"]) - ALLOWED_LABELS)
    if unknown:
        raise ValueError(
            "Unknown labels in corpus.csv: "
            + ", ".join(unknown)
            + ". Expected only: easy, medium, hard"
        )

    present = sorted(set(df["label"]))
    if len(present) < 2:
        raise ValueError(
            "Need at least 2 classes to train. Present labels: " + ", ".join(present)
        )

    X = build_feature_df(df["text"])
    y = df["label"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        stratified = True
    except ValueError:
        # Sur petits datasets ou classes rares, stratify peut échouer.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        stratified = False

    # Features mélangeant ratios et compteurs : on scale pour une convergence plus stable.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=20000, random_state=42)),
        ],
        memory=None,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics: Dict[str, object] = {
        "classes_present": present,
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratify": bool(stratified),
            "n_total": int(len(df)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "label_counts_total": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
            "label_counts_train": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
            "label_counts_test": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
        },
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
