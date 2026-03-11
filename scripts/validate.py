"""Validation honnête sur un jeu de données séparé (anti-triche).

But:
- Évaluer le modèle sur des phrases qui ne sont PAS dans le corpus d'entraînement.
- Détecter les chevauchements (fuite de données) et PREVENIR.
    Par défaut: warning uniquement (ne bloque pas).
    Optionnel: --fail-on-overlap pour rendre ça bloquant.

Usage:
  ./.venv/Scripts/python scripts/validate.py
  ./.venv/Scripts/python scripts/validate.py --data data/input/validation_phrases.csv
  ./.venv/Scripts/python scripts/validate.py --json

Entrée (par défaut): data/input/validation_phrases.csv
Colonnes attendues: text,label (id optionnel)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Permet d'importer feature_extract depuis scripts/
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from feature_extract import extract_features

ROOT = Path(__file__).resolve().parent.parent


def _norm_text_strict(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def _is_punct(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch).startswith("P")


def _norm_text_loose(text: str) -> str:
    s = _norm_text_strict(text)
    return "".join(ch for ch in s if (not ch.isspace()) and (not _is_punct(ch)))


def _find_model_path(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    p = ROOT / "data" / "raw" / "baseline_model.joblib"
    if p.exists():
        return p

    raise FileNotFoundError(f"No baseline model found at: {p}")


def _build_feature_df(texts: Iterable[str]) -> pd.DataFrame:
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows).fillna(0.0)


def _load_texts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Validation CSV must contain columns: text, label")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(_norm_text_strict)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df


def _overlap_report(corpus_texts: List[str], val_texts: List[str]) -> Tuple[List[str], List[str]]:
    corpus_strict = {_norm_text_strict(t) for t in corpus_texts}
    corpus_loose = {_norm_text_loose(t) for t in corpus_texts}

    overlap_strict: List[str] = []
    overlap_loose: List[str] = []

    for t in val_texts:
        if _norm_text_strict(t) in corpus_strict:
            overlap_strict.append(t)
        elif _norm_text_loose(t) and _norm_text_loose(t) in corpus_loose:
            overlap_loose.append(t)

    return overlap_strict, overlap_loose


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the baseline model on a holdout CSV")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "input" / "validation_phrases.csv"),
        help="Path to validation CSV with columns text,label",
    )
    parser.add_argument("--model", type=str, default=None, help="Path to .joblib model (optional)")
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Fail (exit non-zero) if validation texts overlap with training corpus",
    )
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    args = parser.parse_args()

    val_path = Path(args.data)
    if not val_path.is_absolute():
        val_path = (ROOT / val_path).resolve()

    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    corpus_path = ROOT / "data" / "input" / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Missing corpus file: {corpus_path}. Run: ./.venv/Scripts/python scripts/corpus.py"
        )

    val_df = _load_texts(val_path)

    corpus_df = pd.read_csv(corpus_path)
    if "text" not in corpus_df.columns:
        raise ValueError("corpus.csv must contain column: text")

    corpus_texts = corpus_df["text"].dropna().astype(str).tolist()
    val_texts = val_df["text"].dropna().astype(str).tolist()

    overlap_strict, overlap_loose = _overlap_report(corpus_texts, val_texts)
    if overlap_strict or overlap_loose:
        msg = [
            "WARNING: le fichier de validation contient des phrases déjà présentes dans le corpus.",
            "Risque de fuite de données (validation 'trichée').",
        ]
        if overlap_strict:
            msg.append(f"- Overlap STRICT (exact): {len(overlap_strict)}")
            msg.extend([f"  * {t}" for t in overlap_strict[:5]])
        if overlap_loose:
            msg.append(f"- Overlap LOOSE (sans ponctuation/espaces): {len(overlap_loose)}")
            msg.extend([f"  * {t}" for t in overlap_loose[:5]])
        msg.append("Conseil: enlève ces phrases de la validation OU du corpus.")
        print("\n".join(msg), file=sys.stderr)
        if args.fail_on_overlap:
            raise SystemExit(2)

    model_path = _find_model_path(args.model)
    model = joblib.load(model_path)

    x_val = _build_feature_df(val_df["text"].tolist())
    y_true = val_df["label"].tolist()
    y_pred = [str(x) for x in model.predict(x_val)]

    metrics: Dict[str, Any] = {
        "n_val": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, output_dict=True),
        "model_path": str(model_path),
        "val_path": str(val_path),
        "overlap": {
            "strict": int(len(overlap_strict)),
            "loose": int(len(overlap_loose)),
            "fail_on_overlap": bool(args.fail_on_overlap),
        },
    }

    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    print(f"Model: {model_path}")
    print(f"Validation file: {val_path}")
    print(f"n_val={metrics['n_val']} | accuracy={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
