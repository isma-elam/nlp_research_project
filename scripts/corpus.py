"""Construit un corpus MVP à partir de phrases manuelles (CSV).

Objectif : éviter toute dépendance à des sources web.

Convention :
- Entrée  : data/input/manual_phrases.csv
- Sortie  : data/input/corpus.csv

Le fichier `corpus.csv` est une version standardisée (nettoyée) du dataset manuel.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

MANUAL_PATH = INPUT_DIR / "manual_phrases.csv"
OUT_PATH = INPUT_DIR / "corpus.csv"


def main() -> None:
    """Construit data/input/corpus.csv à partir de data/input/manual_phrases.csv."""
    if not MANUAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing manual dataset: {MANUAL_PATH}. Create it with columns text,label."
        )

    print(f"Loading manual dataset {MANUAL_PATH}...")
    df = pd.read_csv(MANUAL_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("manual_phrases.csv must contain columns: text, label")

    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate text rows")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved manual corpus ({len(df)} rows) -> {OUT_PATH}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
