"""Construit un corpus à partir de phrases manuelles (CSV).

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
            f"Dataset manuel introuvable : {MANUAL_PATH}. Crée-le avec les colonnes text,label."
        )

    print(f"Chargement du dataset manuel {MANUAL_PATH}...")
    df = pd.read_csv(MANUAL_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("manual_phrases.csv doit contenir les colonnes : text, label")

    df = df[["text", "label"]].copy()
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    removed = before - len(df)
    if removed:
        print(f"Suppression de {removed} doublons (text)")

    df.to_csv(OUT_PATH, index=False)
    print(f"Corpus sauvegardé ({len(df)} lignes) -> {OUT_PATH}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
