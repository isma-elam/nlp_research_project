"""Génère un lookup JSON à partir du vocabulaire JLPT déjà nettoyé."""

import json
from pathlib import Path

import pandas as pd

# Chemins d'entrée/sortie
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "dictionaries" / "jlpt_vocab.csv"
JSON_DIR = ROOT / "data" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)
LOOKUP_JSON = JSON_DIR / f"{RAW_PATH.stem}.json"

LEVELS = ["N5", "N4", "N3", "N2", "N1"]


def normalize_levels(series: pd.Series) -> pd.Series:
    """Uniformise les libellés de niveaux (JLPT -> N1..N5)."""
    cleaned = series.astype(str).str.upper().str.replace("JLPT", "", regex=False).str.strip()
    return cleaned


def main() -> None:
    """Pipeline: lecture CSV nettoyé -> export JSON."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    # Normaliser les colonnes attendues
    df = df.rename(columns={
        "Original": "word",
        "Furigana": "furigana",
        "English": "english",
        "JLPT Level": "level",
    })

    df["word"] = df["word"].astype(str).str.strip()
    df["furigana"] = df["furigana"].astype(str).str.strip()
    df["english"] = df["english"].astype(str).str.strip()
    df["level"] = normalize_levels(df["level"])

    # Garder uniquement les niveaux attendus et supprimer les lignes vides
    df = df[df["level"].isin(LEVELS)].copy()
    df = df[df["word"] != ""]

    # Supprimer les doublons exacts
    df = df.drop_duplicates(subset=["word", "level"])

    # Construire un dictionnaire {niveau: [mots]}
    lookup = {lvl: sorted(df[df["level"] == lvl]["word"].unique()) for lvl in LEVELS}
    LOOKUP_JSON.write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved lookup JSON: {LOOKUP_JSON}")
    for lvl in LEVELS:
        print(f"{lvl}: {len(lookup[lvl])} words")


if __name__ == "__main__":
    main()
