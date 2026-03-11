"""Génère un lookup JSON à partir de la grammaire JLPT déjà nettoyée.

Entrée : data/dictionaries/jlpt_grammar.csv
Sortie : data/json/jlpt_grammar_normalized.json
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "dictionaries" / "jlpt_grammar.csv"
JSON_DIR = ROOT / "data" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)
LOOKUP_JSON = JSON_DIR / f"{RAW_PATH.stem}.json"

LEVELS = ["N5", "N4", "N3", "N2", "N1"]


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    required = {"Original", "English", "JLPT Level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"jlpt_grammar.csv missing columns: {sorted(missing)}")

    # Normalisation de base
    df = df.rename(columns={
        "Original": "pattern",
        "English": "english",
        "JLPT Level": "level",
    })

    df["pattern"] = df["pattern"].astype(str).str.strip()
    df["english"] = df["english"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.upper().str.strip()

    # Filtrer niveaux attendus + enlever vides
    df = df[df["level"].isin(LEVELS)].copy()
    df = df[df["pattern"] != ""]

    # Déduplication (pattern + level)
    df = df.drop_duplicates(subset=["pattern", "level"])

    lookup = {
        lvl: sorted(df[df["level"] == lvl]["pattern"].unique())
        for lvl in LEVELS
    }
    LOOKUP_JSON.write_text(json.dumps(lookup, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Lookup grammaire sauvegardé : {LOOKUP_JSON}")
    for lvl in LEVELS:
        print(f"{lvl}: {len(lookup[lvl])} patterns")


if __name__ == "__main__":
    main()
