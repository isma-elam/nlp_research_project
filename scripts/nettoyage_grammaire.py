"""Génère un lookup JSON à partir de la grammaire JLPT déjà nettoyée.

Entrée : data/dictionaries/jlpt_grammar.csv
Sortie : data/json/jlpt_grammar_normalized.json
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "dictionaries" / "jlpt_grammar.csv"
JSON_DIR = ROOT / "data" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)
LOOKUP_JSON = JSON_DIR / f"{RAW_PATH.stem}.json"

LEVELS = ["N5", "N4", "N3", "N2", "N1"]

# Les patterns de grammaire sont matchés comme sous-chaînes littérales plus tard.
# Certaines sources encodent des alternatives dans une seule cellule (ex: "だ・です", "すら / ですら").
# On déplie ici pour que le lookup soit exploitable par l'extraction de features.
MIN_GRAMMAR_LEN = 2


def _expand_grammar_variants(pattern: str) -> list[str]:
    p = str(pattern).strip()
    if not p:
        return []
    parts = re.split(r"\s*[/／・]\s*", p)
    out: list[str] = []
    for part in parts:
        part = str(part).strip()
        if not part:
            continue
        if len(part) < MIN_GRAMMAR_LEN:
            continue
        out.append(part)
    return out


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Fichier d'entrée manquant : {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    required = {"Original", "English", "JLPT Level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"jlpt_grammar.csv : colonnes manquantes : {sorted(missing)}")

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

    # Déplier des alternatives du type 'だ・です' -> ['です'] (en supprimant les variantes trop courtes)
    df["pattern"] = df["pattern"].apply(_expand_grammar_variants)
    df = df.explode("pattern")
    df = df.dropna(subset=["pattern"]).copy()
    df["pattern"] = df["pattern"].astype(str).str.strip()
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
