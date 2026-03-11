"""Construit un mapping kanji -> niveau JLPT à partir de joyo.csv."""

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
JOY0_CSV = ROOT / "data" / "dictionaries" / "joyo.csv"
JSON_DIR = ROOT / "data" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = JSON_DIR / f"{JOY0_CSV.stem}.json"
JSON_META_OUT = JSON_DIR / "joyo_meta.json"
PY_OUT = ROOT / "data" / "dictionaries" / "jlpt_data.py"


def load_kanji_levels(csv_path: Path) -> Dict[str, int]:
    """Charge joyo.csv et extrait un dict {kanji: jlpt} (N1..N5)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing joyo CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    if "kanji" not in df.columns or "jlpt" not in df.columns:
        raise ValueError("joyo.csv must contain columns 'kanji' and 'jlpt'")

    df = df[["kanji", "jlpt"]].dropna()
    df["kanji"] = df["kanji"].astype(str).str.strip()
    df["jlpt"] = df["jlpt"].astype(int)
    df = df[df["kanji"] != ""]

    # Supprimer les doublons en gardant la première occurrence
    df = df.drop_duplicates(subset=["kanji"], keep="first")

    levels = dict(zip(df["kanji"], df["jlpt"].astype(int)))
    return levels


def load_kanji_meta(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """Charge joyo.csv et extrait un dict kanji -> meta (jlpt, strokes, frequency).

    Champs retenus:
    - jlpt: niveau (1..5)
    - strokes: nombre de traits
    - frequency: rang/score de fréquence (source du CSV)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing joyo CSV at {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"kanji", "jlpt", "strokes", "frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"joyo.csv missing columns: {sorted(missing)}")

    df = df[["kanji", "jlpt", "strokes", "frequency"]].dropna()
    df["kanji"] = df["kanji"].astype(str).str.strip()
    df = df[df["kanji"] != ""]

    df["jlpt"] = df["jlpt"].astype(int)
    df["strokes"] = df["strokes"].astype(int)
    df["frequency"] = df["frequency"].astype(int)

    df = df.drop_duplicates(subset=["kanji"], keep="first")

    meta: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        k = row["kanji"]
        meta[k] = {
            "jlpt": int(row["jlpt"]),
            "strokes": int(row["strokes"]),
            "frequency": int(row["frequency"]),
        }
    return meta


def save_json(levels: Dict[str, int], path: Path) -> None:
    """Écrit le mapping kanji -> niveau JLPT en JSON."""
    path.write_text(json.dumps(levels, ensure_ascii=False, indent=2), encoding="utf-8")


def save_py(levels: Dict[str, int], path: Path) -> None:
    """Optionnel: génère un module Python 'jlpt_data.py' (legacy)."""
    # Generates a lightweight Python dict module for legacy compatibility
    lines = ["jlpt_data = {"]
    for k, v in levels.items():
        lines.append(f"    '{k}': {v},")
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(emit_py: bool) -> None:
    """Point d'entrée: crée kanji_levels.json (et jlpt_data.py si demandé)."""
    levels = load_kanji_levels(JOY0_CSV)
    meta = load_kanji_meta(JOY0_CSV)
    save_json(levels, JSON_OUT)
    JSON_META_OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if emit_py:
        save_py(levels, PY_OUT)

    counts = pd.Series(levels.values()).value_counts().sort_index()
    print(f"Saved {len(levels)} kanji levels -> {JSON_OUT}")
    print(f"Saved {len(meta)} kanji meta -> {JSON_META_OUT}")
    if emit_py:
        print(f"Saved Python dict -> {PY_OUT}")
    for lvl, cnt in counts.items():
        print(f"JLPT N{lvl}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build kanji->JLPT level mapping from joyo.csv")
    parser.add_argument("--emit-py", action="store_true", help="Also write jlpt_data.py for legacy code")
    args = parser.parse_args()
    main(emit_py=args.emit_py)
