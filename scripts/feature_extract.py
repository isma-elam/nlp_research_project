"""Extraction de features explicables pour le japonais (MVP).

Entrées :
- data/json/jlpt_vocab.json
- data/json/joyo.json
- data/json/jlpt_grammar.json

Sortie :
- dictionnaire de features numériques pour un texte.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "data" / "json"
VOCAB_JSON = JSON_DIR / "jlpt_vocab.json"
KANJI_JSON = JSON_DIR / "joyo.json"
KANJI_META_JSON = JSON_DIR / "joyo_meta.json"
GRAMMAR_JSON = JSON_DIR / "jlpt_grammar.json"

LEVELS = ["N5", "N4", "N3", "N2", "N1"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_punct(ch: str) -> bool:
    """Return True if ch is Unicode punctuation.

    We exclude punctuation from script ratios (kanji/hira/kata/latin/digits).
    """

    # Unicode categories starting with 'P' are punctuation.
    return unicodedata.category(ch).startswith("P")


def _char_stats(text: str) -> Dict[str, float]:
    # IMPORTANT: ratios are computed on characters excluding punctuation and spaces.
    counted_chars = [ch for ch in text if (not ch.isspace()) and (not _is_punct(ch))]
    total = max(len(counted_chars), 1)

    kanji = sum(1 for ch in counted_chars if "\u4e00" <= ch <= "\u9fff")
    hira = sum(1 for ch in counted_chars if "\u3040" <= ch <= "\u309f")
    kata = sum(1 for ch in counted_chars if "\u30a0" <= ch <= "\u30ff")
    latin = sum(1 for ch in counted_chars if "a" <= ch.lower() <= "z")
    digits = sum(1 for ch in counted_chars if ch.isdigit())

    nonspace_total = max(sum(1 for ch in text if not ch.isspace()), 1)
    punct_count = sum(1 for ch in text if _is_punct(ch))
    comma_count = sum(1 for ch in text if ch in {"、", ","})

    return {
        "len_chars": float(len(text)),
        "ratio_kanji": kanji / total,
        "ratio_hira": hira / total,
        "ratio_kata": kata / total,
        "ratio_latin": latin / total,
        "ratio_digits": digits / total,
        "punct_count": float(punct_count),
        "comma_count": float(comma_count),
        "ratio_punct": punct_count / nonspace_total,
    }


def _sentence_stats(text: str) -> Dict[str, float]:
    # Séparation simple par ponctuation japonaise/latine
    sentences = [s for s in re.split(r"[。！？!?]", text) if s.strip()]
    count = len(sentences)
    avg_len = sum(len(s) for s in sentences) / max(count, 1)
    return {
        "sent_count": float(count),
        "sent_avg_len": float(avg_len),
    }


def _prepare_vocab(vocab_json: dict) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for level in LEVELS:
        for w in vocab_json.get(level, []):
            if w:
                pairs.append((w, level))
    return pairs


def _vocab_features(text: str, vocab_pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    counts = dict.fromkeys(LEVELS, 0)
    matched_total = 0
    # Matching simple (présence) pour MVP
    for word, level in vocab_pairs:
        if word in text:
            counts[level] += 1
            matched_total += 1

    features = {f"vocab_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["vocab_total"] = float(matched_total)
    return features


def _kanji_features(text: str, kanji_map: Dict[str, int]) -> Dict[str, float]:
    counts = dict.fromkeys(LEVELS, 0)
    total_kanji = 0
    for ch in text:
        if ch in kanji_map:
            total_kanji += 1
            lvl_num = kanji_map[ch]
            lvl = f"N{lvl_num}"
            if lvl in counts:
                counts[lvl] += 1

    features = {f"kanji_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["kanji_total"] = float(total_kanji)
    return features


def _collect_kanji_meta_lists(
    text: str, kanji_meta: Dict[str, Dict[str, int]]
) -> Tuple[List[int], List[int], List[int]]:
    strokes: List[int] = []
    freqs: List[int] = []
    jlpt_levels: List[int] = []

    for ch in text:
        meta = kanji_meta.get(ch)
        if not meta:
            continue

        strokes_val = meta.get("strokes")
        if isinstance(strokes_val, int):
            strokes.append(strokes_val)

        freq_val = meta.get("frequency")
        if isinstance(freq_val, int):
            freqs.append(freq_val)

        jlpt_val = meta.get("jlpt")
        if isinstance(jlpt_val, int):
            jlpt_levels.append(jlpt_val)

    return strokes, freqs, jlpt_levels


def _kanji_complexity_features(text: str, kanji_meta: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    strokes, freqs, jlpt_levels = _collect_kanji_meta_lists(text, kanji_meta)

    if not strokes and not freqs and not jlpt_levels:
        return {
            "kanji_strokes_avg": 0.0,
            "kanji_strokes_max": 0.0,
            "kanji_freq_avg": 0.0,
            "kanji_freq_min": 0.0,
            "kanji_jlpt_avg_num": 0.0,
        }

    def _avg(xs: List[int]) -> float:
        return float(sum(xs) / max(len(xs), 1))

    return {
        "kanji_strokes_avg": _avg(strokes) if strokes else 0.0,
        "kanji_strokes_max": float(max(strokes)) if strokes else 0.0,
        "kanji_freq_avg": _avg(freqs) if freqs else 0.0,
        "kanji_freq_min": float(min(freqs)) if freqs else 0.0,
        "kanji_jlpt_avg_num": _avg(jlpt_levels) if jlpt_levels else 0.0,
    }


def _grammar_features(text: str, grammar_json: dict) -> Dict[str, float]:
    counts = dict.fromkeys(LEVELS, 0)
    for lvl in LEVELS:
        patterns = grammar_json.get(lvl, [])
        for p in patterns:
            if p and p in text:
                counts[lvl] += 1

    features = {f"grammar_{lvl}": float(counts[lvl]) for lvl in LEVELS}
    features["grammar_total"] = float(sum(counts.values()))
    return features


def extract_features(text: str) -> Dict[str, float]:
    vocab_json = _load_json(VOCAB_JSON)
    kanji_json = _load_json(KANJI_JSON)
    kanji_meta = _load_json(KANJI_META_JSON) if KANJI_META_JSON.exists() else {}
    grammar_json = _load_json(GRAMMAR_JSON)

    features: Dict[str, float] = {}
    features.update(_char_stats(text))
    features.update(_sentence_stats(text))
    features.update(_vocab_features(text, _prepare_vocab(vocab_json)))
    features.update(_kanji_features(text, kanji_json))
    features.update(_kanji_complexity_features(text, kanji_meta))
    features.update(_grammar_features(text, grammar_json))
    return features


if __name__ == "__main__":
    # Test rapide
    sample = "今日はいい天気です。"
    feats = extract_features(sample)
    for k in sorted(feats.keys()):
        print(f"{k}: {feats[k]}")
