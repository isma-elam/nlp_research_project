"""Prédit la difficulté d'une phrase japonaise avec le modèle baseline.

Usage:
  python scripts/predict.py --text "今日はいい天気です。"
  python scripts/predict.py            # mode interactif

Sortie:
    - label prédit (ex: easy/medium/hard) avec couleur optionnelle
  - probas si disponibles
    - score facile (0–100) + pourcentage de difficulté
    - aperçu de quelques features explicables

Notes:
  - Le script cherche le modèle dans `data/raw/` (compat).
  - Si tu modifies `scripts/feature_extract.py` (ex: ponctuation), il faut ré-entraîner.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Permet d'importer feature_extract depuis scripts/
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from feature_extract import extract_features


ROOT = Path(__file__).resolve().parent.parent


def _norm_text_strict(text: str) -> str:
    # Normalisation stable pour comparer des phrases (évite les surprises fullwidth, etc.)
    return unicodedata.normalize("NFKC", str(text)).strip()


def _is_punct(ch: str) -> bool:
    # Unicode category 'P*' => punctuation
    return bool(ch) and unicodedata.category(ch).startswith("P")


def _norm_text_loose(text: str) -> str:
    # Comparaison plus permissive : ignore espaces + ponctuation
    s = _norm_text_strict(text)
    return "".join(ch for ch in s if (not ch.isspace()) and (not _is_punct(ch)))


def _candidate_paths() -> Tuple[Path, ...]:
    return (
        ROOT / "data" / "raw" / "baseline_model.joblib",
    )


def _corpus_path() -> Path:
    return ROOT / "data" / "input" / "corpus.csv"


def _warn_if_in_corpus(text: str) -> None:
    corpus_path = _corpus_path()
    if not corpus_path.exists():
        return

    try:
        df = pd.read_csv(corpus_path)
    except Exception:
        return

    if "text" not in df.columns:
        return

    needle_strict = _norm_text_strict(text)
    needle_loose = _norm_text_loose(text)

    hay_strict = {_norm_text_strict(t) for t in df["text"].dropna().astype(str).tolist()}
    if needle_strict in hay_strict:
        print(
            "WARNING: cette phrase est DÉJÀ dans data/input/corpus.csv (donc vue au train/test). "
            "Une 'validation' sur cette phrase n'est pas honnête.",
            file=sys.stderr,
        )
        return

    # Loose match (ignore ponctuation + espaces) pour attraper des variantes triviales.
    hay_loose = {_norm_text_loose(t) for t in df["text"].dropna().astype(str).tolist()}
    if needle_loose and needle_loose in hay_loose:
        print(
            "WARNING: phrase très proche d'une phrase du corpus (match sans ponctuation/espaces). "
            "Risque de 'triche' si tu valides dessus.",
            file=sys.stderr,
        )


def _find_model_path(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    for p in _candidate_paths():
        if p.exists():
            return p

    raise FileNotFoundError(
        "No baseline model found. Expected one of: "
        + ", ".join(str(p) for p in _candidate_paths())
    )


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    # VS Code + Windows Terminal supporte généralement ANSI.
    return sys.stdout.isatty()


def _colorize(text: str, color_code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{color_code}m{text}\x1b[0m"


def _build_feature_row(text: str) -> pd.DataFrame:
    feats = extract_features(text)
    return pd.DataFrame([feats]).fillna(0.0)


def _safe_predict_proba(model: Any, X: pd.DataFrame) -> Optional[Dict[str, float]]:
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return None
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_"))
    return {str(cls): float(p) for cls, p in zip(classes, proba[0])}


def _score_easy_0_100(proba: Optional[Dict[str, float]], predicted_label: str) -> Optional[float]:
    """Return an easy-score in [0,100]. 100 = easy, 0 = hard.

    - 3 classes (easy/medium/hard): score = 100 * (P(easy) + 0.5 * P(medium))
    """
    if proba is None:
        return None

    peasy = proba.get("easy")
    pmed = proba.get("medium")
    phard = proba.get("hard")

    # 3 classes (preferred)
    if peasy is not None:
        return 100.0 * (float(peasy) + 0.5 * float(pmed or 0.0))

    # 2 classes fallback
    if phard is not None:
        return 100.0 * (1.0 - float(phard))

    # Last resort: use probability of predicted label if present
    if predicted_label in proba:
        return 100.0 * float(proba[predicted_label])

    return None


def _difficulty_0_100(score_easy: Optional[float]) -> Optional[float]:
    if score_easy is None:
        return None
    return 100.0 - float(score_easy)


def _band_from_score(score_easy: Optional[float]) -> Optional[str]:
    """Simple interpretation bands.

    Examples from user:
      23% -> relatively difficult
      53% -> medium
    """
    if score_easy is None:
        return None
    if score_easy < 33.0:
        return "difficile"
    if score_easy < 66.0:
        return "moyen"
    return "facile"


def _feature_preview(feats: Dict[str, float]) -> Dict[str, float]:
    # Aperçu court et lisible (tu peux étendre ensuite)
    keys = [
        "len_chars",
        "ratio_kanji",
        "ratio_hira",
        "ratio_kata",
        "punct_count",
        "comma_count",
        "ratio_punct",
        "sent_count",
        "sent_avg_len",
        "vocab_total",
        "kanji_total",
        "grammar_total",
    ]
    return {k: float(feats[k]) for k in keys if k in feats}


def _build_json_output(
    *,
    text: str,
    label: str,
    proba: Optional[Dict[str, float]],
    score_easy: Optional[float],
    difficulty: Optional[float],
    band: Optional[str],
    preview: Dict[str, float],
    model_path: Path,
) -> Dict[str, Any]:
    return {
        "text": text,
        "label": label,
        "proba": proba,
        "score_easy_0_100": None if score_easy is None else float(score_easy),
        "difficulty_0_100": None if difficulty is None else float(difficulty),
        "band": band,
        "features_preview": preview,
        "model_path": str(model_path),
    }


def _print_human_output(
    *,
    model_path: Path,
    label_str: str,
    proba: Optional[Dict[str, float]],
    score_easy: Optional[float],
    difficulty: Optional[float],
    band: Optional[str],
    preview: Dict[str, float],
    use_color: bool,
) -> None:
    if label_str.lower() in {"easy", "facile", "0"}:
        colored = _colorize(label_str, "32", use_color)
    elif label_str.lower() in {"medium", "moyen"}:
        colored = _colorize(label_str, "33", use_color)
    elif label_str.lower() in {"hard", "difficile", "1"}:
        colored = _colorize(label_str, "31", use_color)
    else:
        colored = _colorize(label_str, "33", use_color)

    print(f"Model: {model_path}")
    print(f"Prediction: {colored}")

    if score_easy is not None:
        score_txt = (
            f"{score_easy:.0f}/100"
            if abs(score_easy - round(score_easy)) < 1e-9
            else f"{score_easy:.1f}/100"
        )
        if band is not None:
            print(f"Score (facile): {score_txt}  ->  {band}")
        else:
            print(f"Score (facile): {score_txt}")

    if difficulty is not None:
        diff_txt = (
            f"{difficulty:.0f}/100"
            if abs(difficulty - round(difficulty)) < 1e-9
            else f"{difficulty:.1f}/100"
        )
        print(f"Difficulté: {diff_txt}")

    if proba is not None:
        proba_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(proba.items()))
        print(f"Proba: {proba_str}")

    print("Features (aperçu):")
    for k, v in preview.items():
        print(f"  - {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict difficulty for a Japanese sentence")
    parser.add_argument("--text", type=str, default=None, help="Input Japanese text")
    parser.add_argument("--model", type=str, default=None, help="Path to .joblib model (optional)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument(
        "--no-leak-check",
        action="store_true",
        help="Disable warning when input text is found in data/input/corpus.csv",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON object (label/proba/features) for downstream use",
    )
    args = parser.parse_args()

    text = args.text

    if not text:
        text = input("Entre une phrase japonaise: ").strip()

    if not text:
        raise SystemExit("Empty input text")

    if not args.no_leak_check:
        _warn_if_in_corpus(text)

    model_path = _find_model_path(args.model)
    model = joblib.load(model_path)

    X = _build_feature_row(text)
    label = model.predict(X)[0]
    label_str = str(label)

    feats = extract_features(text)
    preview = _feature_preview(feats)
    proba = _safe_predict_proba(model, X)
    score_easy = _score_easy_0_100(proba, predicted_label=label_str)
    difficulty = _difficulty_0_100(score_easy)
    band = _band_from_score(score_easy)

    if args.json:
        out = _build_json_output(
            text=text,
            label=label_str,
            proba=proba,
            score_easy=score_easy,
            difficulty=difficulty,
            band=band,
            preview=preview,
            model_path=model_path,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    use_color = _supports_color() and (not args.no_color)

    _print_human_output(
        model_path=model_path,
        label_str=label_str,
        proba=proba,
        score_easy=score_easy,
        difficulty=difficulty,
        band=band,
        preview=preview,
        use_color=use_color,
    )


if __name__ == "__main__":
    main()
