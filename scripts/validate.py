"""Validation honnête sur un jeu de données séparé (anti-triche).

But :
- Évaluer le modèle sur des phrases qui ne sont PAS dans le corpus d'entraînement.
- Détecter les chevauchements (fuite de données) et PRÉVENIR.
    - Par défaut : avertissement uniquement (ne bloque pas).
    - Optionnel : --fail-on-overlap pour rendre ça bloquant.

Utilisation :
    ./.venv/Scripts/python scripts/validate.py
    ./.venv/Scripts/python scripts/validate.py --data data/input/validation_phrases.csv
    ./.venv/Scripts/python scripts/validate.py --json

Rapport détaillé (par phrase):
    ./.venv/Scripts/python scripts/validate.py --details
    ./.venv/Scripts/python scripts/validate.py --details-out data/raw/validation_details.csv

Entrée (par défaut): data/input/validation_phrases.csv
Colonnes attendues: text,label (id optionnel)
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Permet d'importer feature_extract depuis scripts/
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from model_features import build_feature_df_for_model

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
            raise FileNotFoundError(f"Modèle introuvable : {p}")
        return p

    p = ROOT / "data" / "raw" / "baseline_model.joblib"
    if p.exists():
        return p

    raise FileNotFoundError(f"Aucun modèle baseline trouvé ici : {p}")


def _build_feature_df(texts: Iterable[str]) -> pd.DataFrame:
    return build_feature_df_for_model(texts)


def _safe_predict_proba_df(model: Any, X: Any) -> Optional[pd.DataFrame]:
    """Renvoie un DataFrame de probabilités aligné sur model.classes_ si disponible.

    Supporte les deux contrats :
    - ancien modèle: X = DataFrame de features numériques
    - nouveau modèle: X = liste/array de textes bruts
    """
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return None
    try:
        proba = model.predict_proba(X)
        classes = [str(c) for c in getattr(model, "classes_")]
        return pd.DataFrame(proba, columns=[f"p_{c}" for c in classes])
    except Exception:
        return None


def _load_texts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Le CSV de validation doit contenir les colonnes : text, label")
    df = df.dropna(subset=["text", "label"]).copy()
    if "id" in df.columns:
        df["id"] = df["id"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).map(_norm_text_strict)
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    return df


def _build_details_df(
    *,
    val_df: pd.DataFrame,
    y_pred: Sequence[str],
    proba_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    details = pd.DataFrame(
        {
            "id": val_df["id"].tolist() if "id" in val_df.columns else [f"V{i+1:04d}" for i in range(len(val_df))],
            "label_true": val_df["label"].astype(str).tolist(),
            "label_pred": [str(x) for x in y_pred],
            "text": val_df["text"].astype(str).tolist(),
        }
    )
    details["correct"] = (details["label_true"] == details["label_pred"]).astype(int)

    if proba_df is not None:
        for col in ["p_easy", "p_medium", "p_hard"]:
            if col not in proba_df.columns:
                proba_df[col] = 0.0
        proba_df = proba_df[["p_easy", "p_medium", "p_hard"]]
        details = pd.concat([details.reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
    else:
        details["p_easy"] = float("nan")
        details["p_medium"] = float("nan")
        details["p_hard"] = float("nan")

    return details[["id", "label_true", "label_pred", "correct", "p_easy", "p_medium", "p_hard", "text"]]


def _print_details_csv(details_df: pd.DataFrame, *, max_rows: int, text_max: int) -> None:
    df = details_df.copy()
    if text_max > 0:
        df["text"] = df["text"].astype(str).map(lambda s: s if len(s) <= text_max else (s[: text_max - 1] + "…"))
    for c in ["p_easy", "p_medium", "p_hard"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if max_rows and max_rows > 0:
        df = df.head(max_rows)
    # Laisse pandas gérer correctement les guillemets/retours à la ligne/virgules.
    df.to_csv(sys.stdout, index=False, encoding="utf-8", float_format="%.4f")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valider le modèle baseline sur un CSV holdout")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "input" / "validation_phrases.csv"),
        help="Chemin vers le CSV de validation (colonnes text,label)",
    )
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le modèle .joblib (optionnel)")
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        help="Quitter en erreur si des textes de validation chevauchent le corpus d'entraînement",
    )
    parser.add_argument("--json", action="store_true", help="Sortir les métriques au format JSON")
    parser.add_argument("--details", action="store_true", help="Afficher les prédictions (par phrase) en CSV")
    parser.add_argument(
        "--details-max",
        type=int,
        default=0,
        help="Nombre max de lignes avec --details (0 = toutes)",
    )
    parser.add_argument(
        "--details-text-max",
        type=int,
        default=120,
        help="Nombre max de caractères de texte avec --details (0 = pas de troncature)",
    )
    parser.add_argument(
        "--details-out",
        type=str,
        default=None,
        help="Écrire les détails par phrase en CSV (complet, non tronqué)",
    )
    return parser.parse_args()


def _resolve_existing_path(p: str, *, what: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{what} introuvable : {path}")
    return path


def _read_corpus_texts(corpus_path: Path) -> List[str]:
    corpus_df = pd.read_csv(corpus_path)
    if "text" not in corpus_df.columns:
        raise ValueError("corpus.csv doit contenir la colonne : text")
    return corpus_df["text"].dropna().astype(str).tolist()


def _warn_overlap(
    *,
    overlap_strict: List[str],
    overlap_loose: List[str],
    fail_on_overlap: bool,
) -> None:
    if not overlap_strict and not overlap_loose:
        return
    msg = [
        "AVERTISSEMENT: le fichier de validation contient des phrases déjà présentes dans le corpus.",
        "Risque de fuite de données (validation 'trichée').",
    ]
    if overlap_strict:
        msg.append(f"- Chevauchement STRICT (exact) : {len(overlap_strict)}")
        msg.extend([f"  * {t}" for t in overlap_strict[:5]])
    if overlap_loose:
        msg.append(f"- Chevauchement RELÂCHÉ (sans ponctuation/espaces) : {len(overlap_loose)}")
        msg.extend([f"  * {t}" for t in overlap_loose[:5]])
    msg.append("Conseil: enlève ces phrases de la validation OU du corpus.")
    print("\n".join(msg), file=sys.stderr)
    if fail_on_overlap:
        raise SystemExit(2)


def _maybe_write_details(details_df: pd.DataFrame, details_out: Optional[str]) -> Optional[Path]:
    if not details_out:
        return None
    out_path = Path(details_out)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def _print_summary(*, model_path: Path, val_path: Path, metrics: Dict[str, Any], details_out_path: Optional[Path]) -> None:
    print(f"Modèle: {model_path}")
    print(f"Fichier de validation: {val_path}")
    print(f"n_val={metrics['n_val']} | accuracy={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}")
    if details_out_path is not None:
        print(f"CSV détails: {details_out_path}")


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
    args = _parse_args()
    val_path = _resolve_existing_path(args.data, what="Fichier de validation")
    corpus_path = ROOT / "data" / "input" / "corpus.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Fichier corpus introuvable : {corpus_path}. Lance : ./.venv/Scripts/python scripts/corpus.py"
        )

    val_df = _load_texts(val_path)

    corpus_texts = _read_corpus_texts(corpus_path)
    val_texts = val_df["text"].dropna().astype(str).tolist()

    overlap_strict, overlap_loose = _overlap_report(corpus_texts, val_texts)
    _warn_overlap(overlap_strict=overlap_strict, overlap_loose=overlap_loose, fail_on_overlap=bool(args.fail_on_overlap))

    model_path = _find_model_path(args.model)
    model = joblib.load(model_path)

    val_texts = val_df["text"].astype(str).tolist()
    y_true = val_df["label"].astype(str).tolist()

    # Nouveau modèle: predict directement sur les textes.
    # Compatibilité: si ça échoue, on retombe sur les features numériques.
    try:
        y_pred_raw = model.predict(val_texts)
        y_pred = [str(x) for x in y_pred_raw]
        proba_df = _safe_predict_proba_df(model, val_texts)
    except Exception:
        x_val = _build_feature_df(val_texts)
        y_pred_raw = model.predict(x_val)
        y_pred = [str(x) for x in y_pred_raw]
        proba_df = _safe_predict_proba_df(model, x_val)

    details_df = _build_details_df(val_df=val_df, y_pred=y_pred, proba_df=proba_df)

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
        # On autorise quand même l'export des détails en mode JSON.

    details_out_path = _maybe_write_details(details_df, args.details_out)

    if args.details:
        max_rows = int(args.details_max) if int(args.details_max) != 0 else 0
        _print_details_csv(details_df, max_rows=max_rows, text_max=int(args.details_text_max))

    if not args.json:
        _print_summary(model_path=model_path, val_path=val_path, metrics=metrics, details_out_path=details_out_path)


if __name__ == "__main__":
    main()
