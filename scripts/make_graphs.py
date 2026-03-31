"""Générer des graphes (PNG) à partir des résultats de validation.

Entrée par défaut:
- data/raw/validation_details.csv (produit par scripts/validate.py --details-out ...)

Sortie par défaut:
- data/raw/figures/*.png

Utilisation:
    ./.venv/Scripts/python scripts/make_graphs.py
    ./.venv/Scripts/python scripts/make_graphs.py --show

Graphes générés (4) 
1) Matrice de confusion (comptes)
2) Précision/Rappel/F1 par classe
3) Distribution du score "easy" 0–100 par label vrai (boîte à moustaches)
4) Feature explicable : ratio_kanji vs score 0–100 (coloré par label vrai)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Permet d'importer feature_extract depuis scripts/
sys.path.append(str(Path(__file__).resolve().parent))

from feature_extract import extract_features


ROOT = Path(__file__).resolve().parent.parent
LABELS: List[str] = ["easy", "medium", "hard"]


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Générer 4 graphes (PNG) depuis validation_details.csv")
    p.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "data" / "raw" / "validation_details.csv"),
        help="Chemin vers validation_details.csv",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(ROOT / "data" / "raw" / "figures"),
        help="Dossier de sortie pour les PNG",
    )
    p.add_argument("--show", action="store_true", help="Afficher les figures en plus de les enregistrer")
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_existing_path(path_str: str, *, what: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"{what} introuvable : {p}")
    return p


def _resolve_outdir(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_validation_details(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"label_true", "label_pred", "correct", "p_easy", "p_medium", "p_hard", "text"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "validation_details.csv incomplet. Colonnes manquantes: "
            + ", ".join(missing)
            + "\nConseil: regénère-le via scripts/validate.py --details-out data/raw/validation_details.csv"
        )

    df = df.dropna(subset=["label_true", "label_pred", "correct", "text"]).copy()
    df["label_true"] = df["label_true"].astype(str).str.strip()
    df["label_pred"] = df["label_pred"].astype(str).str.strip()
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

    for c in ["p_easy", "p_medium", "p_hard"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["p_easy", "p_medium", "p_hard"]].isna().any().any():
        raise ValueError(
            "Les colonnes p_easy/p_medium/p_hard contiennent des NaN. "
            "Assure-toi que le modèle supporte predict_proba et que validate.py exporte les probabilités."
        )

    df = df[df["label_true"].isin(LABELS) & df["label_pred"].isin(LABELS)].copy()
    return df


def _easy_score_0_100(p_easy: np.ndarray, p_medium: np.ndarray) -> np.ndarray:
    return 100.0 * (p_easy + 0.5 * p_medium)


def _save(fig: plt.Figure, path: Path, *, show: bool) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        fig.show()
    plt.close(fig)


def _plot_confusion_matrix(df: pd.DataFrame) -> plt.Figure:
    y_true = df["label_true"].astype(str).tolist()
    y_pred = df["label_pred"].astype(str).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion (comptes)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")

    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _plot_per_class_metrics(df: pd.DataFrame) -> plt.Figure:
    y_true = df["label_true"].astype(str).tolist()
    y_pred = df["label_pred"].astype(str).tolist()

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )

    x = np.arange(len(LABELS))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, f1, width, label="f1")

    ax.set_xticks(x, LABELS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Métriques par classe (validation holdout)")
    ax.legend(loc="lower right")

    for i, s in enumerate(support):
        ax.text(i, 1.02, f"n={int(s)}", ha="center", va="bottom")

    return fig


def _plot_easy_score_distribution(df: pd.DataFrame) -> plt.Figure:
    score = _easy_score_0_100(df["p_easy"].to_numpy(), df["p_medium"].to_numpy())

    data: List[np.ndarray] = []
    y_true = df["label_true"].to_numpy()
    for lbl in LABELS:
        data.append(score[y_true == lbl])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot(data, tick_labels=LABELS, showfliers=True)
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Score easy (0–100)")
    ax.set_title('Distribution du score "easy" 0–100 par label vrai')
    return fig


def _plot_ratio_kanji_vs_score(df: pd.DataFrame) -> plt.Figure:
    score = _easy_score_0_100(df["p_easy"].to_numpy(), df["p_medium"].to_numpy())

    ratio_kanji: List[float] = []
    for text in df["text"].astype(str).tolist():
        feats: Dict[str, float] = extract_features(text)
        ratio_kanji.append(float(feats.get("ratio_kanji", 0.0)))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y_true = df["label_true"].astype(str).to_numpy()

    for lbl in LABELS:
        mask = y_true == lbl
        ax.scatter(
            np.asarray(ratio_kanji)[mask],
            score[mask],
            label=lbl,
            alpha=0.85,
            s=36,
        )

    ax.set_xlabel("ratio_kanji (feature explicable)")
    ax.set_ylabel("Score easy (0–100)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("ratio_kanji vs score (coloré par label vrai)")
    ax.legend(loc="upper right")
    return fig


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = _resolve_existing_path(args.input, what="CSV validation_details")
    outdir = _resolve_outdir(args.outdir)

    df = _load_validation_details(input_path)
    if len(df) == 0:
        raise ValueError("validation_details.csv ne contient aucune ligne exploitable.")

    figures: List[Tuple[str, plt.Figure]] = [
        ("01_matrice_de_confusion.png", _plot_confusion_matrix(df)),
        ("02_metrique_par_classe.png", _plot_per_class_metrics(df)),
        ("03_easy_score_0_100_vrai_label.png", _plot_easy_score_distribution(df)),
        ("04_ratio_kanji_vs_easy_score.png", _plot_ratio_kanji_vs_score(df)),
    ]

    for filename, fig in figures:
        _save(fig, outdir / filename, show=bool(args.show))

    print(f"OK: {len(figures)} figures écrites dans: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
