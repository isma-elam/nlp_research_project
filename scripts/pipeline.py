"""Pipeline d'automatisation (reproductible) : dictionnaires -> corpus -> train -> predict.

Objectif : éviter de relancer manuellement chaque étape.

Exemples :
  python scripts/pipeline.py --all
  python scripts/pipeline.py --all --text "今日はいい天気です。"
    python scripts/pipeline.py --dicts --corpus --train
    python scripts/pipeline.py --train --validate
  python scripts/pipeline.py --predict --text "彼が来ると言っていたにもかかわらず…"

Vérification (validation + rapports):
    python scripts/pipeline.py --verify
    python scripts/pipeline.py --verify --stress --stress-terms "無料" --stress-out data/raw/stress_tests.csv

Notes :
- Utilise l'interpréteur courant (sys.executable).
- Sorties attendues :
  - data/json/joyo_meta.json (si --dicts)
    - data/json/jlpt_vocab.json + data/json/jlpt_grammar.json (si --dicts)
  - data/input/corpus.csv (si --corpus)
  - data/raw/baseline_model.joblib + data/raw/baseline_metrics.json (si --train)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(cmd: Sequence[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _script(path: Path) -> str:
    # On garde explicite pour Windows/PowerShell et les chemins avec espaces.
    return str(path)


def _maybe_require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Manquant ({what}) : {path}")


def _default_steps(args: argparse.Namespace) -> List[str]:
    # Si aucune étape n'est demandée, on fait comme --all.
    any_flag = any(
        [
            args.all,
            args.dicts,
            args.corpus,
            args.train,
            args.validate,
            args.verify,
            args.predict,
        ]
    )
    if args.all or not any_flag:
        # Si --text est fourni: dicts->corpus->train->validate->predict
        # Sinon: dicts->corpus->train->validate
        steps = ["dicts", "corpus", "train", "validate", "predict" if args.text else ""]
        return [s for s in steps if s]
    steps: List[str] = []
    if args.dicts:
        steps.append("dicts")
    if args.corpus:
        steps.append("corpus")
    if args.train:
        steps.append("train")
    if args.validate:
        steps.append("validate")
    if args.verify:
        # verify implique validate
        if "validate" not in steps:
            steps.append("validate")
    if args.predict:
        steps.append("predict")
    return [s for s in steps if s]


def _build_validate_cmd(args: argparse.Namespace, validate_path: Path) -> List[str]:
    cmd: List[str] = [PYTHON, _script(validate_path)]

    if args.verify and not args.validate_details_out:
        # Chemin de sortie par défaut pour --verify
        cmd += ["--details-out", str(ROOT / "data" / "raw" / "validation_details.csv")]
    elif args.validate_details_out:
        cmd += ["--details-out", args.validate_details_out]

    # Note : --verify exporte un CSV par défaut, sans spammer stdout.
    # Pour imprimer une ligne par phrase, exécute validate.py avec --details.

    if args.stress:
        cmd.append("--stress")
    if args.stress_terms:
        cmd += ["--stress-terms", args.stress_terms]
    if args.stress_out:
        cmd += ["--stress-out", args.stress_out]

    return cmd


def _build_predict_cmd(args: argparse.Namespace, predict_path: Path) -> List[str]:
    if not args.text:
        # mode interactif
        return [PYTHON, _script(predict_path)]

    cmd = [PYTHON, _script(predict_path), "--text", args.text]
    if args.json:
        cmd.append("--json")
    return cmd


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lancer le pipeline du projet de bout en bout")
    parser.add_argument("--all", action="store_true", help="Lancer dicts+corpus+train (+predict si --text)")
    parser.add_argument(
        "--dicts",
        action="store_true",
        help="Reconstruire les dictionnaires (kanji + vocab + grammaire) dans data/json/",
    )
    parser.add_argument("--corpus", action="store_true", help="Exécuter scripts/corpus.py")
    parser.add_argument("--train", action="store_true", help="Exécuter scripts/train_model.py")
    parser.add_argument("--validate", action="store_true", help="Exécuter scripts/validate.py (validation holdout)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validation + export des CSV de rapport (détails + stress-tests optionnels)",
    )
    parser.add_argument("--predict", action="store_true", help="Exécuter scripts/predict.py")
    parser.add_argument("--text", type=str, default=None, help="Si fourni, prédire sur ce texte")
    parser.add_argument("--json", action="store_true", help="En mode prédiction, sortir du JSON")

    # Options validate/verify (passées à scripts/validate.py)
    parser.add_argument(
        "--validate-details-out",
        type=str,
        default=None,
        help="Écrire les détails (par phrase) en CSV (passé à validate.py --details-out)",
    )
    parser.add_argument("--stress", action="store_true", help="Exécuter les stress-tests de validate.py")
    parser.add_argument(
        "--stress-terms",
        type=str,
        default=None,
        help="Termes de stress-test séparés par des virgules (passé à validate.py --stress-terms)",
    )
    parser.add_argument(
        "--stress-out",
        type=str,
        default=None,
        help="Écrire les résultats de stress-tests en CSV (passé à validate.py --stress-out)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    steps = _default_steps(args)

    scripts_dir = ROOT / "scripts"
    paths = {
        "dicts_kanji": scripts_dir / "nettoyage_kanji.py",
        "dicts_vocab": scripts_dir / "nettoyage_vocab.py",
        "dicts_grammar": scripts_dir / "nettoyage_grammaire.py",
        "corpus": scripts_dir / "corpus.py",
        "train": scripts_dir / "train_model.py",
        "validate": scripts_dir / "validate.py",
        "predict": scripts_dir / "predict.py",
    }

    def _run_dicts() -> None:
        # Explicite : un seul flag reconstruit tout ce qui est requis par l'extraction.
        for p in (paths["dicts_kanji"], paths["dicts_vocab"], paths["dicts_grammar"]):
            _maybe_require(p, "script de construction de dictionnaire")
            _run([PYTHON, _script(p)])

    handlers: dict[str, Callable[[], None]] = {
        "dicts": _run_dicts,
        "corpus": lambda: _run([PYTHON, _script(paths["corpus"])]),
        "train": lambda: _run([PYTHON, _script(paths["train"])]),
        "validate": lambda: _run(_build_validate_cmd(args, paths["validate"])),
        "predict": lambda: _run(_build_predict_cmd(args, paths["predict"])),
    }

    for step in steps:
        handler = handlers.get(step)
        if handler is None:
            raise ValueError(f"Étape inconnue : {step}")

        # On ne vérifie l'existence d'un script direct que pour les étapes 1:1.
        # L'étape 'dicts' est composite et fait ses propres vérifications.
        p = paths.get(step)
        if p is not None:
            _maybe_require(p, f"script pour l'étape '{step}'")
        handler()

    print("\nPipeline terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
