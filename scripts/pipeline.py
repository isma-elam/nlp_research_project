"""Pipeline d'automatisation (MVP) : dictionnaires -> corpus -> train -> predict.

Objectif : éviter de relancer manuellement chaque étape.

Exemples :
  python scripts/pipeline.py --all
  python scripts/pipeline.py --all --text "今日はいい天気です。"
  python scripts/pipeline.py --dicts --corpus --train
    python scripts/pipeline.py --train --validate
  python scripts/pipeline.py --predict --text "彼が来ると言っていたにもかかわらず…"

Notes :
- Utilise l'interpréteur courant (sys.executable).
- Sorties attendues :
  - data/json/joyo_meta.json (si --dicts)
  - data/input/corpus.csv (si --corpus)
  - data/raw/baseline_model.joblib + data/raw/baseline_metrics.json (si --train)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(cmd: Sequence[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _script(path: Path) -> str:
    # Keep it explicit for Windows/PowerShell and spaces
    return str(path)


def _maybe_require(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: {path}")


def _default_steps(args: argparse.Namespace) -> List[str]:
    # If no specific step flags were provided, default to --all.
    any_flag = any(
        [
            args.all,
            args.dicts,
            args.corpus,
            args.train,
            args.validate,
            args.predict,
        ]
    )
    if args.all or not any_flag:
        # Si --text est fourni: dicts->corpus->train->validate->predict
        # Sinon: dicts->corpus->train->validate
        return ["dicts", "corpus", "train", "validate", "predict" if args.text else ""]
    steps: List[str] = []
    if args.dicts:
        steps.append("dicts")
    if args.corpus:
        steps.append("corpus")
    if args.train:
        steps.append("train")
    if args.validate:
        steps.append("validate")
    if args.predict:
        steps.append("predict")
    return [s for s in steps if s]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the project pipeline end-to-end")
    parser.add_argument("--all", action="store_true", help="Run dicts+corpus+train (+predict if --text)")
    parser.add_argument("--dicts", action="store_true", help="Run scripts/nettoyage_kanji.py")
    parser.add_argument("--corpus", action="store_true", help="Run scripts/corpus.py")
    parser.add_argument("--train", action="store_true", help="Run scripts/train_model.py")
    parser.add_argument("--validate", action="store_true", help="Run scripts/validate.py (holdout validation)")
    parser.add_argument("--predict", action="store_true", help="Run scripts/predict.py")
    parser.add_argument("--text", type=str, default=None, help="If provided, run prediction on this text")
    parser.add_argument("--json", action="store_true", help="When predicting, output JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    steps = _default_steps(args)

    scripts_dir = ROOT / "scripts"
    paths = {
        "dicts": scripts_dir / "nettoyage_kanji.py",
        "corpus": scripts_dir / "corpus.py",
        "train": scripts_dir / "train_model.py",
        "validate": scripts_dir / "validate.py",
        "predict": scripts_dir / "predict.py",
    }

    for step in steps:
        _maybe_require(paths[step], f"script for step '{step}'")

        if step == "dicts":
            _run([PYTHON, _script(paths[step])])
        elif step == "corpus":
            _run([PYTHON, _script(paths[step])])
        elif step == "train":
            _run([PYTHON, _script(paths[step])])
        elif step == "validate":
            _run([PYTHON, _script(paths[step])])
        elif step == "predict":
            if not args.text:
                # interactive predict mode
                cmd = [PYTHON, _script(paths[step])]
            else:
                cmd = [PYTHON, _script(paths[step]), "--text", args.text]
                if args.json:
                    cmd.append("--json")
            _run(cmd)

    print("\nPipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
