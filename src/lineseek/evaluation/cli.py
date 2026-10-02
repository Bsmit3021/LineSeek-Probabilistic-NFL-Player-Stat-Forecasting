"""`lineseek-baselines` CLI: run the frozen Phase 4A baseline evaluation, or validate a run."""

import argparse
import logging
import sys
from pathlib import Path

from lineseek.evaluation.phase4a import (
    EVAL_SEASONS,
    POINT_IN_TIME_WARNING,
    UNIVERSE_WARNING,
    WARMUP_SEASONS,
    evaluate,
    evaluation_root,
    load_inputs,
    summarize,
    validate_run,
    write_run,
)
from lineseek.features.player_game import features_root
from lineseek.grade.labels import labels_root
from lineseek.ingest.snapshot import data_root

log = logging.getLogger("lineseek.baselines")


def _print_summary(summary: dict) -> None:
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  WARNING: {POINT_IN_TIME_WARNING}")
    print(f"  WARNING: {UNIVERSE_WARNING}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-baselines")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help=f"predict, fold and score: warm-up {WARMUP_SEASONS[0]}-"
                                 f"{WARMUP_SEASONS[-1]}, evaluation {EVAL_SEASONS[0]}-"
                                 f"{EVAL_SEASONS[-1]} (frozen)")
    v = sub.add_parser("validate", help="validate a Phase 4A run directory")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "validate":
        m = validate_run(args.path)
        print(f"OK {args.path}: {m['files']['baseline_predictions']['rows']:,} predictions, "
              f"{m['files']['folds']['rows']} folds, baseline set v{m['baseline_set_version']}")
        return 0
    log.info("raw: %s, labels: %s, features: %s, evaluation: %s", data_root().resolve(),
             labels_root().resolve(), features_root().resolve(), evaluation_root().resolve())
    inputs = load_inputs((*WARMUP_SEASONS, *EVAL_SEASONS), EVAL_SEASONS)
    run = evaluate(inputs, EVAL_SEASONS)
    path = write_run(run, inputs, EVAL_SEASONS, WARMUP_SEASONS)
    print(f"Phase 4A run -> {path}")
    _print_summary(summarize(run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
