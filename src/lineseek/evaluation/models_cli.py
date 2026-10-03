"""`lineseek-models` CLI: Phase 4B walk-forward Ridge/LightGBM on the frozen Phase 4A run."""

import argparse
import logging
import sys
from pathlib import Path

from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation.phase4a import (
    EVAL_SEASONS,
    WARMUP_SEASONS,
    evaluation_root,
    load_inputs,
)

log = logging.getLogger("lineseek.models")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-models")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fit/predict on the frozen Phase 4A folds and score")
    b.add_argument("--phase4a", type=Path, default=None,
                   help="Phase 4A run directory (default: latest under the evaluation root)")
    b.add_argument("--workers", type=int, default=None,
                   help="processes (one market/model per task; results do not depend on it)")
    v = sub.add_parser("validate", help="validate a Phase 4B run directory")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "validate":
        m = p4b.validate_run(args.path)
        print(f"OK {args.path}: {m['files']['model_predictions']['rows']:,} predictions, "
              f"model set v{m['model_set_version']}, Phase 4A {m['phase4a']['content_sha256'][:12]}")
        return 0
    p4a_path = args.phase4a or p4b.latest_phase4a()
    log.info("Phase 4A run: %s; evaluation root: %s", p4a_path, evaluation_root().resolve())
    p4a = p4b.load_phase4a(p4a_path)
    inputs = load_inputs((*WARMUP_SEASONS, *EVAL_SEASONS), EVAL_SEASONS)
    run = p4b.evaluate(inputs, p4a, EVAL_SEASONS, workers=args.workers)
    path = p4b.write_run(run, inputs, p4a, EVAL_SEASONS, WARMUP_SEASONS)
    print(f"Phase 4B run -> {path}")
    for k, val in p4b.summarize(run).items():
        print(f"  {k}: {val}")
    for w in (p4b.POINT_IN_TIME_WARNING, p4b.UNIVERSE_WARNING, p4b.F3_NOTE):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
