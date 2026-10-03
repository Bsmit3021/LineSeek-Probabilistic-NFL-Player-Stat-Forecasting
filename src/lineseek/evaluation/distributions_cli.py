"""`lineseek-distributions` CLI: Phase 4C distributional forecasts on the frozen protocol."""

import argparse
import logging
import sys
from pathlib import Path

from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation.phase4a import EVAL_SEASONS, WARMUP_SEASONS, load_inputs

log = logging.getLogger("lineseek.distributions")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-distributions")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="quantile/mean/zero models + empirical baseline, scored")
    b.add_argument("--phase4a", type=Path, default=None, help="default: latest Phase 4A run")
    b.add_argument("--phase4b", type=Path, default=None, help="default: latest Phase 4B run")
    b.add_argument("--workers", type=int, default=None)
    v = sub.add_parser("validate", help="validate a Phase 4C run directory")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "validate":
        m = p4c.validate_run(args.path)
        print(f"OK {args.path}: {m['files']['distribution_predictions']['rows']:,} rows, "
              f"distribution set v{m['distribution_set_version']}")
        return 0
    p4a = p4b.load_phase4a(args.phase4a or p4c.latest_run("phase4a"))
    p4bb = p4c.load_phase4b(args.phase4b or p4c.latest_run("phase4b"), p4a)
    log.info("Phase 4A: %s; Phase 4B: %s", p4a.path, p4bb.path)
    inputs = load_inputs((*WARMUP_SEASONS, *EVAL_SEASONS), EVAL_SEASONS)
    run = p4c.evaluate(inputs, p4a, p4bb, EVAL_SEASONS, workers=args.workers)
    path = p4c.write_run(run, inputs, p4a, p4bb, EVAL_SEASONS, WARMUP_SEASONS)
    print(f"Phase 4C run -> {path}")
    for k, val in p4c.summarize(run).items():
        print(f"  {k}: {val}")
    for w in (p4c.POINT_IN_TIME_WARNING, p4c.UNIVERSE_WARNING, p4c.F3_NOTE,
              p4c.RECONSTRUCTED_WARNING):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
