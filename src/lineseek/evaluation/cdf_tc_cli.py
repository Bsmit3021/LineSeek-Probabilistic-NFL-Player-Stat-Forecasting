"""`lineseek-cdf-tc` CLI: Phase 4C.1b threshold-conditioned monotone CDF (final Phase 4C gate)."""

import argparse
import logging
import sys
from pathlib import Path

from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation import phase4c1 as p4c1
from lineseek.evaluation import phase4c1b as p4c1b
from lineseek.evaluation.phase4a import EVAL_SEASONS, WARMUP_SEASONS, load_inputs

log = logging.getLogger("lineseek.cdf_tc")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-cdf-tc")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="threshold-conditioned CDF, scored against Phase 4C.1")
    b.add_argument("--phase4a", type=Path, default=None, help="default: latest Phase 4A run")
    b.add_argument("--phase4c", type=Path, default=None, help="default: latest Phase 4C run")
    b.add_argument("--phase4c1", type=Path, default=None, help="default: latest Phase 4C.1 run")
    b.add_argument("--workers", type=int, default=p4c1b.WORKERS)
    v = sub.add_parser("validate", help="validate a Phase 4C.1b run directory")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "validate":
        m = p4c1b.validate_run(args.path)
        print(f"OK {args.path}: {m['files']['cdf_predictions']['rows']:,} rows, "
              f"CDF set v{m['cdf_set_version']}, Phase 5: {m['summary']['phase5']}")
        return 0
    p4a = p4b.load_phase4a(args.phase4a or p4c.latest_run("phase4a"))
    p4cc = p4c1.load_phase4c(args.phase4c or p4c.latest_run("phase4c"), p4a)
    p41 = p4c1b.load_phase4c1(args.phase4c1 or p4c.latest_run("phase4c1"), p4a, p4cc)
    log.info("Phase 4A: %s; Phase 4C: %s; Phase 4C.1: %s", p4a.path, p4cc.path, p41.path)
    inputs = load_inputs((*WARMUP_SEASONS, *EVAL_SEASONS), EVAL_SEASONS)
    run = p4c1b.evaluate(inputs, p4a, p4cc, p41, EVAL_SEASONS, workers=args.workers)
    path = p4c1b.write_run(run, inputs, p4a, p4cc, p41, EVAL_SEASONS)
    print(f"Phase 4C.1b run -> {path}")
    for k, val in p4c1b.summarize(run).items():
        print(f"  {k}: {val}")
    for w in (p4c1b.POINT_IN_TIME_WARNING, p4c1b.UNIVERSE_WARNING, p4c1b.F3_NOTE,
              p4c.RECONSTRUCTED_WARNING):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
