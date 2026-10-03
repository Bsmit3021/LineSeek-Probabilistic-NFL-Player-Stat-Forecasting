"""`lineseek-cdf` CLI: Phase 4C.1 discrete CDF P(Y <= k) on the frozen protocol."""

import argparse
import logging
import sys
from pathlib import Path

from lineseek.evaluation import phase4b as p4b
from lineseek.evaluation import phase4c as p4c
from lineseek.evaluation import phase4c1 as p4c1
from lineseek.evaluation.phase4a import EVAL_SEASONS, WARMUP_SEASONS, load_inputs

log = logging.getLogger("lineseek.cdf")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lineseek-cdf")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="threshold classifiers + empirical CDF, scored")
    b.add_argument("--phase4a", type=Path, default=None, help="default: latest Phase 4A run")
    b.add_argument("--phase4c", type=Path, default=None, help="default: latest Phase 4C run")
    b.add_argument("--workers", type=int, default=None)
    v = sub.add_parser("validate", help="validate a Phase 4C.1 run directory")
    v.add_argument("path", type=Path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "validate":
        m = p4c1.validate_run(args.path)
        print(f"OK {args.path}: {m['files']['cdf_predictions']['rows']:,} rows, "
              f"CDF set v{m['cdf_set_version']}, Phase 5: {m['summary']['phase5']}")
        return 0
    p4a = p4b.load_phase4a(args.phase4a or p4c.latest_run("phase4a"))
    p4cc = p4c1.load_phase4c(args.phase4c or p4c.latest_run("phase4c"), p4a)
    log.info("Phase 4A: %s; Phase 4C: %s", p4a.path, p4cc.path)
    inputs = load_inputs((*WARMUP_SEASONS, *EVAL_SEASONS), EVAL_SEASONS)
    run = p4c1.evaluate(inputs, p4a, p4cc, EVAL_SEASONS, workers=args.workers)
    path = p4c1.write_run(run, inputs, p4a, p4cc, EVAL_SEASONS)
    print(f"Phase 4C.1 run -> {path}")
    for k, val in p4c1.summarize(run).items():
        print(f"  {k}: {val}")
    for w in (p4c1.POINT_IN_TIME_WARNING, p4c1.UNIVERSE_WARNING, p4c1.F3_NOTE,
              p4c.RECONSTRUCTED_WARNING):
        print(f"  WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
