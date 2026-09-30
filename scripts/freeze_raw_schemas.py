"""Regenerate src/lineseek/contracts/raw_schemas.json from current nflreadpy output (network)."""

import json
from pathlib import Path

import nflreadpy as nfl

OUT = Path(__file__).resolve().parents[1] / "src/lineseek/contracts/raw_schemas.json"

# (variant, loader, reference season): the newest season of each format.
REFERENCES = [
    ("player_stats", nfl.load_player_stats, 2025),
    ("snap_counts", nfl.load_snap_counts, 2025),
    ("schedules", nfl.load_schedules, 2025),
    ("pbp", nfl.load_pbp, 2025),
    ("rosters_weekly", nfl.load_rosters_weekly, 2025),
    ("injuries/timestamped", nfl.load_injuries, 2024),
    ("injuries/untimestamped", nfl.load_injuries, 2025),
    ("depth_charts/legacy", nfl.load_depth_charts, 2024),
    ("depth_charts/snapshot", nfl.load_depth_charts, 2025),
    ("players", lambda season: nfl.load_players(), 2025),
]


def main() -> None:
    schemas = {v: {c: str(t) for c, t in f(s).schema.items()} for v, f, s in REFERENCES}
    OUT.write_text(json.dumps(schemas, indent=1) + "\n")
    print(f"wrote {OUT} ({len(schemas)} variants)")


if __name__ == "__main__":
    main()
