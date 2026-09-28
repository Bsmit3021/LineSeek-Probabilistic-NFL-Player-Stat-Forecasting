"""Frozen raw-source schemas (raw_schemas.json) and the conform step that enforces them."""

import json
from importlib.resources import files

import polars as pl

RAW_SCHEMAS: dict[str, dict[str, str]] = json.loads(
    (files("lineseek.contracts") / "raw_schemas.json").read_text()
)

# Reviewed historical dtype drift, cast to the current-season dtype; anything else fails.
KNOWN_CASTS: dict[str, dict[str, pl.DataType]] = {
    "injuries": {"season": pl.Int32, "week": pl.Int32},  # Float64 in 2018-2020
    "rosters_weekly": {"height": pl.Int32},  # Float64 before 2025
    "pbp": {"goal_to_go": pl.Float64},  # Int32 in 2021-2023
}


class SchemaDriftError(ValueError):
    pass


def schema_variant(source: str, season: int) -> str:
    if source == "depth_charts":
        return "depth_charts/legacy" if season <= 2024 else "depth_charts/snapshot"
    if source == "injuries":
        return "injuries/timestamped" if season <= 2024 else "injuries/untimestamped"
    return source


def conform(df: pl.DataFrame, source: str, season: int) -> pl.DataFrame:
    """Apply known casts, then require an exact column/dtype match with the frozen schema."""
    variant = schema_variant(source, season)
    casts = {c: t for c, t in KNOWN_CASTS.get(source, {}).items()
             if c in df.columns and df.schema[c] != t}
    df = df.with_columns(pl.col(c).cast(t, strict=True) for c, t in casts.items())
    expected = RAW_SCHEMAS[variant]
    actual = {c: str(t) for c, t in df.schema.items()}
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    changed = {c: (expected[c], actual[c]) for c in set(expected) & set(actual)
               if expected[c] != actual[c]}
    if added or removed or changed:
        raise SchemaDriftError(
            f"{variant} season={season}: added={added} removed={removed} changed={changed}"
        )
    return df
