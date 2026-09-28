import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import polars as pl
import pytest

from lineseek.contracts import CONTRACT_VERSION, injury_report_available_at
from lineseek.contracts.raw import SchemaDriftError, conform
from lineseek.ingest import pipeline
from lineseek.ingest.snapshot import DATA_ROOT_ENV, data_root, list_manifests, write_snapshot
from lineseek.ingest.sources import SOURCES, Source

FIX = Path(__file__).parent / "fixtures"
T0 = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)


def fixture(name: str) -> pl.DataFrame:
    return pl.read_parquet(FIX / f"{name}.parquet")


@pytest.fixture
def fake_loaders(monkeypatch):
    """Point every source loader at checked-in fixtures instead of the network."""
    def use(**frames: pl.DataFrame):
        for name, df in frames.items():
            monkeypatch.setitem(SOURCES, name, Source(
                name, lambda season, df=df: df, SOURCES[name].prepare,
                SOURCES[name].completed_games_only,
            ))
    return use


def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    assert data_root() == Path("data/raw")
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_root() == tmp_path


def test_manifest_contents(tmp_path):
    df = fixture("snap_counts_2025")
    m = write_snapshot(df, "snap_counts", "snap_counts", [2025], [1], T0, {"snaps": "postgame"},
                       root=tmp_path)
    parquet = tmp_path / m.path
    on_disk = json.loads(parquet.with_suffix(".manifest.json").read_text())
    assert on_disk == json.loads(m.to_json())
    assert on_disk["source"] == "snap_counts"
    assert on_disk["seasons"] == [2025] and on_disk["weeks"] == [1]
    assert on_disk["pulled_at"] == "2026-09-28T12:00:00+00:00"
    assert on_disk["row_count"] == df.height == 6
    assert on_disk["schema"]["offense_snaps"] == "Float64"
    assert on_disk["sha256"] == sha256(parquet.read_bytes()).hexdigest()
    assert on_disk["bytes"] == parquet.stat().st_size
    assert on_disk["nflreadpy_version"] == "0.1.5"
    assert on_disk["contract_version"] == CONTRACT_VERSION
    assert on_disk["temporal_quality"] == {"snaps": "postgame"}
    assert "git_revision" in on_disk
    assert pl.read_parquet(parquet).equals(df)
    assert m.path.startswith("snap_counts/season=2025/")


def test_snapshots_never_overwrite(tmp_path):
    df = fixture("snap_counts_2025")
    first = write_snapshot(df, "snap_counts", "snap_counts", [2025], None, T0, root=tmp_path)
    original = (tmp_path / first.path).read_bytes()
    with pytest.raises(FileExistsError):
        write_snapshot(df, "snap_counts", "snap_counts", [2025], None, T0, root=tmp_path)
    assert (tmp_path / first.path).read_bytes() == original
    write_snapshot(df.head(2), "snap_counts", "snap_counts", [2025], None,
                   T0.replace(second=1), root=tmp_path)
    assert len(list_manifests("snap_counts", 2025, tmp_path)) == 2


def test_backfill_skips_existing_and_appends_on_force(tmp_path, fake_loaders):
    fake_loaders(snap_counts=fixture("snap_counts_2025"))
    assert len(pipeline.backfill([2025], ["snap_counts"], root=tmp_path)) == 1
    assert pipeline.backfill([2025], ["snap_counts"], root=tmp_path) == []
    pipeline.backfill([2025], ["snap_counts"], force=True, root=tmp_path)
    assert len(list_manifests("snap_counts", 2025, tmp_path)) == 2


def test_schedule_kickoff_et_to_utc_across_dst_change():
    out = SOURCES["schedules"].build(fixture("schedules_2026"), 2026, T0)
    kick = dict(out.select("game_id", "kickoff_utc").iter_rows())
    # 13:00 ET is 17:00Z under EDT (Oct 25) and 18:00Z under EST (Nov 1).
    assert kick["2026_07_SF_ATL"] == datetime(2026, 10, 25, 17, 0, tzinfo=UTC)
    assert kick["2026_08_MIN_DET"] == datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
    assert out.schema["kickoff_utc"] == pl.Datetime("us", "UTC")
    assert set(out["temporal_quality"]) == {"exact"}


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: df.drop("gametime"), "removed=\\['gametime'\\]"),
        (lambda df: df.with_columns(pl.lit(1).alias("new_col")), "added=\\['new_col'\\]"),
        (lambda df: df.with_columns(pl.col("week").cast(pl.Int64)), "changed=.*week"),
    ],
)
def test_schema_drift_fails_loudly(mutate, match):
    with pytest.raises(SchemaDriftError, match=match):
        SOURCES["schedules"].build(mutate(fixture("schedules_2026")), 2026, T0)


def test_known_historical_cast_is_applied():
    raw = fixture("injuries_2020")
    assert raw.schema["season"] == pl.Float64
    assert conform(raw, "injuries", 2020).schema["season"] == pl.Int32


def test_depth_chart_format_switch_by_season():
    legacy_cols = fixture("injuries_2020")  # any non-depth frame is drift for depth charts
    with pytest.raises(SchemaDriftError, match="depth_charts/legacy"):
        conform(legacy_cols, "depth_charts", 2024)
    out = SOURCES["depth_charts"].build(fixture("depth_charts_2025"), 2025, T0)
    assert set(out["temporal_quality"]) == {"exact"}
    assert out["available_at"].dtype == pl.Datetime("us", "UTC")
    assert out["season"].to_list() == [2025] * out.height


def test_injury_temporal_quality_by_era():
    old = SOURCES["injuries"].build(fixture("injuries_2020"), 2020, T0, fixture("schedules_2020"))
    assert set(old["temporal_quality"]) == {"exact"}
    assert old["available_at"].equals(old["date_modified"], check_names=False)

    sched = fixture("schedules_2025")
    new = SOURCES["injuries"].build(fixture("injuries_2025"), 2025, T0, sched)
    rule = injury_report_available_at(sched["gameday"][0])
    assert set(new["temporal_quality"]) == {"inferred"}
    assert set(new["available_at"]) == {rule}
    # Observed before the rule time -> our observation time is the tighter bound.
    early = SOURCES["injuries"].build(fixture("injuries_2025"), 2025, datetime(2025, 9, 1, tzinfo=UTC),
                                      sched)
    assert set(early["available_at"]) == {datetime(2025, 9, 1, tzinfo=UTC)}


def test_injuries_without_scheduled_game_raise():
    wrong_week = fixture("schedules_2025").with_columns(pl.lit(9, dtype=pl.Int32).alias("week"))
    with pytest.raises(ValueError, match="without a scheduled game"):
        SOURCES["injuries"].build(fixture("injuries_2025"), 2025, T0, wrong_week)


def test_cancelled_game_injuries_use_reviewed_gameday():
    # 2022 wk 17 BUF@CIN is absent from schedules; the explicit entry supplies its gameday.
    inj = fixture("injuries_2020").with_columns(
        pl.lit(2022.0).alias("season"), pl.lit(17.0).alias("week"), pl.lit("BUF").alias("team")
    )
    out = SOURCES["injuries"].build(inj, 2022, T0, fixture("schedules_2020"))
    assert set(out["rule_available_at"]) == {injury_report_available_at("2023-01-02")}


def test_pull_filters_to_completed_games(tmp_path, fake_loaders):
    snaps = fixture("snap_counts_2025")
    keep = snaps["game_id"][0]
    fake_loaders(snap_counts=snaps)
    m, df = pipeline.pull("snap_counts", 2025, weeks=[1], completed_game_ids=[keep], root=tmp_path)
    assert set(df["game_id"]) == {keep}
    assert m.row_count == df.height and m.weeks == [1]
    assert set(df["temporal_quality"]) == {"postgame"}
