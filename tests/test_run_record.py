"""Run-record schema, append, and the stub CLI (no THRML sample)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unsga3_extropic.fidelity import load_front
from unsga3_extropic.results import (
    METRIC_KEYS,
    RUN_RECORD_KEYS,
    SCHEDULE_KEYS,
    append_run_record,
    measure_potts_chain,
    read_run_records,
    stub_run_record,
    validate_run_record,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "benchmarks" / "run_record.schema.json"


def test_schema_file_matches_the_validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["required"] == list(RUN_RECORD_KEYS)
    assert schema["properties"]["schedule"]["required"] == list(SCHEDULE_KEYS)
    assert schema["properties"]["metrics"]["required"] == list(METRIC_KEYS)
    assert schema["properties"]["schema_version"]["const"] == 1


def test_append_and_read_roundtrip(tmp_path):
    front = tmp_path / "artifacts" / "stub.npz"
    record = stub_run_record(front_path=front)
    validate_run_record(record)
    dest = tmp_path / "runs.jsonl"
    append_run_record(dest, record)
    append_run_record(dest, record)
    loaded = read_run_records(dest)
    assert len(loaded) == 2
    assert loaded[0]["problem"] == "stub"
    assert loaded[0]["metrics"]["nd_count"] == 0
    assert loaded[0]["metrics"]["hypervolume_2d"] is None
    assert front.is_file()


def test_rejects_a_record_missing_metrics(tmp_path):
    record = stub_run_record(front_path=tmp_path / "front.npz")
    del record["metrics"]
    with pytest.raises(ValueError, match="metrics"):
        validate_run_record(record)
    record = stub_run_record(front_path=tmp_path / "front2.npz")
    record["metrics"]["coverage"] = 1.5
    with pytest.raises(ValueError, match="coverage"):
        append_run_record(tmp_path / "bad.jsonl", record)


def test_stub_cli_appends_jsonl(tmp_path):
    script = ROOT / "benchmarks" / "append_run.py"
    out = tmp_path / "runs.jsonl"
    front = tmp_path / "artifacts" / "stub.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stub",
            "--out",
            str(out),
            "--front",
            str(front),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stub" in completed.stdout
    loaded = read_run_records(out)
    assert loaded[0]["backend"] == "none"
    assert loaded[0]["notes"].startswith("stub record")
    assert "appended stub" in completed.stdout


@pytest.mark.thrml
def test_potts_smoke_record_fills_metrics(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("thrml")
    front_path = tmp_path / "front.npz"
    record = measure_potts_chain(smoke=True, front_path=front_path)
    append_run_record(tmp_path / "runs.jsonl", record)
    assert record["issue"] == "4"
    assert record["phase"] == "1"
    assert record["problem"] == "potts_chain"
    assert record["backend"] == "thrml_potts"
    assert record["eval_budget"] == 2 * 2 * 4
    assert record["notes"].startswith("profile=smoke.")
    assert "schedule_product=16." in record["notes"]
    assert record["metrics"]["nd_count"] >= 1
    assert record["metrics"]["hypervolume_2d"] >= 0.0
    assert record["metrics"]["generational_distance"] >= 0.0
    assert 0.0 <= record["metrics"]["coverage"] <= 1.0
    loaded = load_front(front_path, n_obj=2)
    assert len(loaded) == record["metrics"]["nd_count"]
    assert read_run_records(tmp_path / "runs.jsonl")[0]["backend"] == "thrml_potts"
