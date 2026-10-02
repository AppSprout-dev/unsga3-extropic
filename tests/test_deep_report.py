"""The deep summary is a table over records. It does not sample."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _report():
    path = ROOT / "benchmarks" / "deep_report.py"
    spec = importlib.util.spec_from_file_location("deep_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(*, backend: str, seed: int, coverage: float, notes: str) -> dict:
    return {
        "backend": backend,
        "problem": "codon_ising" if backend != "thrml_potts" else "potts_chain",
        "seeds": [seed, seed + 1009],
        "schedule": {
            "betas": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
            "n_warmup": 24,
            "n_samples": 24,
            "steps_per_sample": 2,
            "n_weights": 11,
        },
        "eval_budget": 1584,
        "metrics": {
            "nd_count": 4,
            "hypervolume_2d": 1.5,
            "hv_ref": [0.0, 1.0],
            "generational_distance": 0.25,
            "coverage": coverage,
        },
        "git_sha": "abc123def4567890",
        "notes": notes,
    }


def test_summary_labels_deep_and_refuses_hardware_claims():
    report = _report()
    notes = (
        "profile=deep. base_seed=7. replicate=1/3. "
        "instance=n_spins=12, J=1, field_seed=0, global_bias=0.05. "
        "Reference front size 9. "
        "Closer-in-niche survival kept 3 of 4 non-dominated rows across 11 directions (quota 1). "
        "elapsed_s=1.250."
    )
    deep = [
        _record(backend="thrml_ising", seed=7, coverage=0.5, notes=notes),
        _record(
            backend="thrml_ising",
            seed=11,
            coverage=1.0,
            notes=notes.replace("base_seed=7", "base_seed=11").replace("replicate=1/3", "replicate=2/3"),
        ),
    ]
    prior = [
        {
            "backend": "thrml_ising",
            "problem": "codon_ising",
            "seeds": [7, 1016],
            "schedule": {
                "betas": [1.0, 4.0],
                "n_warmup": 4,
                "n_samples": 6,
                "steps_per_sample": 1,
                "n_weights": 2,
            },
            "eval_budget": 24,
            "metrics": {
                "nd_count": 1,
                "hypervolume_2d": 8.0,
                "hv_ref": [0.0, 1.0],
                "generational_distance": 0.0,
                "coverage": 0.3,
            },
            "git_sha": "34044ac48cb4",
            "notes": "historical smoke without a profile tag",
        }
    ]
    torx = {
        "p_swap": 0.3,
        "n_samples": 100_000,
        "git_sha": "abc123def4567890",
        "draws": [
            {
                "seed": 0,
                "stay_rate": 0.7,
                "swap_rate": 0.3,
                "elapsed_s": 0.5,
                "within_bound": True,
            }
        ],
    }
    text = report.render_deep_results(
        deep_records=deep,
        prior_records=prior,
        torx=torx,
        generated_at="2026-10-02T18:00:00Z",
    )
    assert "not Z1 measurements" in text
    assert "do not use Thermalizers" in text
    assert "thrml_ising" in text
    assert "1584" in text
    assert "3/4" in text
    assert "smoke (historical)" in report.prior_label(prior[0])
    assert "0.5000 / 0.7500 / 1.0000" in text
    assert "not a THRML program" in text
    assert "Mean swap rate" in text
    assert "n_sites=6, K=3" in text


def test_deep_cli_help_does_not_sample():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "run_deep.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "deep" in completed.stdout
    assert "--only" in completed.stdout
    assert "--skip-torx" in completed.stdout
