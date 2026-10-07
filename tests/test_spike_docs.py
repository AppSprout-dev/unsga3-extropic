"""Spike-triage docs exist, and the ExactEw oracle cells stay in place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXACTEW_MEDIANS = (
    "2.3144182108708304",
    "3.3547378401332653",
    "0.27690988609911626",
)

LABELS = (
    "dominated:igd",
    "dominated:claim",
    "refused:thermalizers",
    "refused:sbx-in-loop",
    "watch:not_triggered",
    "still_true",
)


def test_spike_denominator_and_dominated_archive_exist():
    denominator = (ROOT / "docs/spikes/denominator.md").read_text(encoding="utf-8")
    for field in (
        "attempted",
        "filtered",
        "shipped",
        "proof_surface_pct",
        "proxy_kind",
        "eval_budget",
        "pop_x_gens",
        "wall_seconds_local",
        "backend_label",
        "THRML-native",
        "NumPy ExactEw",
        "classical fidelity",
        "Torx optional",
        "watch-only",
        "energy_identity",
        "import_graph_matches_readme",
        "not_claimed",
        "dominated_path",
        "model_name",
        "prompts",
        "fail_list_shipped",
        "N/A",
    ):
        assert field in denominator

    index = (ROOT / "docs/spikes/README.md").read_text(encoding="utf-8")
    for label in LABELS:
        assert label in index
    assert (ROOT / "docs/spikes/shipped/README.md").is_file()

    card = (
        ROOT / "docs/spikes/dominated/2026-10-07-exactew-continuous-not-unsga3.md"
    ).read_text(encoding="utf-8")
    assert "dominated:igd" in card
    assert "not U-NSGA-III" in card
    assert "not THRML-native" in card
    assert "ORACLE_RESULTS.md" in card
    assert "ORACLE_UNSGA3_RESULTS.md" in card
    assert "PR #19" in card
    for median in EXACTEW_MEDIANS:
        assert median in card

    oracle = (ROOT / "benchmarks/ORACLE_RESULTS.md").read_text(encoding="utf-8")
    for median in EXACTEW_MEDIANS:
        assert median in oracle
    assert "2026-10-07-exactew-continuous-not-unsga3.md" in oracle

    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    assert "Spike triage" in roadmap
    assert "docs/spikes/denominator.md" in roadmap
