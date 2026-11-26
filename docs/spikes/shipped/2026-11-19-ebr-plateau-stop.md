# EBR energy-plateau stopping rule (`PlateauStop`)

Shipped 2026-11-19 (spike card backfilled 2026-11-26). Opt-in per-stage stopping rule on `ExactEwMetropolisBackend(plateau=...)`; default path unchanged.

See [docs/ebr_note.md](../../ebr_note.md) and [benchmarks/EBR_SPIKE_RESULTS.md](../../../benchmarks/EBR_SPIKE_RESULTS.md).

## Denominator

```text
attempted: 2
filtered: 1
shipped: 1
proof_surface_pct: 100

proxy_kind: eval_budget
proxy_value: ~50% fewer recorded samples and 34-35% fewer energy evaluations on ExactEw codon (12-spin) and Potts (6-site, K=3), seeds 7/11/19, window=8 tol=0, with HV/GD/coverage >= baseline on all 6 task-seed pairs
proxy_artifact: benchmarks/EBR_SPIKE_RESULTS.md

backend_label: NumPy ExactEw
energy_identity: n/a
import_graph_matches_readme: yes

not_claimed: learned neural energy; PyTorch EBR trainer; THRML live in-stage stopping (THRML was only scored offline); default-on behavior; replacement of the non-dominated archive
dominated_path: none

model_name: N/A
prompts: N/A
fail_list_shipped: yes
```

`attempted: 2` covers `PlateauStop` (shipped opt-in) and `LowestEnergySelection` as an archive replacement (filtered and archived in [docs/spikes/dominated/2026-11-19-ebr-lowest-energy-selection.md](../dominated/2026-11-19-ebr-lowest-energy-selection.md)).
