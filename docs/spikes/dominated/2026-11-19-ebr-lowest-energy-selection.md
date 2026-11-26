# EBR lowest-energy selection as archive replacement (`LowestEnergySelection`)

Archived 2026-11-19 (spike card backfilled 2026-11-26). Label: `dominated:claim`. The committed cells in [benchmarks/EBR_SPIKE_RESULTS.md](../../../benchmarks/EBR_SPIKE_RESULTS.md) are `still_true`.

Keeping only the per-weight `argmin E_w` sample instead of archiving the full trajectory collapses coverage on multi-modal / non-convex fronts (Potts 6-site coverage dropped from `1.0` to `0.4` on two of three seeds), and on THRML the argmin row was already the exact per-weight minimum on all 66 weights. The helper stays opt-in in `unsga3_extropic.ebr` for callers who explicitly want a single best point per weight, not as a replacement for the archive.

See [docs/ebr_note.md](../../ebr_note.md) and [benchmarks/EBR_SPIKE_RESULTS.md](../../../benchmarks/EBR_SPIKE_RESULTS.md).

## Denominator

```text
attempted: 1
filtered: 1
shipped: 0
proof_surface_pct: 100

proxy_kind: eval_budget
proxy_value: deep schedule (11 weights, 6 betas, 24 warmup, 24 samples, 2 steps; 1584 rows/seed, seeds 7/11/19); Potts coverage dropped 1.0 -> 0.4 on 2/3 seeds
proxy_artifact: benchmarks/EBR_SPIKE_RESULTS.md

backend_label: NumPy ExactEw
energy_identity: n/a
import_graph_matches_readme: yes

not_claimed: archive replacement; front quality improvement; default-on selection
dominated_path: dominated:claim

model_name: N/A
prompts: N/A
fail_list_shipped: yes
```
