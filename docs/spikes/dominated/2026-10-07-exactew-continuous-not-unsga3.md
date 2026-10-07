# ExactEw continuous is not U-NSGA-III

Archived 2026-10-07. The table was generated 2026-10-02. Label: `dominated:igd`. The committed cells are `still_true`.

This card is the continuous NumPy ExactEw weight-sweep on ZDT1, ZDT2, and DTLZ2 (`ExactEwContinuousBackend`). It is **not U-NSGA-III**. It is **not THRML-native**. It is not an Ising model, not a Potts factor, and not a device run. Bend and C# are fidelity references for the published medians below. This repository is not a port of either one.

The apples-to-apples fidelity path is the separate classical column: [benchmarks/ORACLE_UNSGA3_RESULTS.md](../../../benchmarks/ORACLE_UNSGA3_RESULTS.md), recorded in PR #19. That column is NumPy generational U-NSGA-III. `WeightSweepLoop` does not call it, and this card does not move SBX or polynomial mutation into `WeightSweepLoop`.

## Denominator

```text
attempted: 1
filtered: 1
shipped: 0
proof_surface_pct: 100

proxy_kind: eval_budget
proxy_value: zdt1 5200; zdt2 13000; dtlz2 13741 objective calls (pop×gens yardstick 5200 / 13000 / 13800)
proxy_artifact: benchmarks/ORACLE_RESULTS.md

backend_label: NumPy ExactEw
energy_identity: n/a
import_graph_matches_readme: yes

not_claimed: U-NSGA-III; THRML-native; Ising or Potts factor energy; Thermalizers; Z1 or any device; a Bend or C# port; SBX or polynomial mutation inside WeightSweepLoop
dominated_path: dominated:igd

model_name: N/A
prompts: N/A
fail_list_shipped: yes
```

`attempted: 1` is this spike, not a seed count. Seeds 1–15 on each of the three problems are inside the artifact. All of them stay. `shipped: 0` means no README product claim treats this sampler as U-NSGA-III or as THRML. `proof_surface_pct: 100` means the medians and the not-U-NSGA-III / not-THRML labels are the committed text of `benchmarks/ORACLE_RESULTS.md`. `energy_identity: n/a` means the THRML EBM check does not apply. Host-side \(E_w = w \cdot f(x)\) is exact on this backend by construction, and that scalarization is not an `IsingEBM` or Potts factor energy. `fail_list_shipped: yes` means the seed table is still in the oracle file. `model_name` and `prompts` are N/A. Nothing in this spike was a language-model run.

## Median IGD

Source: [benchmarks/ORACLE_RESULTS.md](../../../benchmarks/ORACLE_RESULTS.md). Extropic cells are the middle `igd=` value over seeds 1–15. Bend and C# cells are the published PymooCompatible medians cited in that file. A larger IGD is farther from the reference front.

| Problem | median ExactEw | median Bend (published) | median C# (published) |
|---------|---------------:|------------------------:|----------------------:|
| zdt1 | 2.3144182108708304 | 0.071475 | 0.082152 |
| zdt2 | 3.3547378401332653 | 0.025621 | 0.018684 |
| dtlz2 | 0.27690988609911626 | 0.004243 | 0.004512 |

Rounded from those cells: about 2.31, about 3.35, and about 0.277. The full strings above are the record. Do not delete them from `benchmarks/ORACLE_RESULTS.md`.
