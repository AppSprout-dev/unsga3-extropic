# Spike denominator

Copy the block below into every Extropic spike writeup. The process shape is attempt, then a multi-objective filter, then ship the winners. Filtered tries stay in the archive. This file is a template. It is not a measurement.

Fill every field. A blank is `N/A`, not an omission. Do not invent AGMAI `model_name` or `prompts`.

`proof_surface_pct` is an integer from 0 to 100. It is the share of the stated claim a reader can check from a committed artifact. 0 means the claim is assertion only. 100 means every stated number and label is in a committed file. It is not a quality score and it is not a license to widen the claim.

`energy_identity: yes` means a THRML-native backend's EBM energy matches \(w \cdot f\) within the tested tolerance in [ROADMAP.md](../../ROADMAP.md). `n/a` means that test does not apply (`NumPy ExactEw`, `classical fidelity`, `Torx optional`, `watch-only`). `fail` means the test was run and did not hold.

`import_graph_matches_readme: yes` means the README names only libraries this spike imports.

`fail_list_shipped: yes` means the tries that failed the filter are written down (this archive, or the artifact named in `proxy_artifact`). `no` means those tries were dropped. Shipping a winner does not allow dropping the fails.

`dominated_path` is one label from [docs/spikes/README.md](README.md), or `none` when the spike shipped.

```text
attempted:
filtered:
shipped:
proof_surface_pct:

proxy_kind:            # eval_budget | pop_x_gens | wall_seconds_local
proxy_value:
proxy_artifact:

backend_label:         # THRML-native | NumPy ExactEw | classical fidelity | Torx optional | watch-only
energy_identity:       # yes | n/a | fail
import_graph_matches_readme:

not_claimed:
dominated_path:

model_name: N/A
prompts: N/A
fail_list_shipped:     # yes | no
```

A README claim that a spike shipped names this filled denominator and a checkable artifact. A claim without both does not ship. The template does not add a sampler, and it does not move SBX or polynomial mutation into `WeightSweepLoop`.
