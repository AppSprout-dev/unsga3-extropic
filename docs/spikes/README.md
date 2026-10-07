# Spike archive

Index for Extropic spike writeups. The process shape is attempt, then a multi-objective filter, then ship the winners. Fails stay here. The fill-in block is [denominator.md](denominator.md). The phase is **Spike triage** in [ROADMAP.md](../../ROADMAP.md). It is docs/process. It is not a sampler.

## Labels

| Label | Use when |
|-------|----------|
| `dominated:igd` | Measured IGD on the stated budget is farther from the reference front than the fidelity numbers the card cites |
| `dominated:claim` | The writeup claimed more than the import graph or the energy identity supports |
| `refused:thermalizers` | The proposal invented a Thermalizers API. No public package is the trigger in ROADMAP phase 5 |
| `refused:sbx-in-loop` | The proposal put SBX or polynomial mutation into `WeightSweepLoop`. Those operators stay in `unsga3_extropic.classical` |
| `watch:not_triggered` | Thermalizers or hardware was watched. The public-API trigger has not tripped |
| `still_true` | The archived fact is still the record. Numbers named by the card are not deleted |

## Active

None.

## Shipped

None yet. A spike lands in [shipped/](shipped/) only after its denominator is filled and the README claim names a checkable artifact.

## Dominated

| Date | Card | Labels |
|------|------|--------|
| 2026-10-07 | [ExactEw continuous, not U-NSGA-III](dominated/2026-10-07-exactew-continuous-not-unsga3.md) | `dominated:igd`, `still_true` |
