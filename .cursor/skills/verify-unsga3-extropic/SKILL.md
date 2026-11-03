---
name: verify-unsga3-extropic
description: Verify unsga3-extropic (THRML/Torx weight-sweep U-NSGA-III library) end to end through its real user surfaces, the demo CLIs, benchmark CLIs, and the CI pytest matrix, with a scripted launch/doctor/drive/cleanup harness. Reach for it before claiming a change to the loop, backends, problems, classical column, benchmarks, or records works, and for the monthly verification upkeep.
---

# Verify unsga3-extropic

The product is a Python library. Users touch it through four CLI surfaces, all run from the repo root:

| Surface | Entry points | Feature file |
|---|---|---|
| THRML demos (primary) | `demos/run_codon_thrml.py`, `demos/run_potts_thrml.py`, `demos/run_domain_wall.py` (`--smoke` or default) | [features/thrml-demos.md](features/thrml-demos.md) |
| Torx PSWAP (optional extra) | `demos/run_torx_pswap.py`, `pytest -m torx` | [features/torx-pswap.md](features/torx-pswap.md) |
| Oracle columns | `benchmarks/run_classical_unsga3.py`, `benchmarks/run_oracle_continuous.py` | [features/oracle-columns.md](features/oracle-columns.md) |
| Run records + deep | `benchmarks/append_run.py`, `benchmarks/run_deep.py` | [features/run-records-and-deep.md](features/run-records-and-deep.md) |

There is no server, port, or UI. Every drive is a short-lived process; "launch" means build the venv once. Nothing here talks to the network after install, calls TypeSafe, or touches hardware (no Thermalizers, no Z1).

Helper: `.cursor/skills/verify-unsga3-extropic/scripts/verify.sh` (below as `V`). Every drive writes a transcript per command (`$ cmd`, cwd, UTC date, short SHA, output, `rc=`) and appends a line to `SUMMARY.txt` in the evidence dir.

```bash
V=.cursor/skills/verify-unsga3-extropic/scripts/verify.sh
```

## Launch

```bash
$V launch     # creates .venv (python3.12 > 3.11 > python3), pip install -e ".[dev,cpu,torx]", pip install pymoo
```

Ready when it prints `READY venv=<path>`. CI uses Python 3.12; `extro-torx` needs >= 3.11. pymoo is not a package dependency; it is only for unsga3-bend's `ab/igd_vs_pymoo.py` (oracle journey). Override the venv with `VERIFY_VENV=/path`. Teardown of the venv is `rm -rf .venv` and is not part of normal cleanup.

## Doctor

```bash
$V doctor     # read-only; exit 0 and "DOCTOR OK" means worth driving
```

It checks: HEAD SHA and branch; venv exists; `unsga3_extropic` imports from **this** checkout (an editable install from another clone on the same box is the classic false pass); thrml/jax/numpy/pytest versions; torx and pymoo presence; the unsga3-bend IGD script (`$UNSGA3_BEND_ROOT` or `/workspace/unsga3-bend`); whether tracked benchmark records are already modified. Run it first and whenever a result looks off.

## Drive

```bash
$V drive ci          # exact mirror of .github/workflows/ci.yml (test + torx jobs)
$V drive demos       # three THRML smokes + proof of the runs.jsonl side effect
$V drive torx        # pytest -m torx + PSWAP demo
$V drive classical   # classical U-NSGA-III seed 1 on zdt1/zdt2/dtlz2; must be byte-identical to committed fronts
$V drive oracle      # ExactEw continuous seed 1 + Bend IGD; igd= must equal committed sidecars
$V drive records     # append_run --stub/--smoke into evidence, schema-validated
$V drive deep-codon  # run_deep.py --only codon --only exact_ew --skip-torx (~3 min CPU), redirected into evidence
```

Exit code is non-zero if any step failed. `drive` refuses to start if `benchmarks/records/{runs.jsonl,oracle_fronts,classical_unsga3}` already have uncommitted changes, because cleanup restores them with git. Multiple drives in one session share one evidence dir until `cleanup`.

Isolation: two runs in the same checkout collide on `benchmarks/records/runs.jsonl`, `demos/_out/`, and the tracked front CSVs. Use a separate clone or `git worktree` per concurrent run (on the shared box, create it under the `shared-box-lock` rules). JAX on CPU is process-local, so parallel clones are fine.

## Evidence

Default location: `.verify-evidence/<UTC stamp>/` in the repo (gitignored), or `VERIFY_EVIDENCE=/abs/path`. Contents: `<step>.log` transcripts, `SUMMARY.txt`, `demos_out/` (summary JSON, `.npy`/`.npz` fronts), `demos_runs_jsonl.numstat`, `classical_*_s1.md`, `classical_diff.txt`, `oracle_*_s1.md`, `oracle_runs.jsonl`, `oracle_diff.txt`, `records/`, `deep/`.

Proof standards:
- Drive the CLIs a user runs, not internal functions. Pytest is CI parity, not a substitute for a journey.
- Prove the action and the end state: the printed metrics **and** the files the CLI writes (front `.npz`/CSV, JSONL rows, report markdown).
- Side effects are part of the proof: demos append one row each to the tracked `runs.jsonl` (check the numstat), and benchmark CLIs rewrite tracked front CSVs. A rerun at the same SHA must reproduce committed fronts byte for byte (classical) or by `igd=` value (oracle).
- No mocks. THRML, Torx, and pymoo run for real on CPU. The only external boundary is unsga3-bend's IGD script, which is run as-is.
- Report the date, SHA, Python/JAX/THRML versions (from doctor), and each step's `rc`.

## Cleanup

```bash
$V cleanup    # git checkout the tracked record paths, remove demos/_out, drop .verify-evidence/.state
```

It only restores paths when this harness's `drive` started from a clean records state, and never deletes the evidence dir; it lists what it kept. There are no background processes to kill.

## Feature map

[features/README.md](features/README.md) is the maintained index. Keep it honest with `/maintain-verification-skill` when CLIs, flags, or outputs change.
