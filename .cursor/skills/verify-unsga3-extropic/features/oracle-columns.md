# Oracle columns

Two benchmark CLIs score continuous ZDT1 (n=30), ZDT2 (n=30), and DTLZ2 (M=3, k=10) at the Bend ORACLE-MULTISEED budgets: a classical NumPy U-NSGA-III (SBX + polynomial mutation) with in-repo IGD, and a NumPy ExactEw weight sweep whose IGD comes only from unsga3-bend's `ab/igd_vs_pymoo.py`. Both write front CSVs plus `igd=` sidecars and a markdown table.

## Sub-features

- `classical-run` one problem/seed reproduces the committed front CSV and `igd=` sidecar under `records/classical_unsga3/`.
- `classical-report` `--report` writes a (partial) results table.
- `oracle-run` one problem/seed reproduces the committed `records/oracle_fronts/` CSV and `igd=` value via Bend's script.
- `oracle-resume` a row already in `--log` with an `igd=` sidecar is resumed, not rerun.

## How to get to it (user POV)

- `python benchmarks/run_classical_unsga3.py [--problem zdt1|zdt2|dtlz2|all] [--seed N ...] [--report PATH]`
- `UNSGA3_BEND_ROOT=/path/to/unsga3-bend python benchmarks/run_oracle_continuous.py [--problems a,b] [--seeds 1-15] [--log PATH] [--report PATH]`

## Driving it with verify.sh

Preconditions:

- Doctor shows `pymoo:` with a version. `bend:` either names the pinned private clone, says one will be cloned (network needed), or names an explicit `UNSGA3_BEND_ROOT`.

- **Classical.** `$V drive classical`. Three `classical_*_s1` steps end `rc=0`; Seed-1 lines: `zdt1 seed 1: igd=0.05827076935009233 front_rows=52 evals=5200`, `zdt2 seed 1: igd=0.01703131322864674 front_rows=52 evals=13000`, `dtlz2 seed 1: igd=0.003679800984466357 front_rows=92 evals=13800`. `SUMMARY.txt` says `classical fronts byte-identical to committed`.
- **Oracle.** `$V drive oracle`. Three `oracle_*_s1` steps end `rc=0`; Seed-1 lines: `scored zdt1 seed 1 igd=2.573954812980915 pf_rows=500`, `scored zdt2 seed 1 igd=2.800481960915036 pf_rows=500`, `scored dtlz2 seed 1 igd=0.22501888314606172 pf_rows=91`. `SUMMARY.txt` says `oracle seed-1 IGD and fronts identical to committed`.
- **Reports.** `classical_*_s1.md` and `oracle_*_s1.md` exist in evidence and are marked partial; the tracked `ORACLE_RESULTS.md` / `ORACLE_UNSGA3_RESULTS.md` are untouched.

## Gotchas

- Called directly without `UNSGA3_BEND_ROOT`, `run_oracle_continuous.py` falls back to `/workspace/unsga3-bend`, a shared box checkout on whatever branch another agent left it. Go through `$V drive oracle`, which always passes the pinned clone explicitly (`oracle_bend.txt` records which).
- Without pymoo, Bend's script exits 0 and prints `skip: pymoo is not installed` to stderr; `run_oracle_continuous.py` then raises `igd_vs_pymoo.py did not print igd=`. Install pymoo in the venv (`$V launch` does).
- Both CLIs always write front CSVs/sidecars into the tracked `benchmarks/records/` dirs; there is no flag to redirect them. The harness diffs then restores them.
- Oracle `igd.txt` sidecars embed `front=<absolute path of the checkout that scored them>`, so any rerun in a different directory shows a one-line diff. Ignore that line; compare `igd=` and the CSV.
- Pass `--log` to an evidence path for the oracle journey; with the default tracked log, already-logged seeds are resumed and nothing is re-measured.
- Full 15-seed runs take minutes and rewrite the tracked results tables; that is a benchmark refresh, not a verification.
