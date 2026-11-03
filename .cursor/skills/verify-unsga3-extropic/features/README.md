# unsga3-extropic verification map

Maintained source for verifying the user-facing CLIs of unsga3-extropic. Read this index, then use the matching feature file as the recipe. Harness: `.cursor/skills/verify-unsga3-extropic/scripts/verify.sh` (`$V`).

## Baseline preconditions

- A clone of AppSprout-dev/unsga3-extropic you own (not a tree another agent is driving).
- `$V launch` printed `READY`, and `$V doctor` printed `DOCTOR OK` with the package path inside this clone.
- `benchmarks/records/` has no uncommitted changes (doctor shows `records:  clean`).
- `JAX_PLATFORMS=cpu` (the harness exports it).

## Driving conventions

- Run everything from the repo root through `$V drive <journey>`; it logs `$ cmd`, output, and `rc=` per step.
- Redirect outputs with the CLIs' own flags (`--out`, `--report`, `--log`, `--summary`) whenever they exist. Where they do not, the harness diffs the tracked files and `cleanup` restores them.
- Treat commands as literal; seeds and budgets are part of the expected numbers.

## Proof and skip reporting

- CLI proof is the command, stdout/stderr, exit code, and the files it wrote.
- Reproducibility proof is a git diff against committed records at the same SHA.
- A journey that fails because a precondition is missing (no torx, no pymoo, no unsga3-bend) is reported as failed with the reason, never silently skipped or swapped for pytest.

## Feature entry contract

Each feature file has an H1, a one-paragraph description, then exactly: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify.sh`, `Gotchas`.

## Features

- [THRML demos](./thrml-demos.md): codon Ising, Potts, and domain-wall weight sweeps (`--smoke` and default).
- [Torx PSWAP](./torx-pswap.md): optional Torx circuit stay/swap rate.
- [Oracle columns](./oracle-columns.md): classical U-NSGA-III and NumPy ExactEw on ZDT1/ZDT2/DTLZ2 with IGD.
- [Run records and deep](./run-records-and-deep.md): JSONL record CLI, schema validation, and the deep benchmark profile.
