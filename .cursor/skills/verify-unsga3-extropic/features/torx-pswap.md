# Torx PSWAP

The optional Torx extra builds the documented `DiscretePCircuit([PSWAP([0, 1])])` quickstart with `p = 0.3`, samples it from `|10)`, and checks the stay and swap rates. It is not a search front and is not wired into `WeightSweepLoop`.

## Sub-features

- `torx-demo` the demo prints stay/swap rates and exits non-zero outside tolerance.
- `torx-pytest` the `torx` marker test checks the same rates at 20,000 samples.

## How to get to it (user POV)

- `pip install -e ".[dev,cpu,torx]"` (Python >= 3.11), then `python demos/run_torx_pswap.py`.
- `python -m pytest -q -m torx`.

## Driving it with verify.sh

Preconditions:

- Doctor shows `torx:` with a version (not `missing`).

- **Run it.** `$V drive torx`. Steps `torx_pytest` and `torx_demo` end `rc=0`.
- **Rates.** `torx_demo.log` contains `stay |10): 0.699, swap |01): 0.301 (p=0.3, n=20000, seed=0)` (any value within the demo's tolerance of 0.7/0.3 passes; the demo enforces it via its exit code).
- **No record.** `git status` shows no change to `benchmarks/records/runs.jsonl` from this journey.

## Gotchas

- On Python 3.10 `pip install -e ".[torx]"` cannot install `extro-torx`; the torx test then skips. A skip is not a pass for this journey.
- The deep 100,000-sample check lives in `run_deep.py` and writes the tracked `records/torx_pswap_deep.json`; use `--torx-out` to redirect it.
