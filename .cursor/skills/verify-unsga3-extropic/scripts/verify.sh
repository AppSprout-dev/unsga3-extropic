#!/usr/bin/env bash
# Verification harness for unsga3-extropic. See ../SKILL.md.
#
#   verify.sh launch            # venv + pip install -e ".[dev,cpu,torx]" + pymoo
#   verify.sh doctor            # read-only: is this checkout worth driving?
#   verify.sh drive <journey>   # ci | demos | torx | classical | oracle | records | deep-codon
#   verify.sh cleanup           # restore tracked records the drives touched; keep evidence
#
# Env:
#   VERIFY_VENV      venv path            (default: <repo>/.venv)
#   VERIFY_EVIDENCE  evidence directory   (default: <repo>/.verify-evidence/<UTC stamp>)
#   UNSGA3_BEND_ROOT use this unsga3-bend checkout as-is (read only; never fetched or switched)
#   VERIFY_BEND_SHA  pinned unsga3-bend commit for the private clone (default below)
#   VERIFY_BEND_DIR  where that private clone lives (default: /tmp/unsga3-bend-<sha12>)
set -uo pipefail

REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
VENV="${VERIFY_VENV:-$REPO/.venv}"
STATE="$REPO/.verify-evidence/.state"
if [[ -z "${VERIFY_EVIDENCE:-}" ]]; then
  if [[ -f "$STATE/current" ]]; then VERIFY_EVIDENCE="$(cat "$STATE/current")"
  else VERIFY_EVIDENCE="$REPO/.verify-evidence/$(date -u +%Y%m%dT%H%M%SZ)"; fi
fi
EVID="$VERIFY_EVIDENCE"
export JAX_PLATFORMS=cpu

# Tracked files the CLIs write by default. Drives redirect what they can;
# these are the ones they cannot (see features/*.md Gotchas).
TOUCHED=(benchmarks/records/runs.jsonl benchmarks/records/oracle_fronts benchmarks/records/classical_unsga3)

# unsga3-bend commit whose ab/igd_vs_pymoo.py reproduced the committed oracle IGDs.
BEND_URL="https://github.com/AppSprout-dev/unsga3-bend.git"
BEND_SHA="${VERIFY_BEND_SHA:-d68ae5ea3764bc49998b5b79f8397b07b42d4958}"
BEND_DIR="${VERIFY_BEND_DIR:-/tmp/unsga3-bend-${BEND_SHA:0:12}}"

die() { echo "verify: $*" >&2; exit 1; }
py() { "$VENV/bin/python" "$@"; }
need_venv() { [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV; run: $0 launch"; }

# Run one command, tee its transcript to $EVID/<name>.log, record rc.
step() {
  local name="$1"; shift
  mkdir -p "$EVID"
  { echo "\$ $*"; echo "# cwd=$REPO date=$(date -u +%FT%TZ) sha=$(git -C "$REPO" rev-parse --short HEAD)"; } > "$EVID/$name.log"
  (cd "$REPO" && "$@") >> "$EVID/$name.log" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$EVID/$name.log"
  printf '%-28s rc=%s\n' "$name" "$rc" | tee -a "$EVID/SUMMARY.txt"
  return $rc
}

# Fingerprint of the tracked record paths: porcelain status, the binary diff
# against HEAD, and the bytes of any untracked files under them.
records_fingerprint() {
  {
    git -C "$REPO" status --porcelain --untracked-files=all -- "${TOUCHED[@]}"
    git -C "$REPO" diff --binary HEAD -- "${TOUCHED[@]}"
    git -C "$REPO" ls-files -z --others --exclude-standard -- "${TOUCHED[@]}" \
      | (cd "$REPO" && xargs -0 -r sha256sum)
  } | sha256sum | cut -d' ' -f1
}

snapshot_records() { records_fingerprint > "$STATE/records.sha256"; }

# Refuse unless the records match the snapshot taken when this run started
# (clean HEAD) or at the end of its previous drive. Any other change is
# someone's edit, and cleanup's git checkout would destroy it.
check_snapshot() {
  local want have
  want="$(cat "$STATE/records.sha256")"; have="$(records_fingerprint)"
  if [[ "$want" != "$have" ]]; then
    git -C "$REPO" status --porcelain --untracked-files=all -- "${TOUCHED[@]}" >&2
    die "tracked benchmark records changed since this run's last snapshot (hand edit or another run?); refusing. Inspect them, then move them aside or rm -rf '$STATE' to abandon this run."
  fi
}

guard_clean() {
  mkdir -p "$STATE"
  if [[ ! -f "$STATE/records.sha256" ]]; then
    # Run start: records must be clean so cleanup can restore them with git.
    local dirty
    dirty="$(git -C "$REPO" status --porcelain --untracked-files=all -- "${TOUCHED[@]}")"
    if [[ -n "$dirty" ]]; then
      echo "$dirty" >&2
      die "tracked benchmark records are already modified; commit or move them before driving"
    fi
    snapshot_records
    echo "$EVID" > "$STATE/current"
  else
    check_snapshot
  fi
  mkdir -p "$EVID"
}

# Resolve the unsga3-bend checkout for the oracle journey. An explicit
# UNSGA3_BEND_ROOT is used read-only. Otherwise use a private clone at the
# pinned SHA, created atomically; an existing one at another SHA is refused.
ensure_bend() {
  if [[ -n "${UNSGA3_BEND_ROOT:-}" ]]; then
    [[ -f "$UNSGA3_BEND_ROOT/ab/igd_vs_pymoo.py" ]] || die "UNSGA3_BEND_ROOT=$UNSGA3_BEND_ROOT has no ab/igd_vs_pymoo.py"
    echo "$UNSGA3_BEND_ROOT"; return 0
  fi
  if [[ ! -d "$BEND_DIR/.git" ]]; then
    local tmp; tmp="$(mktemp -d "${BEND_DIR}.tmp.XXXXXX")" || die "mktemp failed"
    git clone -q "$BEND_URL" "$tmp" >&2 && git -C "$tmp" -c advice.detachedHead=false checkout -q "$BEND_SHA" >&2 \
      || { rm -rf "$tmp"; die "could not clone $BEND_URL at $BEND_SHA"; }
    mv -T "$tmp" "$BEND_DIR" 2>/dev/null || rm -rf "$tmp"   # lost a race: keep the winner
  fi
  local head; head="$(git -C "$BEND_DIR" rev-parse HEAD 2>/dev/null)"
  [[ "$head" == "$BEND_SHA" ]] || die "$BEND_DIR is at ${head:-?}, not pinned $BEND_SHA; remove it or set VERIFY_BEND_DIR"
  echo "$BEND_DIR"
}

cmd_launch() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    local pybin; pybin="$(command -v python3.12 || command -v python3.11 || command -v python3)"
    "$pybin" -m venv "$VENV" || die "venv creation failed"
  fi
  py -m pip install -q --upgrade pip || die "pip upgrade failed"
  (cd "$REPO" && py -m pip install -q -e ".[dev,cpu,torx]") || die "install failed"
  # pymoo is only needed by unsga3-bend's ab/igd_vs_pymoo.py (oracle journey).
  py -m pip install -q pymoo || echo "verify: pymoo install failed; oracle journey will fail" >&2
  echo "READY venv=$VENV"
}

cmd_doctor() {
  local ok=0
  echo "repo:     $REPO"
  echo "head:     $(git -C "$REPO" rev-parse --short HEAD) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
  if [[ ! -x "$VENV/bin/python" ]]; then echo "venv:     MISSING ($VENV)"; return 1; fi
  py - "$REPO" <<'PY' || ok=1
import importlib, sys, pathlib
repo = pathlib.Path(sys.argv[1])
print("python:  ", sys.version.split()[0])
import unsga3_extropic as u
src = pathlib.Path(u.__file__).resolve()
print("package: ", u.__version__, src)
if repo.resolve() not in src.parents:
    print("FAIL: unsga3_extropic is imported from another checkout"); sys.exit(1)
for mod in ("thrml", "jax", "numpy", "pytest"):
    m = importlib.import_module(mod); print(f"{mod+':':9} {getattr(m, '__version__', '?')}")
for mod, why in (("torx", "torx journey"), ("pymoo", "oracle journey")):
    try:
        m = importlib.import_module(mod); print(f"{mod+':':9} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{mod+':':9} missing ({why} will fail): {exc}")
import jax
print("jax dev: ", jax.devices())
PY
  if [[ -n "${UNSGA3_BEND_ROOT:-}" ]]; then
    if [[ -f "$UNSGA3_BEND_ROOT/ab/igd_vs_pymoo.py" ]]; then echo "bend:     $UNSGA3_BEND_ROOT (UNSGA3_BEND_ROOT, used read-only)"
    else echo "bend:     UNSGA3_BEND_ROOT=$UNSGA3_BEND_ROOT has no ab/igd_vs_pymoo.py (oracle journey will fail)"; fi
  elif [[ "$(git -C "$BEND_DIR" rev-parse HEAD 2>/dev/null)" == "$BEND_SHA" ]]; then
    echo "bend:     $BEND_DIR at pinned ${BEND_SHA:0:12}"
  else
    echo "bend:     no private clone yet; drive oracle will clone $BEND_URL at ${BEND_SHA:0:12} into $BEND_DIR"
  fi
  if [[ -f "$STATE/records.sha256" ]]; then
    if [[ "$(records_fingerprint)" == "$(cat "$STATE/records.sha256")" ]]; then echo "snapshot: run in progress, records match last snapshot"
    else echo "snapshot: run in progress, records DIFFER from last snapshot (next drive and cleanup will refuse)"; fi
  fi
  local dirty; dirty="$(git -C "$REPO" status --porcelain -- "${TOUCHED[@]}")"
  if [[ -n "$dirty" ]]; then echo "records:  MODIFIED"; echo "$dirty"; else echo "records:  clean"; fi
  [[ $ok -eq 0 ]] && echo "DOCTOR OK" || echo "DOCTOR FAIL"
  return $ok
}

drive_ci() {
  # Mirrors .github/workflows/ci.yml jobs `test` and `torx`, step for step.
  local rc=0
  step ci_pytest_numpy  py -m pytest -q -m 'not thrml and not torx and not deep' || rc=1
  step ci_pytest_core   py -m pytest -q -m 'not torx and not deep'               || rc=1
  drive_demos || rc=1
  drive_torx  || rc=1
  return $rc
}

drive_demos() {
  local rc=0 d
  for d in codon_thrml potts_thrml domain_wall; do
    step "demo_${d}_smoke" py "demos/run_${d}.py" --smoke || rc=1
  done
  # Side-effect proof: each demo appends exactly one row to the tracked log.
  git -C "$REPO" diff --numstat -- benchmarks/records/runs.jsonl > "$EVID/demos_runs_jsonl.numstat"
  cp -r "$REPO/demos/_out" "$EVID/demos_out" 2>/dev/null || true
  return $rc
}

drive_torx() {
  local rc=0
  step torx_pytest py -m pytest -q -m torx || rc=1
  step torx_demo   py demos/run_torx_pswap.py || rc=1
  return $rc
}

drive_classical() {
  # Reproduce committed seed-1 fronts; a byte diff on the tracked CSV/igd is a regression.
  local rc=0 p
  for p in zdt1 zdt2 dtlz2; do
    step "classical_${p}_s1" py benchmarks/run_classical_unsga3.py --problem "$p" --seed 1 --report "$EVID/classical_${p}_s1.md" || rc=1
  done
  git -C "$REPO" diff --stat -- benchmarks/records/classical_unsga3 > "$EVID/classical_diff.txt"
  if [[ -s "$EVID/classical_diff.txt" ]]; then echo "classical fronts DIFFER from committed (see classical_diff.txt)" | tee -a "$EVID/SUMMARY.txt"; rc=1
  else echo "classical fronts byte-identical to committed" | tee -a "$EVID/SUMMARY.txt"; fi
  return $rc
}

drive_oracle() {
  local rc=0 p bend
  bend="$(ensure_bend)" || return 1
  echo "bend=$bend" > "$EVID/oracle_bend.txt"
  git -C "$bend" rev-parse HEAD >> "$EVID/oracle_bend.txt"
  for p in zdt1 zdt2 dtlz2; do
    step "oracle_${p}_s1" env UNSGA3_BEND_ROOT="$bend" "$VENV/bin/python" benchmarks/run_oracle_continuous.py --problems "$p" --seeds 1 --log "$EVID/oracle_runs.jsonl" --report "$EVID/oracle_${p}_s1.md" || rc=1
  done
  # Compare only the igd= value and the CSV; the sidecar's front= line is an absolute path.
  git -C "$REPO" diff -U0 -- benchmarks/records/oracle_fronts > "$EVID/oracle_diff.txt"
  if grep -E '^[-+](igd=|[0-9])' "$EVID/oracle_diff.txt" >/dev/null; then
    echo "oracle IGD or front DIFFERS from committed (see oracle_diff.txt)" | tee -a "$EVID/SUMMARY.txt"; rc=1
  else echo "oracle seed-1 IGD and fronts identical to committed (front= path line ignored)" | tee -a "$EVID/SUMMARY.txt"; fi
  return $rc
}

drive_records() {
  local rc=0
  step records_stub  py benchmarks/append_run.py --stub  --out "$EVID/records/runs.jsonl" || rc=1
  step records_smoke py benchmarks/append_run.py --smoke --out "$EVID/records/runs.jsonl" || rc=1
  # The path goes in argv, never into the Python source.
  step records_validate py -c '
import sys; sys.path.insert(0, "src")
from unsga3_extropic.results import read_run_records, validate_run_record
rows = read_run_records(sys.argv[1])
for r in rows: validate_run_record(r)
assert [r["backend"] for r in rows] == ["none", "thrml_potts"], rows
print("validated", len(rows), "rows")' "$EVID/records/runs.jsonl" || rc=1
  return $rc
}

drive_deep_codon() {
  # Slow (~3 min CPU). Same matrix as deep.yml for codon + exact_ew, redirected out of the repo.
  step deep_codon_exact_ew py benchmarks/run_deep.py --only codon --only exact_ew --skip-torx \
    --out "$EVID/deep/runs.jsonl" --summary "$EVID/deep/DEEP_RESULTS.md"
}

cmd_drive() {
  need_venv; guard_clean
  local j="${1:-}"; [[ -n "$j" ]] || die "usage: $0 drive <ci|demos|torx|classical|oracle|records|deep-codon>"
  case "$j" in
    ci) drive_ci ;; demos) drive_demos ;; torx) drive_torx ;; classical) drive_classical ;;
    oracle) drive_oracle ;; records) drive_records ;; deep-codon) drive_deep_codon ;;
    *) die "unknown journey $j" ;;
  esac
  local rc=$?
  snapshot_records   # this drive's own writes become the new baseline
  echo "evidence: $EVID"
  return $rc
}

cmd_cleanup() {
  if [[ -f "$STATE/records.sha256" ]]; then
    check_snapshot
    git -C "$REPO" checkout -- "${TOUCHED[@]}"
    git -C "$REPO" clean -fdq -- benchmarks/records/oracle_fronts benchmarks/records/classical_unsga3
    rm -rf "$REPO/demos/_out"
    echo "restored tracked records: ${TOUCHED[*]}"
  else
    echo "no drive state; nothing restored"
  fi
  rm -rf "$STATE"
  echo "evidence kept: $EVID"
  [[ -d "$EVID" ]] && ls "$EVID"
}

case "${1:-}" in
  launch) cmd_launch ;; doctor) cmd_doctor ;; drive) shift; cmd_drive "$@" ;; cleanup) cmd_cleanup ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
