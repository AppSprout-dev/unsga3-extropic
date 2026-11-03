#!/usr/bin/env bash
# Self-tests for verify.sh. Runs in a throwaway clone under $TMPDIR, so the
# real checkout's records, state, and evidence are never touched.
#
#   .cursor/skills/verify-unsga3-extropic/scripts/test_verify.sh
#
# Needs the venv from `verify.sh launch` (VERIFY_VENV, default <repo>/.venv)
# and network for the pinned unsga3-bend clone. About 40 s on CPU.
set -uo pipefail

SRC_REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export VERIFY_VENV="${VERIFY_VENV:-$SRC_REPO/.venv}"
[[ -x "$VERIFY_VENV/bin/python" ]] || { echo "no venv at $VERIFY_VENV; run verify.sh launch" >&2; exit 2; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
REPO="$TMP/repo"
git clone -q --no-hardlinks "$SRC_REPO" "$REPO"
# Test the working-tree harness, not just what is committed.
cp "$SRC_REPO/.cursor/skills/verify-unsga3-extropic/scripts/verify.sh" "$REPO/.cursor/skills/verify-unsga3-extropic/scripts/verify.sh"
V="$REPO/.cursor/skills/verify-unsga3-extropic/scripts/verify.sh"
unset VERIFY_EVIDENCE UNSGA3_BEND_ROOT
export VERIFY_BEND_DIR="$TMP/bend-pinned"
fails=0
ok()   { echo "PASS $*"; }
bad()  { echo "FAIL $*"; fails=$((fails+1)); }
LOG="$TMP/out.log"

# (2) records edited between drives: next drive and cleanup refuse, edit survives.
"$V" drive demos > "$LOG" 2>&1 && ok "demos drive (writes tracked runs.jsonl)" || { cat "$LOG"; bad "demos drive"; }
"$V" drive records > "$LOG" 2>&1 && ok "second drive accepted after the harness's own record writes" || { cat "$LOG"; bad "second drive after own writes"; }
echo '{"hand": "edit"}' >> "$REPO/benchmarks/records/runs.jsonl"
if "$V" drive records > "$LOG" 2>&1; then bad "drive after hand edit should refuse"
else grep -q "changed since this run's last snapshot" "$LOG" && ok "drive refuses after a between-drives hand edit" || { cat "$LOG"; bad "wrong refusal message"; }; fi
if "$V" cleanup > "$LOG" 2>&1; then bad "cleanup after hand edit should refuse"
else ok "cleanup refuses after a hand edit"; fi
tail -1 "$REPO/benchmarks/records/runs.jsonl" | grep -q '"hand"' && ok "hand edit preserved" || bad "hand edit was reverted"
# Abandon that run the documented way, restore the clone, keep going.
rm -rf "$REPO/.verify-evidence/.state"; git -C "$REPO" checkout -q -- benchmarks/records; rm -rf "$REPO/demos/_out"

# Untracked file appearing between drives is also caught.
"$V" drive records > "$LOG" 2>&1 || { cat "$LOG"; bad "records drive"; }
echo x > "$REPO/benchmarks/records/oracle_fronts/stray.csv"
if "$V" drive records > "$LOG" 2>&1; then bad "drive after untracked stray file should refuse"
else grep -q "changed since this run's last snapshot" "$LOG" && ok "drive refuses after an untracked file appears" || { cat "$LOG"; bad "wrong refusal message (stray file)"; }; fi
rm -f "$REPO/benchmarks/records/oracle_fronts/stray.csv"
"$V" cleanup > "$LOG" 2>&1 && ok "cleanup succeeds once records match the snapshot again" || { cat "$LOG"; bad "cleanup"; }
[[ -z "$(git -C "$REPO" status --porcelain -- benchmarks/records)" ]] && ok "records clean after cleanup" || bad "records dirty after cleanup"

# (3) evidence path with a single quote, double quote, and space.
EV="$TMP/ev'il \"dir\" x"
if VERIFY_EVIDENCE="$EV" "$V" drive records > "$LOG" 2>&1 && grep -q "validated 2 rows" "$EV/records_validate.log"; then
  ok "records_validate with a quoted EVID path"
else cat "$LOG" "$EV/records_validate.log" 2>/dev/null; bad "quoted EVID path"; fi
"$V" cleanup > "$LOG" 2>&1 || { cat "$LOG"; bad "cleanup after quoted EVID"; }
[[ -f "$EV/records_validate.log" ]] && ok "quoted evidence survives cleanup" || bad "quoted evidence removed"

# (1) Bend default: private clone at the pinned SHA; a shared /workspace/unsga3-bend is untouched.
SHARED=/workspace/unsga3-bend
before=""; [[ -d "$SHARED/.git" ]] && before="$(git -C "$SHARED" rev-parse HEAD) $(git -C "$SHARED" rev-parse --abbrev-ref HEAD) $(stat -c %Y "$SHARED/.git/HEAD")"
if "$V" drive oracle > "$LOG" 2>&1; then ok "oracle drive with default Bend"; else cat "$LOG"; bad "oracle drive"; fi
EVD="$(cat "$REPO/.verify-evidence/.state/current")"
pin="$(sed -n 2p "$EVD/oracle_bend.txt")"
grep -q "^bend=$VERIFY_BEND_DIR\$" "$EVD/oracle_bend.txt" && ok "oracle used the private clone" || { cat "$EVD/oracle_bend.txt"; bad "oracle used another Bend checkout"; }
[[ "$pin" == "d68ae5ea3764bc49998b5b79f8397b07b42d4958" ]] && ok "private clone at pinned SHA" || bad "clone at $pin"
if grep -q "$SHARED/" "$EVD"/oracle_*_s1.log; then bad "oracle logs reference $SHARED"; else ok "oracle logs never reference $SHARED"; fi
after=""; [[ -d "$SHARED/.git" ]] && after="$(git -C "$SHARED" rev-parse HEAD) $(git -C "$SHARED" rev-parse --abbrev-ref HEAD) $(stat -c %Y "$SHARED/.git/HEAD")"
[[ "$before" == "$after" ]] && ok "shared Bend tree unchanged ($before)" || bad "shared Bend tree changed: $before -> $after"
git -C "$VERIFY_BEND_DIR" -c advice.detachedHead=false checkout -q HEAD~1 2>/dev/null
"$V" drive oracle > "$LOG" 2>&1 && bad "oracle should refuse a clone off the pinned SHA" || { grep -q "not pinned" "$LOG" && ok "oracle refuses a clone off the pinned SHA" || { cat "$LOG"; bad "wrong off-pin message"; }; }
"$V" cleanup > /dev/null 2>&1

echo "---"; [[ $fails -eq 0 ]] && { echo "ALL PASS"; exit 0; } || { echo "$fails FAILED"; exit 1; }
