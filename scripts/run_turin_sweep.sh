#!/usr/bin/env bash
# Run the Turin sweep (sweep.turin-32c-8b.yaml: Llama 3.1 8B + Qwen3.6-35B-A3B, 12 variants,
# 228 trials) end to end on turin-xcovoid0021-pod-5, as user sohroy.
#
# Deterministic: every step either passes or stops the script with a non-zero exit, and the
# full run starts only after a smoke run in which every variant started and measured.
#
#   scripts/run_turin_sweep.sh          # checks, smoke run, then the full run
#   scripts/run_turin_sweep.sh smoke    # checks and the smoke run only
#   scripts/run_turin_sweep.sh full     # checks and the full run, no smoke run
#   scripts/run_turin_sweep.sh resume   # continue an interrupted full run (--resume)
#
# Run it detached, it takes hours:
#   nohup scripts/run_turin_sweep.sh > sweep-turin.log 2>&1 < /dev/null &
set -euo pipefail

STAGE="${1:-all}"
case "$STAGE" in all|smoke|full|resume) ;; *) echo "usage: $0 [all|smoke|full|resume]" >&2; exit 64;; esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
LLMBENCH="$REPO/.venv/bin/llmbench"
SPEC=sweep.turin-32c-8b.yaml
SMOKE=sweep.turin-smoke.yaml
OUT=out/turin-32c-llama31-qwen36
SMOKE_OUT=out/turin-smoke
EXPECTED_CPUS=160-191
EXPECTED_PLAN="12 deployment(s), 228 online trial(s)"
EXPECTED_SMOKE_ROWS=24          # client rows: pp16 + tg16 for each of 12 variants
MIN_TPS=1.0                     # below this a variant is stalled, not slow (CLAUDE.md open issue)
MAX_IDLE_BUSY_PCT=15            # the ranking refuses rows with >=15% foreign load anyway

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
fail() { log "STOP: $*"; exit 1; }

log "stage=$STAGE repo=$REPO commit=$(git -C "$REPO" describe --always --dirty 2>/dev/null || echo '?')"

# ---- 1. the right pod ---------------------------------------------------------------------
allowed=$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status)
[ "$allowed" = "$EXPECTED_CPUS" ] || fail "this process may use cpus $allowed, the spec needs $EXPECTED_CPUS (wrong pod?)"
log "cpuset ok: $allowed"

# ---- 2. the code works ---------------------------------------------------------------------
[ -x "$PY" ] && [ -x "$LLMBENCH" ] || fail "no venv at $REPO/.venv (python -m venv .venv && .venv/bin/pip install -e '.[test]')"
"$PY" -m pytest -q -p no:cacheprovider || fail "tests failed"
log "tests ok"

# ---- 3. everything the spec points at exists ----------------------------------------------
"$PY" - "$SPEC" <<'EOF' || fail "missing files (above)"
import sys
from pathlib import Path
from llmbench.suite.spec import SuiteSpec
spec = SuiteSpec.from_yaml(sys.argv[1])
missing = []
for name in spec.deployment.backend:
    b = spec.backends[name]
    paths = [b.model, b.server_bin] + [p for p in b.env.get("LD_PRELOAD", "").split(":") if p]
    missing += [f"{name}: {p}" for p in paths if not Path(p).expanduser().exists()]
print("\n".join(missing) or f"all files present for {len(spec.deployment.backend)} variants")
sys.exit(1 if missing else 0)
EOF

# ---- 4. nothing of ours still running, and nobody else on the cores ------------------------
for d in "$SMOKE_OUT" "$OUT"; do
  [ -d "$d" ] || continue
  "$LLMBENCH" sweep cleanup "$d" --dry-run || fail "servers from an earlier run in $d are still alive: $LLMBENCH sweep cleanup $d"
done
busy=$("$PY" - "$EXPECTED_CPUS" <<'EOF'
import sys
from llmbench.suite.contention import snapshot
lo, hi = map(int, sys.argv[1].split("-"))
print(round(snapshot(range(lo, hi + 1), window_s=5.0).busy_pct, 1))
EOF
)
log "cores $EXPECTED_CPUS busy before start: ${busy}%"
"$PY" -c "import sys; sys.exit(0 if $busy < $MAX_IDLE_BUSY_PCT else 1)" \
  || fail "cores $EXPECTED_CPUS are ${busy}% busy before anything of ours runs; someone else is using them"

# ---- 5. the plan is the one we mean --------------------------------------------------------
plan=$("$LLMBENCH" sweep plan --spec "$SPEC")
echo "$plan"
grep -qF "$EXPECTED_PLAN" <<<"$plan" || fail "plan is not '$EXPECTED_PLAN'"
if grep -q "cpus=" <<<"$plan" && grep "cpus=" <<<"$plan" | grep -qv "cpus=$EXPECTED_CPUS .*membind=\[5\]"; then
  fail "a deployment is not on cpus=$EXPECTED_CPUS membind=[5]"
fi
log "plan ok"

# ---- 6. smoke run: every variant starts and measures ---------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = smoke ]; then
  log "smoke run: $SMOKE -> $SMOKE_OUT"
  rc=0; "$LLMBENCH" sweep run --spec "$SMOKE" -q || rc=$?
  cat "$SMOKE_OUT/report.md" 2>/dev/null | sed -n '/## Online results/,/## /p' || true
  [ "$rc" -eq 0 ] || fail "smoke run exited $rc (not every row ok); see $SMOKE_OUT/report.md and $SMOKE_OUT/logs/"
  "$PY" - "$SMOKE_OUT" "$EXPECTED_SMOKE_ROWS" "$MIN_TPS" <<'EOF' || fail "smoke results failed the checks above"
import sys
from llmbench.suite.execute import read_trials
out, want, min_tps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rows, _ = read_trials(f"{out}/trials.jsonl")
client = [r for r in rows if r.kind == "online" and r.src == "client"]
bad = [f"{r.backend} {r.test}: status={r.status} {r.error or ''}" for r in client if r.status != "ok"]
slow = [f"{r.backend} {r.test}: {r.metrics.get('tps_mean', 0):.2f} t/s" for r in client
        if r.status == "ok" and (r.metrics.get("tps_mean") or 0) < min_tps]
problems = bad + [f"stalled (< {min_tps} t/s): {s}" for s in slow]
if len(client) != want:
    problems.append(f"expected {want} client rows, found {len(client)}")
print("\n".join(problems) or f"smoke ok: {len(client)} client rows, all ok, all >= {min_tps} t/s")
sys.exit(1 if problems else 0)
EOF
  log "smoke ok"
fi
[ "$STAGE" = smoke ] && { log "done (smoke only)"; exit 0; }

# ---- 7. the full run -----------------------------------------------------------------------
resume=(); [ "$STAGE" = resume ] && resume=(--resume)
log "full run: $SPEC -> $OUT ${resume[*]:-}"
rc=0; "$LLMBENCH" sweep run --spec "$SPEC" "${resume[@]}" || rc=$?
case "$rc" in
  0)   log "full run finished, every row ok" ;;
  2)   log "full run finished with some error/capacity rows; see $OUT/report.md" ;;
  130) log "full run interrupted; continue with: $0 resume" ;;
  *)   log "full run failed (exit $rc); see $OUT/run.json, then: $LLMBENCH sweep cleanup $OUT && $0 resume" ;;
esac
log "results: $OUT/report.md report.html report.csv report_reps.csv best.json cores/"
exit "$rc"
