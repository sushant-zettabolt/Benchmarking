#!/usr/bin/env bash
# Run a stage of the Turin tuning sweeps (scripts/gen_turin_tune_specs.py) on
# turin-xcovoid0021-pod-5, as user sohroy: both smoke specs, then -- only if every smoke row
# is ok -- the full llama.cpp sweep followed by the full vLLM sweep.
#
#   scripts/run_turin_tune.sh [all|full|resume] [turin-tune|turin-tune2]
#     all      smoke runs, then both full runs (default)
#     full     both full runs, no smoke
#     resume   continue interrupted full runs (--resume)
#   The second argument picks the stage: turin-tune (stage 1, default) or turin-tune2.
#
# Run it detached, it takes ~7-8 hours:
#   setsid nohup scripts/run_turin_tune.sh all turin-tune2 > sweep-tune2.log 2>&1 < /dev/null &
set -euo pipefail

STAGE="${1:-all}"
PREFIX="${2:-turin-tune}"
case "$STAGE" in all|full|resume) ;; *) echo "usage: $0 [all|full|resume] [turin-tune|turin-tune2]" >&2; exit 64;; esac
case "$PREFIX" in turin-tune|turin-tune2) ;; *) echo "unknown stage prefix: $PREFIX" >&2; exit 64;; esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv/bin/python"
LLMBENCH="$REPO/.venv/bin/llmbench"
EXPECTED_CPUS=160-191

step() { printf '\n=== %s  %s\n' "$(date -u +%FT%TZ)" "$*"; }

# Exit non-zero unless the run finished and every row is ok.
all_ok() {
    "$PY" - "$1" <<'EOF'
import json, sys
run = json.load(open(f"{sys.argv[1]}/run.json"))
by = run["counts"]["by_status"]
print(f"{sys.argv[1]}: status={run['status']} rows={by}")
sys.exit(0 if run["status"] == "finished" and by and set(by) == {"ok"} else 1)
EOF
}

step "checks"
cpus=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)
[ "$cpus" = "$EXPECTED_CPUS" ] || { echo "affinity is $cpus, expected $EXPECTED_CPUS" >&2; exit 1; }
for s in llamacpp vllm; do
    "$LLMBENCH" sweep plan --spec "sweep.$PREFIX-$s.yaml" | grep -E 'deployment\(s\)'
done

if [ "$STAGE" = all ]; then
    for s in llamacpp vllm; do
        step "smoke: $s"
        "$LLMBENCH" sweep run --spec "sweep.$PREFIX-$s-smoke.yaml" || true
        all_ok "out/$PREFIX-$s-smoke" || { echo "smoke $s has non-ok rows; not starting the full runs" >&2; exit 1; }
    done
fi

RESUME=()
[ "$STAGE" = resume ] && RESUME=(--resume)
for s in llamacpp vllm; do
    step "full: $s"
    "$LLMBENCH" sweep run --spec "sweep.$PREFIX-$s.yaml" "${RESUME[@]}" || true
    all_ok "out/$PREFIX-$s" || echo "full $s has non-ok rows; see its report.md" >&2
done
step "done"
