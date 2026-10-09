#!/usr/bin/env python3
"""Progress of sweep runs, read from their out/ directories. Standard library only, so it runs
on the workstation (same NFS checkout) as well as in the pod.

    python3 scripts/turin_status.py                 # every out/turin-tune* run (both stages)
    python3 scripts/turin_status.py out/<run> ...   # specific runs

run.json only gets its row counts when a run ends, so live progress comes from events.jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def status(out: Path) -> None:
    run = json.loads((out / "run.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    n_dep, n_trials = plan["n_deployments"], plan["n_online_trials"]

    trials_done: set[str] = set()
    not_ok: list[str] = []
    deps_done = 0
    current = test = ""
    first_ts = last_ts = None
    for line in (out / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        ts = datetime.fromisoformat(e["ts_utc"]).timestamp()
        first_ts = first_ts or ts
        last_ts = ts
        kind = e["kind"]
        if kind == "run_start":            # a resumed run starts counting again
            deps_done, trials_done, not_ok = 0, set(), []
        elif kind == "deployment_start":
            current, test = e["axes"]["backend"], "(starting server)"
        elif kind == "deployment_end":
            deps_done += 1
        elif kind == "trial_start":
            test = e.get("test", "")
        elif kind == "trial" and e.get("src", "client") == "client":
            trials_done.add(e["trial_id"])
            if e.get("status") != "ok":
                not_ok.append(f"{e['trial_id']} {e.get('status')}: {e.get('error') or ''}"[:160])

    state = run["status"]
    print(f"== {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    print(f"   status   {state}  (pid {run.get('pid')}, attempt {run.get('attempt')})"
          + (f"  error: {run['error']}" if run.get("error") else ""))
    done = len(trials_done)
    print(f"   progress {done}/{n_trials} trials, {deps_done}/{n_dep} deployments")
    if state == "running" and first_ts:
        elapsed = last_ts - first_ts
        idle = time.time() - last_ts
        eta = elapsed / done * (n_trials - done) if done else 0
        print(f"   now      {current} / {test}")
        print(f"   elapsed  {_hms(elapsed)}, last event {_hms(idle)} ago"
              + (f", eta ~{_hms(eta)}" if done else ""))
        if idle > 1800:
            print("   !! no event for 30+ min: check the process is alive (see docs/turin-status.md)")
    print(f"   not ok   {len(not_ok)}" + "".join(f"\n            {s}" for s in not_ok[:10]))
    counts = run.get("counts", {}).get("by_status")
    if counts:
        print(f"   rows     {counts}")
    if (out / "report.md").exists() and state != "running":
        print(f"   report   {out / 'report.md'}")


def main(argv: list[str]) -> int:
    outs = [Path(a).resolve() for a in argv] or sorted((REPO / "out").glob("turin-tune*"))
    outs = [o for o in outs if (o / "run.json").exists()]
    if not outs:
        print("no runs found")
        return 1
    for out in outs:
        status(out)
    logs = sorted(REPO.glob("sweep-tune*.log"), key=lambda p: p.stat().st_mtime)
    if not argv and logs:
        steps = [l for l in logs[-1].read_text().splitlines() if l.startswith("=== 20")]
        print(f"\nrun_turin_tune.sh ({logs[-1].name}): last step: "
              f"{steps[-1][4:] if steps else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
