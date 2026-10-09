"""Turn a table of trial results into an answer: which configuration is best, and why.

Three questions get answered, because they have different answers and conflating them is how
benchmark reports mislead:

  `best_per_test`   For each individual workload, which deployment won? This is the honest
                    per-point answer. A config that wins pp1024 routinely loses tg64 -- on
                    this machine the llama.cpp/vLLM crossover sits just under pp64 -- so a
                    single global winner hides a real trade-off.

  `best_overall`    Across the whole workload mix, which deployment is best on average? Only
                    meaningful if you actually care about the mix as specified. Computed by
                    normalising each workload's objective values to that workload's best, then
                    averaging -- so a workload with large absolute numbers does not dominate a
                    small one purely by scale.

  `pareto`          Configurations that are not beaten on every axis at once. Useful when the
                    constraints turn out to be too tight, or too loose, to be interesting.

A constraint with no measurement fails. An unmeasured SLO is not a met SLO, and silently
treating a missing p99 as passing would promote exactly the configurations whose latency data
is missing because they fell over.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable

from .deploy import MEMORY_LOCAL_MIN_FRACTION
from .execute import CONTENTION_WARN_PCT, STATUS_OK, TrialResult
from .spec import Constraint, ObjectiveSpec


@dataclass
class ConstraintVerdict:
    metric: str
    described: str
    value: float | None
    satisfied: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Candidate:
    """One trial row scored against the objective."""

    trial_id: str
    deployment_id: str
    backend: str
    test: str
    src: str
    kind: str
    axes: dict[str, Any]
    value: float | None
    verdicts: list[ConstraintVerdict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    config_key: tuple = ()
    config_label: str = ""
    issues: list[str] = field(default_factory=list)   # see quality_issues()

    @property
    def feasible(self) -> bool:
        return self.value is not None and all(v.satisfied for v in self.verdicts)

    @property
    def violations(self) -> list[ConstraintVerdict]:
        return [v for v in self.verdicts if not v.satisfied]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id, "deployment_id": self.deployment_id,
            "backend": self.backend, "test": self.test, "src": self.src, "kind": self.kind,
            "config_label": self.config_label, "axes": self.axes, "value": self.value,
            "feasible": self.feasible,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "issues": list(self.issues),
            "metrics": self.metrics,
        }


# Axes that identify a *configuration* (a thing you could go and deploy), as opposed to the
# workload it was measured under. Ordering fixes the label's field order.
CONFIG_AXES = (
    "backend", "instances", "cores_per_instance", "threads_per_instance",
    "n_parallel", "n_ctx", "batch", "ubatch", "lb", "tool", "batch_size",
)


def config_key(axes: dict[str, Any]) -> tuple:
    return tuple((k, axes.get(k)) for k in CONFIG_AXES if axes.get(k) is not None)


def config_label(axes: dict[str, Any]) -> str:
    parts = []
    for k, v in config_key(axes):
        short = {
            "backend": "", "instances": "n", "cores_per_instance": "c",
            "threads_per_instance": "t", "n_parallel": "np", "n_ctx": "ctx",
            "batch": "b", "ubatch": "ub", "lb": "lb=", "tool": "", "batch_size": "bs",
        }.get(k, k + "=")
        parts.append(f"{short}{v}" if short and not short.endswith("=") else
                     (f"{v}" if not short else f"{short}{v}"))
    return " ".join(str(p) for p in parts)


def evaluate_constraints(metrics: dict[str, Any], constraints: Iterable[Constraint]) -> list[ConstraintVerdict]:
    out = []
    for c in constraints:
        value = metrics.get(c.metric)
        value = float(value) if isinstance(value, (int, float)) else None
        satisfied = c.satisfied_by(value)
        reason = ""
        if value is None:
            reason = f"{c.metric} was not measured for this trial; treated as not satisfied"
        elif not satisfied:
            reason = f"{c.metric}={value:.4g} violates {c.describe()}"
        out.append(ConstraintVerdict(
            metric=c.metric, described=c.describe(), value=value,
            satisfied=satisfied, reason=reason,
        ))
    return out


def quality_issues(r: TrialResult, objective: ObjectiveSpec) -> list[str]:
    """Reasons a successful row's number is not trustworthy enough to be *the answer*.

    Each of these was already detected and written onto the row as a warning, and the
    ranking used to ignore every one of them: a trial measured while another user's job held
    half its cores, or one that kept 1 request of 40, could win `best`. Read from provenance
    rather than from warning text, so re-rendering an old run applies the same test.
    """
    issues: list[str] = []
    prov = r.provenance or {}
    foreign = (prov.get("contention") or {}).get("foreign_pct")
    if isinstance(foreign, (int, float)) and foreign >= CONTENTION_WARN_PCT:
        issues.append(f"{foreign:.0f}% foreign CPU load on its cores")
    stray = (prov.get("placement") or {}).get("threads_on_other_cpus")
    if stray:
        issues.append(f"{stray} server thread(s) pinned outside the allocation")
    local = (prov.get("placement") or {}).get("memory_local_fraction")
    if isinstance(local, (int, float)) and local < MEMORY_LOCAL_MIN_FRACTION:
        issues.append(f"only {100 * local:.0f}% of server memory on its bound NUMA node(s)")
    n_records, n_errors = prov.get("n_records"), prov.get("n_errors")
    if n_records and n_errors:
        pct = 100.0 * n_errors / n_records
        if pct > objective.max_error_pct:
            issues.append(f"{n_errors} of {n_records} request(s) failed ({pct:.0f}% > "
                          f"max_error_pct {objective.max_error_pct:g}%)")
    return issues


def build_candidates(results: list[TrialResult], objective: ObjectiveSpec) -> list[Candidate]:
    """Score every successful row whose measurement path matches the objective.

    Offline rows carry src='native'. They are scored too, but stay in their own namespace via
    `kind`, because their numbers come from a different measurement boundary and must never
    be ranked head-to-head against online rows.
    """
    out: list[Candidate] = []
    for r in results:
        if r.status != STATUS_OK:
            continue
        if r.kind == "online" and r.src != objective.src:
            continue
        raw = r.metrics.get(objective.metric)
        value = float(raw) if isinstance(raw, (int, float)) else None
        out.append(Candidate(
            trial_id=r.trial_id, deployment_id=r.deployment_id, backend=r.backend,
            test=r.test, src=r.src, kind=r.kind, axes=dict(r.axes), value=value,
            verdicts=evaluate_constraints(r.metrics, objective.constraints),
            metrics=dict(r.metrics),
            config_key=config_key(r.axes), config_label=config_label(r.axes),
            issues=quality_issues(r, objective),
        ))
    return out


def _better(a: float, b: float, goal: str) -> bool:
    return a > b if goal == "max" else a < b


@dataclass
class ConfigScore:
    """A configuration aggregated across every workload it was measured on."""

    config_key: tuple
    config_label: str
    backend: str
    axes: dict[str, Any]
    n_tests: int
    n_feasible: int
    normalised_score: float
    per_test: dict[str, float | None] = field(default_factory=dict)
    fully_feasible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_label": self.config_label, "backend": self.backend, "axes": self.axes,
            "n_tests": self.n_tests, "n_feasible": self.n_feasible,
            "normalised_score": round(self.normalised_score, 6),
            "fully_feasible": self.fully_feasible, "per_test": self.per_test,
        }


@dataclass
class RankingReport:
    objective: str
    metric: str
    goal: str
    src: str
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    best_per_test: dict[str, Candidate] = field(default_factory=dict)
    best_overall: ConfigScore | None = None
    config_scores: list[ConfigScore] = field(default_factory=list)
    pareto: list[Candidate] = field(default_factory=list)
    excluded: list[Candidate] = field(default_factory=list)   # flagged; see quality_issues
    n_feasible: int = 0
    n_infeasible: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective, "metric": self.metric, "goal": self.goal,
            "src": self.src,
            "n_candidates": len(self.candidates),
            "n_feasible": self.n_feasible, "n_infeasible": self.n_infeasible,
            "best": self.best.to_dict() if self.best else None,
            "best_per_test": {k: v.to_dict() for k, v in self.best_per_test.items()},
            "best_overall": self.best_overall.to_dict() if self.best_overall else None,
            "config_scores": [c.to_dict() for c in self.config_scores],
            "pareto": [c.to_dict() for c in self.pareto],
            "excluded": [c.to_dict() for c in self.excluded],
            "notes": list(self.notes),
        }

    def headline(self) -> str:
        if self.best is None:
            if self.n_infeasible and not self.n_feasible:
                return (f"No configuration satisfied the constraints "
                        f"({self.n_infeasible} measured, 0 feasible).")
            return "No rankable results."
        return (f"Best {self.metric}: {self.best.value:.4g} "
                f"on {self.best.config_label} at {self.best.test}")


def rank(results: list[TrialResult], objective: ObjectiveSpec) -> RankingReport:
    candidates = build_candidates(results, objective)
    flagged = [c for c in candidates if c.issues]
    excluded: list[Candidate] = []
    if flagged and not objective.rank_flagged:
        excluded = flagged
        candidates = [c for c in candidates if not c.issues]
    report = RankingReport(
        objective=objective.describe(), metric=objective.metric,
        goal=objective.goal, src=objective.src, candidates=candidates, excluded=excluded,
    )
    if flagged:
        shown = "; ".join(f"{c.trial_id} ({', '.join(c.issues)})" for c in flagged[:10])
        more = f"; and {len(flagged) - 10} more" if len(flagged) > 10 else ""
        report.notes.append(
            f"{len(flagged)} row(s) were excluded from the ranking as untrustworthy: "
            f"{shown}{more}. They are still in the results table; set "
            f"`objective.rank_flagged: true` to rank them anyway"
            if excluded else
            f"{len(flagged)} flagged row(s) are ranked because objective.rank_flagged is set: "
            f"{shown}{more}"
        )
    if not candidates:
        report.notes.append(
            "every successful row was excluded as untrustworthy, so nothing was ranked; "
            "fix the cause (usually contention -- check provenance.contention) and re-run"
            if excluded else
            f"no successful rows with src={objective.src!r} carried a numeric "
            f"{objective.metric!r}; check the metric name against trials.jsonl"
        )
        return report

    online = [c for c in candidates if c.kind == "online"]
    scoreable = online or candidates
    if not online:
        report.notes.append(
            "ranking offline (native-tool) rows only -- these use each backend's own "
            "measurement boundary and are not comparable across backends"
        )

    feasible = [c for c in scoreable if c.feasible]
    report.n_feasible = len(feasible)
    report.n_infeasible = len(scoreable) - len(feasible)

    if not feasible:
        report.notes.append(
            "every measured configuration violated at least one constraint; "
            "reporting the pareto front and per-test winners over infeasible rows so the "
            "trade-off is still visible"
        )
    pool = feasible or scoreable

    # -- single best row --
    valued = [c for c in pool if c.value is not None]
    if not valued:
        # Every row exists but none carried the objective metric. Almost always a typo in
        # `objective.metric`; saying so beats returning an empty ranking with no explanation.
        available = sorted({
            k for c in scoreable for k, v in c.metrics.items() if isinstance(v, (int, float))
        })
        report.notes.append(
            f"none of the {len(scoreable)} scored row(s) carried a numeric value for "
            f"{objective.metric!r}, so nothing could be ranked. Available metrics on these "
            f"rows: {', '.join(available) if available else '(none)'}"
        )
        return report
    report.best = max(valued, key=lambda c: c.value if objective.goal == "max" else -c.value)

    # -- best per test --
    by_test: dict[str, list[Candidate]] = {}
    for c in valued:
        by_test.setdefault(c.test, []).append(c)
    for test, group in by_test.items():
        report.best_per_test[test] = max(
            group, key=lambda c: c.value if objective.goal == "max" else -c.value
        )

    # -- per-config aggregate over the workload mix --
    # Normalise within each test to that test's winner, so a pp1024 row measured in the
    # hundreds of t/s cannot outvote a tg64 row measured in the tens purely by magnitude.
    test_best: dict[str, float] = {
        test: best.value for test, best in report.best_per_test.items() if best.value is not None
    }
    by_config: dict[tuple, list[Candidate]] = {}
    for c in valued:
        by_config.setdefault(c.config_key, []).append(c)

    scores: list[ConfigScore] = []
    for key, group in by_config.items():
        norms, per_test = [], {}
        for c in group:
            ref = test_best.get(c.test)
            per_test[c.test] = c.value
            if not ref or c.value is None:
                continue
            norms.append(c.value / ref if objective.goal == "max" else ref / c.value)
        head = group[0]
        scores.append(ConfigScore(
            config_key=key, config_label=head.config_label, backend=head.backend,
            axes=head.axes, n_tests=len(group),
            n_feasible=sum(1 for c in group if c.feasible),
            normalised_score=(sum(norms) / len(norms)) if norms else 0.0,
            per_test=per_test,
            fully_feasible=all(c.feasible for c in group),
        ))
    scores.sort(key=lambda s: (-s.normalised_score, s.config_label))
    report.config_scores = scores
    if scores:
        report.best_overall = scores[0]
        covered = {s.n_tests for s in scores}
        if len(covered) > 1:
            report.notes.append(
                f"configurations were not all measured on the same number of workloads "
                f"(counts: {sorted(covered)}); `best_overall` averages over whatever each "
                f"config actually completed, so partial configs are not penalised for gaps"
            )

    report.pareto = _pareto_front(pool, objective)
    return report


def _pareto_front(candidates: list[Candidate], objective: ObjectiveSpec) -> list[Candidate]:
    """Front over the objective plus every constrained metric.

    Constrained metrics are treated as minimise-if-`max`-bounded and maximise-if-`min`-bounded,
    which is what the constraint direction already tells us the caller wants.
    """
    axes: list[tuple[str, str]] = [(objective.metric, objective.goal)]
    for c in objective.constraints:
        axes.append((c.metric, "min" if c.max is not None else "max"))

    def vector(cand: Candidate) -> list[float] | None:
        vec = []
        for metric, _ in axes:
            v = cand.metrics.get(metric) if metric != objective.metric else cand.value
            if not isinstance(v, (int, float)):
                return None
            vec.append(float(v))
        return vec

    scored = [(c, vector(c)) for c in candidates]
    scored = [(c, v) for c, v in scored if v is not None]

    def dominates(a: list[float], b: list[float]) -> bool:
        strictly_better = False
        for (metric, goal), av, bv in zip(axes, a, b):
            if _better(av, bv, goal):
                strictly_better = True
            elif av != bv:
                return False
        return strictly_better

    front = [c for c, v in scored if not any(dominates(ov, v) for oc, ov in scored if oc is not c)]
    front.sort(key=lambda c: c.value if objective.goal == "max" else -(c.value or 0), reverse=True)
    return front


__all__ = ["rank", "RankingReport", "Candidate", "ConfigScore", "ConstraintVerdict",
           "build_candidates", "evaluate_constraints", "config_label", "quality_issues"]
