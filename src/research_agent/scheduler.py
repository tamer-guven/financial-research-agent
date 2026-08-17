"""Sequential dependency-aware DAG scheduler with branch-local substitution."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from research_agent.calculations import calculate_comparison
from research_agent.clock import Clock
from research_agent.executor import SourceExecutor
from research_agent.models import (
    BLSParameters,
    EvidenceItem,
    FREDParameters,
    InvestigationPlan,
    ReportPeriodParameters,
    RecoveryAction,
    RecoveryDecision,
    ResearchTask,
    StepStatus,
    validate_task_graph,
)
from research_agent.state import SQLiteRepository


@dataclass
class EngineResult:
    run_id: str
    evidence: list[EvidenceItem]
    recovery_decisions: list[RecoveryDecision]
    missing_evidence: list[str]


class SchedulerStalledError(RuntimeError):
    pass


class SequentialScheduler:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        source_executor: SourceExecutor,
        clock: Clock,
    ) -> None:
        self.repository = repository
        self.source_executor = source_executor
        self.clock = clock

    def _fallback_task(self, failed: ResearchTask) -> ResearchTask:
        if failed.source != "fred" or failed.fallback_source != "bls":
            raise ValueError(f"no dynamic fallback task configured for {failed.source}")
        if not isinstance(failed.parameters, FREDParameters):
            raise ValueError("FRED fallback requires typed FRED parameters")
        dates = list(failed.parameters.dates)
        if len(dates) != 2:
            raise ValueError("FRED fallback requires exactly two requested dates")
        years = sorted({str(date)[:4] for date in dates})
        if len(years) != 2:
            raise ValueError("FRED fallback requires two distinct years")
        return ResearchTask(
            id="bls_cpi",
            objective=(
                f"Retrieve September {years[0]} and September {years[1]} CPI from the "
                "preconfigured unregistered BLS V1 fallback"
            ),
            source="bls",
            parameters=BLSParameters(startyear=years[0], endyear=years[1]),
            dependencies=list(failed.dependencies),
            importance=failed.importance,
            kind="source",
        )

    @staticmethod
    def _select(evidence: list[EvidenceItem], metric: str, period: str) -> EvidenceItem:
        matches = [item for item in evidence if item.metric == metric and item.period == period]
        if not matches:
            raise KeyError(f"missing evidence: {metric}/{period}")
        return matches[-1]

    @staticmethod
    def _priority(task: ResearchTask) -> tuple[int, int, str]:
        kind_order = {"source": 0, "calculation": 1, "report": 2}
        source_order = {"sec": 0, "world_bank": 1, "fred": 2, "bls": 3}
        return (kind_order[task.kind], source_order.get(task.source, 9), task.id)

    def _runtime_tasks(self, run_id: str, plan: InvestigationPlan) -> dict[str, ResearchTask]:
        tasks = {task.id: task for task in plan.tasks}
        for row in self.repository.list_steps(run_id):
            task = ResearchTask.model_validate_json(row["task_json"])
            tasks[task.id] = task
        validate_task_graph(list(tasks.values()))
        return tasks

    @staticmethod
    def _substitutions(rows: dict[str, dict[str, object]]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for step_id, row in rows.items():
            replaced = row.get("substitute_for_step_id")
            if replaced:
                result.setdefault(str(replaced), []).append(step_id)
        return result

    @staticmethod
    def _dependency_ready(
        dependency: str,
        rows: dict[str, dict[str, object]],
        substitutions: dict[str, list[str]],
    ) -> bool:
        if rows[dependency]["status"] == StepStatus.VALIDATED.value:
            return True
        return any(
            rows[substitute]["status"] == StepStatus.VALIDATED.value
            for substitute in substitutions.get(dependency, [])
        )

    def _ensure_fallbacks(
        self,
        run_id: str,
        tasks: dict[str, ResearchTask],
        rows: dict[str, dict[str, object]],
    ) -> bool:
        added = False
        substitutes = self._substitutions(rows)
        for step_id, row in rows.items():
            task = tasks[step_id]
            if (
                row["status"] == StepStatus.FAILED.value
                and row.get("recovery_action") == RecoveryAction.SUBSTITUTE.value
                and task.fallback_source
                and not substitutes.get(step_id)
            ):
                fallback = self._fallback_task(task)
                self.repository.add_step(run_id, fallback, self.clock.now(), substitute_for=step_id)
                tasks[fallback.id] = fallback
                added = True
        return added

    def _calculate(self, run_id: str, task: ResearchTask) -> None:
        evidence = self.repository.list_evidence(run_id)
        if not isinstance(task.parameters, ReportPeriodParameters):
            raise ValueError("calculation task requires typed report periods")
        baseline = task.parameters.baseline_fiscal_year
        target = task.parameters.target_fiscal_year
        month = task.parameters.inflation_month
        artifact_id = f"calc_{uuid4().hex}"
        derived = calculate_comparison(
            run_id=run_id,
            task_id=task.id,
            baseline_revenue=self._select(evidence, "apple_revenue", f"FY{baseline}"),
            target_revenue=self._select(evidence, "apple_revenue", f"FY{target}"),
            baseline_cpi=self._select(evidence, "us_cpi_u_nsa", f"{baseline}-{month:02d}"),
            target_cpi=self._select(evidence, "us_cpi_u_nsa", f"{target}-{month:02d}"),
            baseline_fiscal_year=baseline,
            target_fiscal_year=target,
            inflation_month=month,
            calculated_at=self.clock.now(),
            artifact_id=artifact_id,
        )
        self.repository.persist_derived_evidence(
            run_id,
            task.id,
            artifact_id,
            derived,
            self.clock.now(),
        )

    def _recorded_recoveries(self, run_id: str) -> list[RecoveryDecision]:
        return [
            RecoveryDecision.model_validate(event["event_payload"])
            for event in self.repository.list_events(run_id)
            if event["event_type"] == "RECOVERY_DECISION"
        ]

    async def run(
        self,
        plan: InvestigationPlan | None = None,
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> EngineResult:
        if resume:
            if not run_id:
                raise ValueError("resume requires run_id")
            plan = self.repository.get_plan(run_id)
        else:
            if plan is None:
                raise ValueError("new run requires a plan")
            validate_task_graph(plan.tasks)
            run_id = run_id or f"run_{uuid4().hex[:12]}"
            self.repository.create_run(run_id, plan, self.clock.now())
            for task in plan.tasks:
                self.repository.add_step(run_id, task, self.clock.now())

        assert plan is not None and run_id is not None
        tasks = self._runtime_tasks(run_id, plan)
        missing: list[str] = []
        processed: set[str] = set()
        unavailable_statuses = {StepStatus.FAILED.value, StepStatus.SKIPPED.value}

        while True:
            rows = {row["step_id"]: row for row in self.repository.list_steps(run_id)}
            if self._ensure_fallbacks(run_id, tasks, rows):
                validate_task_graph(list(tasks.values()))
                continue
            rows = {row["step_id"]: row for row in self.repository.list_steps(run_id)}
            substitutions = self._substitutions(rows)
            actionable = [
                task
                for task in tasks.values()
                if task.kind != "report"
                and task.id not in processed
                and rows[task.id]["status"] not in unavailable_statuses
                and all(
                    self._dependency_ready(dependency, rows, substitutions)
                    for dependency in task.dependencies
                )
            ]

            if actionable:
                task = min(actionable, key=self._priority)
                row = rows[task.id]
                if task.kind == "calculation":
                    if row["status"] == StepStatus.VALIDATED.value:
                        self.repository.record_event(
                            run_id,
                            task.id,
                            "CHECKPOINT_RETAINED",
                            {"network_calls_added": 0},
                            self.clock.now(),
                        )
                    else:
                        try:
                            self._calculate(run_id, task)
                        except (KeyError, ValueError) as exc:
                            reason = str(exc)
                            missing.append(reason)
                            self.repository.mark_skipped(run_id, task.id, reason, self.clock.now())
                    processed.add(task.id)
                    continue

                outcome = await self.source_executor.execute(run_id, task)
                processed.add(task.id)
                if outcome.decision and outcome.decision.selected_action in {
                    RecoveryAction.STOP,
                    RecoveryAction.PROCEED_PARTIAL,
                }:
                    missing.append(task.objective)
                continue

            remaining = [
                task
                for task in tasks.values()
                if task.kind != "report"
                and task.id not in processed
                and rows[task.id]["status"] not in unavailable_statuses
            ]
            if not remaining:
                break

            blocked = []
            for task in remaining:
                unavailable = [
                    dependency
                    for dependency in task.dependencies
                    if rows[dependency]["status"] in unavailable_statuses
                    and not self._dependency_ready(dependency, rows, substitutions)
                ]
                if unavailable:
                    blocked.append((task, unavailable))
            if blocked:
                for task, unavailable in blocked:
                    reason = f"blocked by unavailable dependencies: {', '.join(sorted(unavailable))}"
                    missing.append(f"{task.objective}: {reason}")
                    self.repository.record_event(
                        run_id,
                        task.id,
                        "DEPENDENCY_BLOCKED",
                        {"dependencies": sorted(unavailable)},
                        self.clock.now(),
                    )
                    self.repository.mark_skipped(run_id, task.id, reason, self.clock.now())
                continue

            stalled_ids = sorted(task.id for task in remaining)
            self.repository.record_event(
                run_id,
                None,
                "SCHEDULER_STALLED",
                {"pending_steps": stalled_ids},
                self.clock.now(),
            )
            raise SchedulerStalledError(f"no dependency-ready task; stalled steps: {stalled_ids}")

        return EngineResult(
            run_id=run_id,
            evidence=self.repository.list_evidence(run_id),
            recovery_decisions=self._recorded_recoveries(run_id),
            missing_evidence=list(dict.fromkeys(missing)),
        )
    ReportPeriodParameters,
