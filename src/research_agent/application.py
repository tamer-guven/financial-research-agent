"""Application service connecting planning, execution, and verified synthesis."""

from __future__ import annotations

from dataclasses import dataclass

from research_agent.clock import Clock
from research_agent.models import EvidenceBundle, FinalReport
from research_agent.providers import ModelUsage, ResilientModelProvider
from research_agent.scheduler import EngineResult, SequentialScheduler
from research_agent.state import SQLiteRepository


@dataclass
class ApplicationResult:
    engine: EngineResult
    report: FinalReport
    model_usage: ModelUsage


class ResearchApplication:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        scheduler: SequentialScheduler,
        provider: ResilientModelProvider,
        clock: Clock,
    ) -> None:
        self.repository = repository
        self.scheduler = scheduler
        self.provider = provider
        self.clock = clock

    def _record_model_fallbacks(self, run_id: str) -> None:
        for fallback in self.provider.drain_fallbacks():
            self.repository.record_event(
                run_id,
                None,
                "MODEL_FALLBACK",
                {"stage": fallback.stage, "reason": fallback.reason},
                self.clock.now(),
            )

    async def run(
        self,
        question: str,
        *,
        run_id: str | None = None,
        year_override: int | None = None,
    ) -> ApplicationResult:
        plan = self.provider.plan(question, year_override=year_override)
        engine = await self.scheduler.run(plan, run_id=run_id)
        self._record_model_fallbacks(engine.run_id)
        bundle = EvidenceBundle(
            intent=plan.intent,
            evidence=engine.evidence,
            recovery_decisions=engine.recovery_decisions,
            missing_evidence=engine.missing_evidence,
        )
        report = self.provider.synthesize(question, bundle)
        self._record_model_fallbacks(engine.run_id)
        self.repository.save_report(engine.run_id, report, self.clock.now())
        return ApplicationResult(engine=engine, report=report, model_usage=self.provider.usage())

    async def resume(self, run_id: str) -> ApplicationResult:
        self.provider.mark_planner_persisted()
        plan = self.repository.get_plan(run_id)
        engine = await self.scheduler.run(run_id=run_id, resume=True)
        bundle = EvidenceBundle(
            intent=plan.intent,
            evidence=engine.evidence,
            recovery_decisions=engine.recovery_decisions,
            missing_evidence=engine.missing_evidence,
        )
        report = self.provider.synthesize(plan.question, bundle)
        self._record_model_fallbacks(run_id)
        self.repository.save_report(run_id, report, self.clock.now())
        return ApplicationResult(engine=engine, report=report, model_usage=self.provider.usage())
