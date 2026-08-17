import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from research_agent.executor import SourceExecutor
from research_agent.fault_injection import FaultInjector
from research_agent.intent import CANONICAL_QUESTION, canonical_plan as build_canonical_plan
from research_agent.models import InvestigationPlan, RecoveryAction, ResearchTask
from research_agent.recovery import RecoveryPolicy
from research_agent.scheduler import SequentialScheduler
from research_agent.sources import BLSAdapter, FREDAdapter, SECAdapter, WorldBankAdapter
from research_agent.state import SQLiteRepository


FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def now(self) -> datetime:
        return self.current


class NoSleep:
    async def sleep(self, seconds: float) -> None:
        raise AssertionError(f"unexpected sleep: {seconds}")


class NoRandom:
    def uniform(self, lower: float, upper: float) -> float:
        raise AssertionError("unexpected jitter")


def canonical_plan() -> InvestigationPlan:
    return build_canonical_plan(CANONICAL_QUESTION)


def test_canonical_failure_substitutes_without_repeating_completed_sources(tmp_path: Path) -> None:
    payloads = {
        "data.sec.gov": ("sec_companyfacts_trimmed.json", "application/json"),
        "api.worldbank.org": ("world_bank_trimmed.json", "application/json"),
        "fred.stlouisfed.org": ("fred_observation_date.csv", "text/csv"),
        "api.bls.gov": ("bls_v1_trimmed.json", "application/json"),
    }
    calls = {host: 0 for host in payloads}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls[host] += 1
        filename, content_type = payloads[host]
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=(FIXTURES / filename).read_bytes(),
        )

    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={
                    "sec": SECAdapter("test contact@example.com"),
                    "world_bank": WorldBankAdapter(),
                    "fred": FREDAdapter(),
                    "bls": BLSAdapter(),
                },
                client=client,
                clock=FixedClock(),
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=FixedClock()),
                fault_injector=FaultInjector("fred:corrupt_once"),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=FixedClock())
            return await scheduler.run(canonical_plan(), run_id="run_demo")

    result = asyncio.run(run())
    assert calls == {host: 1 for host in payloads}
    assert result.recovery_decisions[0].selected_action is RecoveryAction.SUBSTITUTE
    assert result.recovery_decisions[0].replacement_source == "bls"
    metrics = {item.metric: item.value for item in result.evidence}
    assert metrics["apple_revenue_growth"] > 2.0
    assert metrics["us_cpi_inflation"] > metrics["apple_revenue_growth"]
    steps = {step["step_id"]: step for step in repository.list_steps("run_demo")}
    assert steps["sec_revenue"]["network_call_count"] == 1
    assert steps["world_bank_gdp"]["network_call_count"] == 1
    assert steps["fred_cpi"]["status"] == "FAILED"
    assert steps["bls_cpi"]["status"] == "VALIDATED"
    assert steps["bls_cpi"]["substitute_for_step_id"] == "fred_cpi"
    assert steps["compare"]["status"] == "VALIDATED"


def test_fy2023_failure_uses_dynamic_bls_periods(tmp_path: Path) -> None:
    payloads = {
        "data.sec.gov": ("sec_companyfacts_trimmed.json", "application/json"),
        "api.worldbank.org": ("world_bank_trimmed.json", "application/json"),
        "fred.stlouisfed.org": ("fred_observation_date.csv", "text/csv"),
        "api.bls.gov": ("bls_v1_trimmed.json", "application/json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        filename, content_type = payloads[request.url.host]
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=(FIXTURES / filename).read_bytes(),
        )

    repository = SQLiteRepository(tmp_path / "state.sqlite3")
    plan = build_canonical_plan("Did Apple's FY2023 revenue growth beat inflation?")

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={
                    "sec": SECAdapter("test contact@example.com"),
                    "world_bank": WorldBankAdapter(),
                    "fred": FREDAdapter(),
                    "bls": BLSAdapter(),
                },
                client=client,
                clock=FixedClock(),
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=FixedClock()),
                fault_injector=FaultInjector("fred:corrupt_once"),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=FixedClock())
            return await scheduler.run(plan, run_id="run_fy2023")

    result = asyncio.run(run())
    fallback = repository.get_step("run_fy2023", "bls_cpi")
    fallback_task = json.loads(fallback["task_json"])
    assert fallback_task["parameters"] == {"startyear": "2022", "endyear": "2023"}
    metrics = {item.metric: item for item in result.evidence}
    assert metrics["apple_revenue_growth"].period == "FY2022-FY2023"
    assert metrics["us_cpi_inflation"].period == "2022-09/2023-09"
    assert metrics["revenue_growth_minus_inflation"].value == pytest.approx(-6.5001586517)


def test_out_of_order_tasks_execute_by_dependencies_not_list_position(tmp_path: Path) -> None:
    plan = canonical_plan()
    reordered = plan.model_copy(update={"tasks": list(reversed(plan.tasks))})
    call_order: list[str] = []
    payloads = {
        "data.sec.gov": ("sec_companyfacts_trimmed.json", "application/json"),
        "api.worldbank.org": ("world_bank_trimmed.json", "application/json"),
        "fred.stlouisfed.org": ("fred_observation_date.csv", "text/csv"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        call_order.append(request.url.host)
        filename, content_type = payloads[request.url.host]
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=(FIXTURES / filename).read_bytes(),
        )

    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={
                    "sec": SECAdapter("test contact@example.com"),
                    "world_bank": WorldBankAdapter(),
                    "fred": FREDAdapter(),
                },
                client=client,
                clock=FixedClock(),
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=FixedClock()),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=FixedClock())
            return await scheduler.run(reordered, run_id="run_reordered")

    result = asyncio.run(run())
    assert call_order == ["data.sec.gov", "api.worldbank.org", "fred.stlouisfed.org"]
    assert any(item.metric == "revenue_growth_minus_inflation" for item in result.evidence)


def test_resume_reuses_validated_sources_with_zero_new_calls(tmp_path: Path) -> None:
    payloads = {
        "data.sec.gov": "sec_companyfacts_trimmed.json",
        "api.worldbank.org": "world_bank_trimmed.json",
        "fred.stlouisfed.org": "fred_observation_date.csv",
        "api.bls.gov": "bls_v1_trimmed.json",
    }
    calls = 0

    def first_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        content_type = "text/csv" if request.url.host == "fred.stlouisfed.org" else "application/json"
        return httpx.Response(
            200,
            headers={"content-type": content_type},
            content=(FIXTURES / payloads[request.url.host]).read_bytes(),
        )

    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    async def execute_once(handler, *, resume: bool) -> object:  # type: ignore[no-untyped-def]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={
                    "sec": SECAdapter("test contact@example.com"),
                    "world_bank": WorldBankAdapter(),
                    "fred": FREDAdapter(),
                    "bls": BLSAdapter(),
                },
                client=client,
                clock=FixedClock(),
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=FixedClock()),
                fault_injector=FaultInjector("fred:corrupt_once"),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=FixedClock())
            if resume:
                return await scheduler.run(run_id="run_resume", resume=True)
            return await scheduler.run(canonical_plan(), run_id="run_resume")

    asyncio.run(execute_once(first_handler, resume=False))
    first_call_count = calls

    def forbidden_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected repeat request: {request.url}")

    resumed = asyncio.run(execute_once(forbidden_handler, resume=True))
    assert calls == first_call_count
    events = repository.list_events("run_resume")
    retained_steps = {event["step_id"] for event in events if event["event_type"] == "CHECKPOINT_RETAINED"}
    assert {"sec_revenue", "world_bank_gdp"}.issubset(retained_steps)
    assert any(item.metric == "revenue_growth_minus_inflation" for item in resumed.evidence)


def test_primary_and_fallback_failure_preserves_partial_evidence(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "data.sec.gov":
            return httpx.Response(200, content=(FIXTURES / "sec_companyfacts_trimmed.json").read_bytes())
        if request.url.host == "api.worldbank.org":
            return httpx.Response(200, content=(FIXTURES / "world_bank_trimmed.json").read_bytes())
        if request.url.host == "fred.stlouisfed.org":
            return httpx.Response(200, content=(FIXTURES / "fred_observation_date.csv").read_bytes())
        return httpx.Response(200, json={"status": "REQUEST_FAILED", "Results": {}})

    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={
                    "sec": SECAdapter("test contact@example.com"),
                    "world_bank": WorldBankAdapter(),
                    "fred": FREDAdapter(),
                    "bls": BLSAdapter(),
                },
                client=client,
                clock=FixedClock(),
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=FixedClock()),
                fault_injector=FaultInjector("fred:corrupt_once"),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=FixedClock())
            return await scheduler.run(canonical_plan(), run_id="run_partial")

    result = asyncio.run(run())
    assert result.missing_evidence
    assert any(item.metric == "apple_revenue" for item in result.evidence)
    assert any(item.metric == "world_gdp_growth" for item in result.evidence)
    assert not any(item.metric == "us_cpi_inflation" for item in result.evidence)
    assert repository.get_step("run_partial", "compare")["status"] == "SKIPPED"


def test_expired_validated_artifact_is_refetched_within_resumed_run(tmp_path: Path) -> None:
    clock = MutableClock(NOW)
    task = ResearchTask(
        id="fred_cpi",
        objective="Retrieve September CPI",
        source="fred",
        parameters={"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]},
        importance="critical",
    )
    plan = InvestigationPlan(question="fixture freshness", tasks=[task])
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=(FIXTURES / "fred_observation_date.csv").read_bytes())

    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    async def run(*, resume: bool) -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={"fred": FREDAdapter()},
                client=client,
                clock=clock,
                sleeper=NoSleep(),
                recovery_policy=RecoveryPolicy(random_source=NoRandom(), clock=clock),
            )
            scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=clock)
            if resume:
                return await scheduler.run(run_id="run_freshness", resume=True)
            return await scheduler.run(plan, run_id="run_freshness")

    asyncio.run(run(resume=False))
    clock.current = NOW + timedelta(hours=25)
    asyncio.run(run(resume=True))
    assert calls == 2
    assert repository.get_step("run_freshness", "fred_cpi")["network_call_count"] == 2
