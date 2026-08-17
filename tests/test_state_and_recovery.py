import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx

from research_agent.executor import SourceExecutor
from research_agent.fault_injection import FaultInjector
from research_agent.models import InvestigationPlan, RawArtifact, RecoveryAction, ResearchTask
from research_agent.recovery import RecoveryPolicy
from research_agent.sources import FREDAdapter
from research_agent.state import SQLiteRepository


FIXTURE = Path(__file__).parent / "fixtures" / "fred_observation_date.csv"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


class RecordingSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.calls.append(seconds)


class ScriptedRandom:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.bounds: list[tuple[float, float]] = []

    def uniform(self, lower: float, upper: float) -> float:
        self.bounds.append((lower, upper))
        return next(self.values)


def fred_task() -> ResearchTask:
    return ResearchTask(
        id="fred_cpi",
        objective="Get CPI",
        source="fred",
        parameters={"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]},
        importance="critical",
        fallback_source="bls",
    )


def setup_repository(tmp_path: Path) -> SQLiteRepository:
    repository = SQLiteRepository(tmp_path / "state.sqlite3")
    plan = InvestigationPlan(question="test", tasks=[fred_task()])
    repository.create_run("run_test", plan, NOW)
    repository.add_step("run_test", fred_task(), NOW)
    return repository


def execute(executor: SourceExecutor) -> object:
    return asyncio.run(executor.execute("run_test", fred_task()))


def test_raw_artifact_is_committed_before_validator_runs(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path)

    class InspectingFRED(FREDAdapter):
        saw_committed_artifact = False

        def validate(self, artifact, request):  # type: ignore[no-untyped-def]
            reloaded = SQLiteRepository(repository.path).get_artifact(artifact.artifact_id)
            self.saw_committed_artifact = reloaded.payload == artifact.payload
            return super().validate(artifact, request)

    adapter = InspectingFRED()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FIXTURE.read_bytes(), headers={"content-type": "text/csv"})

    random_source = ScriptedRandom([])
    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={"fred": adapter},
                client=client,
                clock=FixedClock(),
                sleeper=RecordingSleeper(),
                recovery_policy=RecoveryPolicy(random_source=random_source, clock=FixedClock()),
            )
            return await executor.execute("run_test", fred_task())

    outcome = asyncio.run(run())
    assert adapter.saw_committed_artifact
    assert len(outcome.evidence) == 2
    event_types = [event["event_type"] for event in repository.list_events("run_test")]
    assert event_types.index("RAW_ARTIFACT_COMMITTED") < event_types.index("VALIDATION_COMMITTED")


def test_resume_validates_committed_raw_without_network_call(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path)
    adapter = FREDAdapter()
    request = adapter.make_request(fred_task().parameters, NOW.date().isoformat())
    http_request = httpx.Request("GET", adapter.endpoint)
    response = httpx.Response(200, request=http_request, content=FIXTURE.read_bytes(), headers={"content-type": "text/csv"})
    repository.mark_fetching("run_test", "fred_cpi", NOW)
    raw = adapter.artifact_from_response(
        run_id="run_test",
        step_id="fred_cpi",
        request=request,
        response=response,
        now=NOW,
    )
    repository.persist_raw_artifact(raw)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={"fred": adapter},
                client=client,
                clock=FixedClock(),
                sleeper=RecordingSleeper(),
                recovery_policy=RecoveryPolicy(random_source=ScriptedRandom([]), clock=FixedClock()),
            )
            return await executor.execute("run_test", fred_task())

    outcome = asyncio.run(run())
    assert calls == 0
    assert len(outcome.evidence) == 2
    assert repository.get_step("run_test", "fred_cpi")["network_call_count"] == 1


def test_transient_failure_uses_injected_rng_and_sleeper(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"content-type": "text/plain"}, content=b"temporarily unavailable")
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=FIXTURE.read_bytes())

    sleeper = RecordingSleeper()
    random_source = ScriptedRandom([0.25])

    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={"fred": FREDAdapter()},
                client=client,
                clock=FixedClock(),
                sleeper=sleeper,
                recovery_policy=RecoveryPolicy(random_source=random_source, clock=FixedClock()),
            )
            return await executor.execute("run_test", fred_task())

    outcome = asyncio.run(run())
    assert len(outcome.evidence) == 2
    assert calls == 2
    assert sleeper.calls == [0.25]
    assert random_source.bounds == [(0.0, 0.5)]
    step = repository.get_step("run_test", "fred_cpi")
    assert step["attempt_count"] == 2
    assert step["network_call_count"] == 2


def test_invalid_schema_substitutes_without_retry_and_persists_corruption(tmp_path: Path) -> None:
    repository = setup_repository(tmp_path)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, headers={"content-type": "text/csv"}, content=FIXTURE.read_bytes())

    sleeper = RecordingSleeper()
    async def run() -> object:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            executor = SourceExecutor(
                repository=repository,
                adapters={"fred": FREDAdapter()},
                client=client,
                clock=FixedClock(),
                sleeper=sleeper,
                recovery_policy=RecoveryPolicy(random_source=ScriptedRandom([]), clock=FixedClock()),
                fault_injector=FaultInjector("fred:corrupt_once"),
            )
            return await executor.execute("run_test", fred_task())

    outcome = asyncio.run(run())
    assert calls == 1
    assert sleeper.calls == []
    assert outcome.decision is not None
    assert outcome.decision.selected_action is RecoveryAction.SUBSTITUTE
    assert outcome.decision.replacement_source == "bls"
    events = repository.list_events("run_test")
    raw_event = next(event for event in events if event["event_type"] == "RAW_ARTIFACT_COMMITTED")
    raw = repository.get_artifact(raw_event["event_payload"]["artifact_id"])
    assert b"BROKEN_CPI" in raw.payload
    assert b"CPIAUCNS" not in raw.payload
