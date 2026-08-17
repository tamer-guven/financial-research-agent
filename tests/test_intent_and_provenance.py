import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_agent.application import ResearchApplication
from research_agent.intent import (
    CANONICAL_QUESTION,
    UnsupportedQuestionError,
    canonical_plan,
    is_supported_question,
    parse_research_intent,
    validate_plan_against_intent,
)
from research_agent.models import EvidenceBundle, EvidenceItem, FinalReport, ReportClaim
from research_agent.providers import DeterministicDemoProvider, ResilientModelProvider
from research_agent.provenance import ProvenanceError, verify_report
from research_agent.scheduler import EngineResult
from research_agent.state import SQLiteRepository


NOW = datetime(2026, 8, 16, tzinfo=UTC)


class FixedClock:
    def now(self) -> datetime:
        return NOW


@pytest.mark.parametrize(
    "question",
    [
        CANONICAL_QUESTION,
        "Compare AAPL FY24 sales growth with CPI inflation and give the world GDP growth backdrop.",
        "Did Apple top-line growth in fiscal 2022 outpace consumer price inflation? Include global GDP.",
        "For fiscal year 2023, did Apple's revenue grow faster than inflation?",
    ],
)
def test_supported_paraphrases(question: str) -> None:
    assert is_supported_question(question)
    plan = canonical_plan(question)
    assert {task.source for task in plan.tasks if task.kind == "source"} == {"sec", "world_bank", "fred"}
    assert plan.intent is not None
    validate_plan_against_intent(plan, plan.intent)


@pytest.mark.parametrize(
    "question",
    [
        "Did Tesla's FY2024 revenue beat inflation and what was global GDP growth?",
        "Should I buy Apple after FY2024 revenue growth versus CPI and global GDP growth?",
        "What was Apple's FY2024 operating margin?",
        "Did Apple's FY2023 and FY2024 revenue beat inflation?",
        "Did Apple's FY2021 revenue beat inflation?",
        "What is the weather today?",
    ],
)
def test_unrelated_or_out_of_scope_questions_are_rejected(question: str) -> None:
    assert not is_supported_question(question)
    with pytest.raises(UnsupportedQuestionError):
        canonical_plan(question)


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_intent_and_plan_are_year_parameterized(year: int) -> None:
    question = f"Did Apple's FY{year} revenue growth beat US inflation?"
    intent = parse_research_intent(question)
    assert intent.target_fiscal_year == year
    assert intent.baseline_fiscal_year == year - 1
    plan = canonical_plan(question)
    by_source = {task.source: task for task in plan.tasks if task.kind == "source"}
    assert by_source["sec"].parameters["years"] == [year - 1, year]
    assert by_source["fred"].parameters["dates"] == [f"{year - 1}-09-01", f"{year}-09-01"]
    assert by_source["world_bank"].parameters["years"] == [str(year - 1), str(year)]


def test_year_override_conflict_is_rejected() -> None:
    with pytest.raises(UnsupportedQuestionError, match="conflicts"):
        canonical_plan("Did Apple's FY2023 revenue growth beat CPI?", year_override=2024)


def test_supported_year_error_lists_supported_years() -> None:
    with pytest.raises(UnsupportedQuestionError, match="FY2022, FY2023, FY2024"):
        canonical_plan("Did Apple's FY2025 revenue growth beat CPI?")


def test_semantically_wrong_plan_is_rejected() -> None:
    question = "Did Apple's FY2023 revenue growth beat inflation?"
    plan = canonical_plan(question)
    tasks = [
        task.model_copy(update={"parameters": {"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]}})
        if task.source == "fred"
        else task
        for task in plan.tasks
    ]
    wrong = plan.model_copy(update={"tasks": tasks})
    assert plan.intent is not None
    with pytest.raises(ValueError, match="FRED task parameters"):
        validate_plan_against_intent(wrong, plan.intent)


def test_semantically_wrong_primary_plan_falls_back_deterministically() -> None:
    question = "Did Apple's FY2023 revenue growth beat inflation?"
    correct = canonical_plan(question)

    class WrongPrimary:
        def plan(self, question: str, *, year_override: int | None = None):  # type: ignore[no-untyped-def]
            tasks = [
                task.model_copy(update={"parameters": {"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]}})
                if task.source == "fred"
                else task
                for task in correct.tasks
            ]
            wrong = correct.model_copy(update={"tasks": tasks})
            assert correct.intent is not None
            validate_plan_against_intent(wrong, correct.intent)
            return wrong

        def synthesize(self, question: str, bundle: EvidenceBundle):  # type: ignore[no-untyped-def]
            raise AssertionError("not used")

    provider = ResilientModelProvider(
        deterministic=DeterministicDemoProvider(),
        primary=WrongPrimary(),
    )
    selected = provider.plan(question)
    assert selected == correct
    assert [fallback.stage for fallback in provider.drain_fallbacks()] == ["plan"]


def item(evidence_id: str, metric: str, value: float, period: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        run_id="run_model",
        task_id="step",
        metric=metric,
        value=value,
        unit="percent",
        period=period,
        source_name="validated fixture",
        source_url="https://example.com/source",
        artifact_id="art_fixture",
        retrieved_at=NOW,
    )


def complete_evidence(year: int = 2024) -> list[EvidenceItem]:
    values = {
        2022: (7.7937876042, 8.2016696438, -0.4078820396, 6.4850888760, 3.4400109727),
        2023: (-2.8004605303, 3.6996981213, -6.5001586517, 3.4400109727, 2.8593849334),
        2024: (2.0219940775, 2.4406330311, -0.4186389536, 2.8593849334, 2.9000134016),
    }
    revenue, inflation, difference, baseline_gdp, target_gdp = values[year]
    baseline = year - 1
    return [
        item("ev_revenue", "apple_revenue_growth", revenue, f"FY{baseline}-FY{year}"),
        item("ev_inflation", "us_cpi_inflation", inflation, f"{baseline}-09/{year}-09"),
        item("ev_difference", "revenue_growth_minus_inflation", difference, f"FY{year} approximate alignment"),
        item(f"ev_gdp_{baseline}", "world_gdp_growth", baseline_gdp, str(baseline)),
        item(f"ev_gdp_{year}", "world_gdp_growth", target_gdp, str(year)),
    ]


def test_deterministic_report_passes_claim_provenance() -> None:
    provider = DeterministicDemoProvider()
    evidence = complete_evidence()
    report = provider.synthesize(CANONICAL_QUESTION, EvidenceBundle(evidence=evidence))
    verify_report(report, evidence)
    assert report.overall_confidence.value == "MEDIUM"
    assert all(claim.evidence_ids for claim in report.claims)


@pytest.mark.parametrize("year", [2022, 2023, 2024])
def test_deterministic_report_uses_requested_year_and_provenance(year: int) -> None:
    question = f"Did Apple's FY{year} revenue growth beat inflation?"
    evidence = complete_evidence(year)
    report = DeterministicDemoProvider().synthesize(question, EvidenceBundle(evidence=evidence))
    verify_report(report, evidence)
    assert f"FY{year}" in report.direct_answer
    assert str(year - 1) in report.direct_answer
    assert str(year) in report.direct_answer
    assert all(claim.evidence_ids for claim in report.claims)


def test_claim_with_unknown_evidence_is_rejected() -> None:
    report = FinalReport(
        direct_answer="Unsupported value 9.99.",
        claims=[
            ReportClaim(
                claim_id="bad",
                text="Unsupported value 9.99.",
                evidence_ids=["missing"],
                confidence="LOW",
            )
        ],
        missing_evidence=[],
        recovery_summary="none",
        overall_confidence="LOW",
    )
    with pytest.raises(ProvenanceError, match="unknown evidence"):
        verify_report(report, complete_evidence())


def test_direct_answer_cannot_introduce_novel_numbers() -> None:
    evidence = [item("ev_revenue", "apple_revenue_growth", 2.02, "FY2024")]
    report = FinalReport(
        direct_answer="Revenue was 2.02%, and the unsupported figure was 9.99%.",
        claims=[
            ReportClaim(
                claim_id="revenue",
                text="Revenue growth was 2.02%.",
                evidence_ids=["ev_revenue"],
                confidence="HIGH",
            )
        ],
        missing_evidence=[],
        recovery_summary="none",
        overall_confidence="HIGH",
    )
    with pytest.raises(ProvenanceError, match="direct_answer"):
        verify_report(report, evidence)


def test_model_unavailability_records_model_fallback_event(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "state.sqlite3")

    class FailingPrimary:
        def plan(self, question: str):  # type: ignore[no-untyped-def]
            raise RuntimeError("model offline")

        def synthesize(self, question: str, bundle: EvidenceBundle):  # type: ignore[no-untyped-def]
            raise RuntimeError("model offline")

    class FakeScheduler:
        async def run(self, plan, *, run_id=None, resume=False):  # type: ignore[no-untyped-def]
            selected_run_id = run_id or "run_model"
            repository.create_run(selected_run_id, plan, NOW)
            for task in plan.tasks:
                repository.add_step(selected_run_id, task, NOW)
            return EngineResult(
                run_id=selected_run_id,
                evidence=complete_evidence(),
                recovery_decisions=[],
                missing_evidence=[],
            )

    resilient = ResilientModelProvider(
        deterministic=DeterministicDemoProvider(),
        primary=FailingPrimary(),
    )
    application = ResearchApplication(
        repository=repository,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        provider=resilient,
        clock=FixedClock(),
    )
    result = asyncio.run(application.run(CANONICAL_QUESTION, run_id="run_model"))
    assert result.report.overall_confidence.value == "MEDIUM"
    fallback_events = [
        event for event in repository.list_events("run_model") if event["event_type"] == "MODEL_FALLBACK"
    ]
    assert {event["event_payload"]["stage"] for event in fallback_events} == {"plan", "synthesize"}
