import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args, get_origin

import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

from research_agent.application import ResearchApplication
from research_agent.intent import CANONICAL_QUESTION, canonical_plan
from research_agent.models import (
    EvidenceBundle,
    FinalReport,
    InvestigationPlan,
    ReportClaim,
)
from research_agent.providers import (
    DeterministicDemoProvider,
    ResilientModelProvider,
    StructuredLLMProvider,
    build_synthesis_request,
)
from research_agent.provenance import ProvenanceError, verify_report
from research_agent.scheduler import EngineResult
from research_agent.state import SQLiteRepository

from test_intent_and_provenance import NOW, FixedClock, complete_evidence, item


def _assert_every_object_is_closed(node: object) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, node
        for value in node.values():
            _assert_every_object_is_closed(value)
    elif isinstance(node, list):
        for value in node:
            _assert_every_object_is_closed(value)


def _response_models(root: type[BaseModel]) -> set[type[BaseModel]]:
    found: set[type[BaseModel]] = set()

    def visit(annotation: object) -> None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in found:
                return
            found.add(annotation)
            for field in annotation.model_fields.values():
                visit(field.annotation)
            return
        for argument in get_args(annotation):
            visit(argument)

    visit(root)
    return found


def _contains_mapping(annotation: object) -> bool:
    return get_origin(annotation) is dict or any(
        _contains_mapping(argument) for argument in get_args(annotation)
    )


def test_investigation_plan_is_a_closed_openai_strict_schema() -> None:
    generated = InvestigationPlan.model_json_schema()
    strict = to_strict_json_schema(InvestigationPlan)

    _assert_every_object_is_closed(generated)
    _assert_every_object_is_closed(strict)
    assert strict["type"] == "object"
    assert set(strict["required"]) == set(strict["properties"])


def test_openai_response_models_have_no_mapping_fields() -> None:
    for root in (InvestigationPlan, FinalReport):
        for model in _response_models(root):
            for field in model.model_fields.values():
                assert not _contains_mapping(field.annotation), (
                    f"{model.__name__}.{field.name} must not be an unrestricted mapping"
                )


def test_openai_provider_uses_strict_parsing_exact_model_and_low_reasoning() -> None:
    deterministic = DeterministicDemoProvider()
    bundle = EvidenceBundle(evidence=complete_evidence())
    expected_report = deterministic.synthesize(CANONICAL_QUESTION, bundle)

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def parse(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            parsed = canonical_plan(CANONICAL_QUESTION) if len(self.calls) == 1 else expected_report
            return SimpleNamespace(output_parsed=parsed)

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    provider = StructuredLLMProvider(api_key="", model="gpt-5.6-luna", client=client)

    plan = provider.plan(CANONICAL_QUESTION)
    report = provider.synthesize(CANONICAL_QUESTION, bundle)

    assert plan.tasks
    assert report == expected_report
    assert len(responses.calls) == 2
    assert all(call["model"] == "gpt-5.6-luna" for call in responses.calls)
    assert all(call["reasoning"] == {"effort": "low"} for call in responses.calls)
    assert responses.calls[0]["text_format"].__name__ == "InvestigationPlan"
    assert responses.calls[1]["text_format"].__name__ == "FinalReport"


def test_synthesis_payload_excludes_raw_revenue_and_formats_reportable_values() -> None:
    evidence = complete_evidence() + [
        item("raw_revenue_2023", "apple_revenue", 383_285_000_000, "FY2023"),
        item("raw_revenue_2024", "apple_revenue", 391_035_000_000, "FY2024"),
    ]
    request = build_synthesis_request(
        CANONICAL_QUESTION,
        EvidenceBundle(evidence=evidence),
    )
    encoded = request.model_dump_json()
    projected = request.payload.evidence

    assert {entry.metric for entry in projected} == {
        "apple_revenue_growth",
        "us_cpi_inflation",
        "revenue_growth_minus_inflation",
        "world_gdp_growth",
    }
    assert "raw_revenue_2023" not in encoded
    assert "raw_revenue_2024" not in encoded
    assert "383285000000" not in encoded
    assert "391035000000" not in encoded
    assert all(re.fullmatch(r"-?\d+\.\d{2}", entry.value_display) for entry in projected)
    assert request.payload.allowed_numeric_values_by_evidence_id == {
        entry.evidence_id: [entry.value_display] for entry in projected
    }


@pytest.mark.parametrize("unsupported", ["383.285", "391.035"])
def test_report_rejects_unregistered_revenue_unit_conversions(unsupported: str) -> None:
    evidence = complete_evidence()
    report = FinalReport(
        direct_answer=f"Unsupported converted revenue was {unsupported}.",
        claims=[
            ReportClaim(
                claim_id="C1",
                text=f"Unsupported converted revenue was {unsupported}.",
                evidence_ids=["ev_revenue"],
                confidence="LOW",
            )
        ],
        missing_evidence=[],
        recovery_summary="No source recovery was required.",
        overall_confidence="LOW",
    )

    with pytest.raises(ProvenanceError, match="unsupported numeric values"):
        verify_report(report, evidence)


def test_provenance_repair_receives_exact_failure_and_permitted_values() -> None:
    evidence = complete_evidence()
    bundle = EvidenceBundle(evidence=evidence)
    valid = DeterministicDemoProvider().synthesize(CANONICAL_QUESTION, bundle)
    invalid = FinalReport(
        direct_answer="Apple revenue was 383.285 and 391.035 billion dollars.",
        claims=[
            ReportClaim(
                claim_id="C1",
                text="Apple revenue was 383.285 and 391.035 billion dollars.",
                evidence_ids=["ev_revenue"],
                confidence="LOW",
            )
        ],
        missing_evidence=[],
        recovery_summary="No source recovery was required.",
        overall_confidence="LOW",
    )

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def parse(self, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append(kwargs)
            return SimpleNamespace(output_parsed=invalid if len(self.calls) == 1 else valid)

    responses = FakeResponses()
    provider = StructuredLLMProvider(
        api_key="",
        model="gpt-5.6-luna",
        client=SimpleNamespace(responses=responses),
    )
    repaired = provider.synthesize(CANONICAL_QUESTION, bundle)

    verify_report(repaired, evidence)
    assert repaired == valid
    assert len(responses.calls) == 2
    repair_input = json.loads(responses.calls[1]["input"])
    repair = repair_input["repair"]
    assert repair["validation_failure"] == (
        "claim C1 contains unsupported numeric values: ['383.285', '391.035']"
    )
    assert repair["allowed_numeric_values_by_evidence_id"] == {
        entry.evidence_id: [entry.value_display]
        for entry in build_synthesis_request(CANONICAL_QUESTION, bundle).payload.evidence
    }
    assert any("Remove every unsupported" in action for action in repair["required_actions"])
    assert any("direct_answer" in action for action in repair["required_actions"])


def test_successful_openai_stages_record_no_model_fallbacks(tmp_path: Path) -> None:
    bundle = EvidenceBundle(evidence=complete_evidence())
    expected_plan = canonical_plan(CANONICAL_QUESTION)
    expected_report = DeterministicDemoProvider().synthesize(CANONICAL_QUESTION, bundle)

    class FakeResponses:
        def __init__(self) -> None:
            self.outputs = [expected_plan, expected_report]

        def parse(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(output_parsed=self.outputs.pop(0))

    class FakeScheduler:
        async def run(self, plan, *, run_id=None, resume=False):  # type: ignore[no-untyped-def]
            selected = run_id or "run_openai_success"
            repository.create_run(selected, plan, NOW)
            for task in plan.tasks:
                repository.add_step(selected, task, NOW)
            return EngineResult(
                run_id=selected,
                evidence=bundle.evidence,
                recovery_decisions=[],
                missing_evidence=[],
            )

    repository = SQLiteRepository(tmp_path / "openai-success.sqlite3")
    primary = StructuredLLMProvider(
        api_key="",
        model="gpt-5.6-luna",
        client=SimpleNamespace(responses=FakeResponses()),
    )
    resilient = ResilientModelProvider(
        deterministic=DeterministicDemoProvider(),
        primary=primary,
        requested_provider="openai",
        requested_model="gpt-5.6-luna",
    )
    application = ResearchApplication(
        repository=repository,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        provider=resilient,
        clock=FixedClock(),
    )
    result = asyncio.run(application.run(CANONICAL_QUESTION, run_id="run_openai_success"))

    assert not [
        event
        for event in repository.list_events(result.engine.run_id)
        if event["event_type"] == "MODEL_FALLBACK"
    ]
    assert result.model_usage.planner_used == "openai/gpt-5.6-luna"
    assert result.model_usage.synthesizer_used == "openai/gpt-5.6-luna"


@pytest.mark.live_openai
@pytest.mark.skipif(
    os.getenv("RUN_OPENAI_LIVE_TEST") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set RUN_OPENAI_LIVE_TEST=1 and OPENAI_API_KEY to run the live OpenAI contract test",
)
def test_optional_live_openai_plan_and_synthesis() -> None:
    evidence = complete_evidence()
    bundle = EvidenceBundle(evidence=evidence)
    provider = StructuredLLMProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
    )

    plan = provider.plan(CANONICAL_QUESTION)
    report = provider.synthesize(CANONICAL_QUESTION, bundle)

    assert plan.tasks
    verify_report(report, evidence)
