"""Deterministic and optional structured OpenAI model providers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from research_agent.intent import (
    canonical_plan,
    parse_research_intent,
    validate_plan_against_intent,
)
from research_agent.models import (
    Confidence,
    EvidenceBundle,
    FinalReport,
    InvestigationPlan,
    ReportClaim,
    SynthesisEvidence,
    SynthesisPayload,
    SynthesisRepairContext,
    SynthesisRequest,
    ValidationLabel,
)
from research_agent.provenance import verify_report


class ModelProvider(Protocol):
    def plan(self, question: str, *, year_override: int | None = None) -> InvestigationPlan: ...

    def synthesize(self, question: str, bundle: EvidenceBundle) -> FinalReport: ...


def _by_metric(bundle: EvidenceBundle, metric: str) -> list[Any]:
    return [item for item in bundle.evidence if item.metric == metric]


REPORTABLE_SYNTHESIS_METRICS = frozenset(
    {
        "apple_revenue_growth",
        "us_cpi_inflation",
        "revenue_growth_minus_inflation",
        "world_gdp_growth",
    }
)


def build_synthesis_request(
    question: str,
    bundle: EvidenceBundle,
    *,
    repair_failure: str | None = None,
) -> SynthesisRequest:
    """Project validated evidence into the only values a report may quote."""

    projected = [
        SynthesisEvidence(
            evidence_id=item.evidence_id,
            metric=item.metric,
            period=item.period,
            value_display=f"{item.value:.2f}",
            unit=item.unit,
            source_name=item.source_name,
            source_url=item.source_url,
            transformation=item.transformation,
            input_evidence_ids=item.input_evidence_ids,
            validation_label=item.validation_label,
        )
        for item in bundle.evidence
        if item.metric in REPORTABLE_SYNTHESIS_METRICS
        and item.validation_label is ValidationLabel.VALID
    ]
    allowed = {item.evidence_id: [item.value_display] for item in projected}
    payload = SynthesisPayload(
        question=question,
        evidence=projected,
        allowed_numeric_values_by_evidence_id=allowed,
        recovery_decisions=bundle.recovery_decisions,
        missing_evidence=bundle.missing_evidence,
    )
    repair = None
    if repair_failure is not None:
        repair = SynthesisRepairContext(
            validation_failure=repair_failure,
            allowed_numeric_values_by_evidence_id=allowed,
            required_actions=[
                "Remove every unsupported numeric value.",
                "Each claim may use only numbers allowed for that claim's cited evidence IDs.",
                "direct_answer may only repeat numbers already present in verified claims.",
            ],
        )
    return SynthesisRequest(payload=payload, repair=repair)


class DeterministicDemoProvider:
    def plan(self, question: str, *, year_override: int | None = None) -> InvestigationPlan:
        return canonical_plan(question, year_override=year_override)

    def synthesize(self, question: str, bundle: EvidenceBundle) -> FinalReport:
        intent = bundle.intent or parse_research_intent(question)
        baseline = intent.baseline_fiscal_year
        target = intent.target_fiscal_year
        month = intent.inflation_month
        claims: list[ReportClaim] = []
        revenue = [
            item
            for item in _by_metric(bundle, "apple_revenue_growth")
            if item.period == f"FY{baseline}-FY{target}"
        ]
        inflation = [
            item
            for item in _by_metric(bundle, "us_cpi_inflation")
            if item.period == f"{baseline}-{month:02d}/{target}-{month:02d}"
        ]
        difference = [
            item
            for item in _by_metric(bundle, "revenue_growth_minus_inflation")
            if item.period == f"FY{target} approximate alignment"
        ]
        gdp = sorted(
            [
                item
                for item in _by_metric(bundle, "world_gdp_growth")
                if item.period in {str(baseline), str(target)}
            ],
            key=lambda item: item.period,
        )

        if revenue:
            claims.append(
                ReportClaim(
                    claim_id="claim_revenue_growth",
                    text=f"Apple's FY{target} revenue growth was {revenue[-1].value:.2f}%.",
                    evidence_ids=[revenue[-1].evidence_id],
                    confidence="HIGH",
                )
            )
        if inflation:
            claims.append(
                ReportClaim(
                    claim_id="claim_inflation",
                    text=(
                        f"September {baseline} to September {target} US CPI inflation was "
                        f"{inflation[-1].value:.2f}%."
                    ),
                    evidence_ids=[inflation[-1].evidence_id],
                    confidence="MEDIUM",
                    limitation="September-to-September CPI is an approximate alignment to Apple's fiscal year.",
                )
            )
        if difference:
            claims.append(
                ReportClaim(
                    claim_id="claim_comparison",
                    text=(
                        "The revenue-growth minus inflation difference was "
                        f"{difference[-1].value:.2f} percentage points."
                    ),
                    evidence_ids=[difference[-1].evidence_id],
                    confidence="MEDIUM",
                    limitation="This is a nominal comparison, not a fiscal-year real-revenue calculation.",
                )
            )
        if len(gdp) >= 2:
            claims.append(
                ReportClaim(
                    claim_id="claim_global_gdp",
                    text=(
                        f"World GDP growth moved from {gdp[0].value:.2f}% in {baseline} "
                        f"to {gdp[1].value:.2f}% in {target}."
                    ),
                    evidence_ids=[gdp[0].evidence_id, gdp[1].evidence_id],
                    confidence="HIGH",
                )
            )

        if revenue and inflation and difference:
            verdict = "Yes" if difference[-1].value > 0 else "No"
            relation = "above" if difference[-1].value > 0 else "below"
            direct = (
                f"{verdict}. Apple's FY{target} revenue grew {revenue[-1].value:.2f}%, {relation} comparable "
                f"inflation of {inflation[-1].value:.2f}%; the difference was "
                f"{difference[-1].value:.2f} percentage points."
            )
            overall = Confidence.MEDIUM
        else:
            direct = "The central comparison is incomplete because validated critical evidence is unavailable."
            overall = Confidence.INSUFFICIENT

        if gdp and claims:
            direct += (
                f" World GDP growth was {gdp[0].value:.2f}% in {baseline} "
                f"and {gdp[-1].value:.2f}% in {target}."
            )

        if bundle.recovery_decisions:
            recovery_summary = " ".join(
                f"{decision.failure_class.value}: {decision.selected_action.value}"
                + (f" with {decision.replacement_source}" if decision.replacement_source else "")
                + f" ({decision.justification})."
                for decision in bundle.recovery_decisions
            )
            if any(decision.replacement_source == "bls" for decision in bundle.recovery_decisions):
                recovery_summary += (
                    " BLS restored CPI availability as an alternative delivery channel; "
                    "it is not independent corroboration of FRED's BLS-originated series."
                )
        else:
            recovery_summary = "No source recovery was required."

        report = FinalReport(
            direct_answer=direct,
            claims=claims,
            missing_evidence=bundle.missing_evidence,
            recovery_summary=recovery_summary,
            overall_confidence=overall,
        )
        verify_report(report, bundle.evidence)
        return report


class StructuredLLMProvider:
    """Strict Pydantic output over the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5.6-luna",
        client: Any | None = None,
    ) -> None:
        self.client = client or OpenAI(api_key=api_key)
        self.model = model

    def plan(self, question: str, *, year_override: int | None = None) -> InvestigationPlan:
        # The scope guard runs before paid work and again after model parsing.
        intent = parse_research_intent(question, year_override=year_override)
        deterministic_plan = canonical_plan(question, year_override=year_override)
        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=(
                "Return a typed investigation DAG matching the supplied deterministic intent exactly. "
                "Use only source names sec, world_bank, and fred; calculation and report are deterministic. "
                "FRED must name bls as its fallback. Preserve the exact years, dates, parameters, and "
                "dependencies from the deterministic reference plan. Do not calculate values."
            ),
            input=json.dumps(
                {
                    "question": question,
                    "intent": intent.model_dump(mode="json"),
                    "deterministic_reference_plan": deterministic_plan.model_dump(mode="json"),
                },
                sort_keys=True,
            ),
            text_format=InvestigationPlan,
        )
        plan = response.output_parsed
        if plan is None:
            raise ValueError("OpenAI returned no parsed investigation plan")
        validate_plan_against_intent(plan, intent)
        return plan

    def _synthesize_once(
        self,
        question: str,
        bundle: EvidenceBundle,
        repair_instruction: str | None = None,
    ) -> FinalReport:
        instructions = (
            "Create a concise factual report using only the supplied claim-safe VALID evidence projection. "
            "Every factual claim must cite evidence_ids. Each claim may use only the exact two-decimal "
            "numbers listed for its cited evidence IDs in allowed_numeric_values_by_evidence_id. Do not "
            "recalculate, rescale, convert units, or introduce other numeric values. direct_answer may only "
            "repeat numbers already present in verified claims. Disclose recovery and limitations."
        )
        request = build_synthesis_request(
            question,
            bundle,
            repair_failure=repair_instruction,
        )
        if repair_instruction is not None:
            instructions += (
                " The previous report failed provenance validation. Follow the repair object exactly: "
                "remove unsupported numbers and ensure direct_answer repeats only numbers already present "
                "in claims."
            )
        response = self.client.responses.parse(
            model=self.model,
            reasoning={"effort": "low"},
            instructions=instructions,
            input=request.model_dump_json(),
            text_format=FinalReport,
        )
        report = response.output_parsed
        if report is None:
            raise ValueError("OpenAI returned no parsed final report")
        return report

    def synthesize(self, question: str, bundle: EvidenceBundle) -> FinalReport:
        report = self._synthesize_once(question, bundle)
        try:
            verify_report(report, bundle.evidence)
            return report
        except ValueError as first_error:
            repaired = self._synthesize_once(question, bundle, str(first_error))
            verify_report(repaired, bundle.evidence)
            return repaired


@dataclass
class ModelFallback:
    stage: str
    reason: str


@dataclass(frozen=True)
class ModelUsage:
    requested_provider: str
    requested_model: str | None
    planner_used: str
    synthesizer_used: str


def _concise_fallback_reason(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    message = " ".join(message.split())
    return message if len(message) <= 320 else message[:317] + "..."


class ResilientModelProvider:
    def __init__(
        self,
        *,
        deterministic: DeterministicDemoProvider,
        primary: ModelProvider | None = None,
        requested_provider: str | None = None,
        requested_model: str | None = None,
    ) -> None:
        self.deterministic = deterministic
        self.primary = primary
        self.requested_provider = requested_provider or ("openai" if primary else "deterministic")
        self.requested_model = requested_model or getattr(primary, "model", None)
        self._primary_label = (
            f"openai/{self.requested_model}"
            if self.requested_provider == "openai" and self.requested_model
            else self.requested_provider
        )
        self._planner_used = "not run"
        self._synthesizer_used = "not run"
        self._fallbacks: list[ModelFallback] = []

    def plan(self, question: str, *, year_override: int | None = None) -> InvestigationPlan:
        if self.primary is None:
            self._planner_used = "deterministic"
            return self.deterministic.plan(question, year_override=year_override)
        try:
            plan = self.primary.plan(question, year_override=year_override)
            self._planner_used = self._primary_label
            return plan
        except Exception as exc:
            self._planner_used = "deterministic (fallback)"
            self._fallbacks.append(ModelFallback("plan", _concise_fallback_reason(exc)))
            return self.deterministic.plan(question, year_override=year_override)

    def synthesize(self, question: str, bundle: EvidenceBundle) -> FinalReport:
        if self.primary is None:
            self._synthesizer_used = "deterministic"
            return self.deterministic.synthesize(question, bundle)
        try:
            report = self.primary.synthesize(question, bundle)
            self._synthesizer_used = self._primary_label
            return report
        except Exception as exc:
            self._synthesizer_used = "deterministic (fallback)"
            self._fallbacks.append(ModelFallback("synthesize", _concise_fallback_reason(exc)))
            return self.deterministic.synthesize(question, bundle)

    def drain_fallbacks(self) -> list[ModelFallback]:
        fallbacks, self._fallbacks = self._fallbacks, []
        return fallbacks

    def mark_planner_persisted(self) -> None:
        self._planner_used = "persisted plan (not rerun)"

    def usage(self) -> ModelUsage:
        return ModelUsage(
            requested_provider=self.requested_provider,
            requested_model=self.requested_model,
            planner_used=self._planner_used,
            synthesizer_used=self._synthesizer_used,
        )
