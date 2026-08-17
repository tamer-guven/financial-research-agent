"""Typed contracts shared by planning, execution, validation, and reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class ValidationLabel(StrEnum):
    VALID = "VALID"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"


class StepStatus(StrEnum):
    PENDING = "PENDING"
    FETCHING = "FETCHING"
    FETCHED = "FETCHED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class FailureClass(StrEnum):
    TRANSIENT_TRANSPORT = "TRANSIENT_TRANSPORT"
    RATE_LIMITED = "RATE_LIMITED"
    AUTHENTICATION = "AUTHENTICATION"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    CORRUPT_DATA = "CORRUPT_DATA"
    SEMANTIC_INVALID = "SEMANTIC_INVALID"
    STALE_DATA = "STALE_DATA"
    AMBIGUOUS_DATA = "AMBIGUOUS_DATA"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(StrEnum):
    RETRY = "RETRY"
    SUBSTITUTE = "SUBSTITUTE"
    PROCEED_PARTIAL = "PROCEED_PARTIAL"
    STOP = "STOP"
    NONE = "NONE"


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


SUPPORTED_YEARS = (2022, 2023, 2024)


class ResearchIntent(BaseModel):
    """Deterministically extracted parameters for the bounded Apple comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    company: Literal["Apple"] = "Apple"
    ticker: Literal["AAPL"] = "AAPL"
    target_fiscal_year: int
    baseline_fiscal_year: int
    inflation_month: Literal[9] = 9
    question_kind: Literal["revenue_vs_inflation_with_gdp_context"] = (
        "revenue_vs_inflation_with_gdp_context"
    )

    @model_validator(mode="after")
    def validate_periods(self) -> "ResearchIntent":
        if self.target_fiscal_year not in SUPPORTED_YEARS:
            supported = ", ".join(f"FY{year}" for year in SUPPORTED_YEARS)
            raise ValueError(f"supported fiscal years are {supported}")
        if self.baseline_fiscal_year != self.target_fiscal_year - 1:
            raise ValueError("baseline fiscal year must be target fiscal year minus one")
        return self


class TaskParameters(BaseModel):
    """Closed base contract for every executable task parameter object."""

    model_config = ConfigDict(extra="forbid")

    def __getitem__(self, key: str) -> Any:
        """Keep read-only mapping-style access at adapter/test boundaries."""

        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class SECParameters(TaskParameters):
    company: Literal["Apple"]
    ticker: Literal["AAPL"]
    cik: str
    concept: str
    years: list[int]


class WorldBankParameters(TaskParameters):
    country: Literal["WLD"]
    indicator: str
    years: list[str]


class FREDParameters(TaskParameters):
    id: Literal["CPIAUCNS"]
    dates: list[str]


class BLSParameters(TaskParameters):
    startyear: str
    endyear: str


class ReportPeriodParameters(TaskParameters):
    baseline_fiscal_year: int
    target_fiscal_year: int
    inflation_month: Literal[9]


ResearchTaskParameters = (
    SECParameters
    | WorldBankParameters
    | FREDParameters
    | BLSParameters
    | ReportPeriodParameters
)


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    source: Literal["sec", "world_bank", "fred", "bls", "calculation", "report"]
    parameters: ResearchTaskParameters
    dependencies: list[str] = Field(default_factory=list)
    importance: Literal["critical", "supporting"]
    fallback_source: Literal["bls"] | None = None
    kind: Literal["source", "calculation", "report"] = "source"

    @model_validator(mode="after")
    def validate_parameter_contract(self) -> "ResearchTask":
        expected = {
            "sec": SECParameters,
            "world_bank": WorldBankParameters,
            "fred": FREDParameters,
            "bls": BLSParameters,
            "calculation": ReportPeriodParameters,
            "report": ReportPeriodParameters,
        }[self.source]
        if not isinstance(self.parameters, expected):
            raise ValueError(f"{self.source} task requires {expected.__name__}")
        expected_kind = self.source if self.source in {"calculation", "report"} else "source"
        if self.kind != expected_kind:
            raise ValueError(f"{self.source} task must use kind={expected_kind}")
        return self


def validate_task_graph(tasks: Sequence[ResearchTask]) -> None:
    """Validate identifiers and dependencies independently of task-list order."""

    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs must be unique")
    known = set(ids)
    for task in tasks:
        missing = set(task.dependencies) - known
        if missing:
            raise ValueError(f"task {task.id} has missing dependencies: {sorted(missing)}")

    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {task.id: task for task in tasks}

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError("task graph must be acyclic")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].dependencies:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    intent: ResearchIntent | None = None
    tasks: list[ResearchTask]

    @model_validator(mode="after")
    def validate_graph_shape(self) -> "InvestigationPlan":
        validate_task_graph(self.tasks)
        return self


class SourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    endpoint: str
    parameters: dict[str, Any]
    as_of_date: str


class RawArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    run_id: str
    step_id: str
    source: str
    request_url: str
    request_fingerprint: str
    fetched_at: datetime
    valid_until: datetime
    status_code: int | None
    content_type: str | None
    response_headers: dict[str, str] = Field(default_factory=dict)
    payload: bytes
    payload_sha256: str


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: ValidationLabel
    failure_class: FailureClass | None = None
    reasons: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    run_id: str
    task_id: str
    metric: str
    value: float
    unit: str
    period: str
    source_name: str
    source_url: str
    artifact_id: str
    retrieved_at: datetime = Field(default_factory=utc_now)
    transformation: str | None = None
    input_evidence_ids: list[str] = Field(default_factory=list)
    validation_label: ValidationLabel = ValidationLabel.VALID


class RecoveryDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: str
    failure_class: FailureClass
    transient: bool
    considered_actions: list[RecoveryAction]
    selected_action: RecoveryAction
    replacement_source: str | None = None
    justification: str


class ReportClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    evidence_ids: list[str]
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    limitation: str | None = None


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direct_answer: str
    claims: list[ReportClaim]
    missing_evidence: list[str]
    recovery_summary: str
    overall_confidence: Confidence


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: ResearchIntent | None = None
    evidence: list[EvidenceItem]
    recovery_decisions: list[RecoveryDecision] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


class SynthesisEvidence(BaseModel):
    """Claim-safe evidence projection sent to a synthesis model."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    metric: str
    period: str
    value_display: str = Field(pattern=r"^-?\d+\.\d{2}$")
    unit: str
    source_name: str
    source_url: str
    transformation: str | None
    input_evidence_ids: list[str]
    validation_label: ValidationLabel


class SynthesisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    evidence: list[SynthesisEvidence]
    allowed_numeric_values_by_evidence_id: dict[str, list[str]]
    recovery_decisions: list[RecoveryDecision]
    missing_evidence: list[str]


class SynthesisRepairContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation_failure: str
    allowed_numeric_values_by_evidence_id: dict[str, list[str]]
    required_actions: list[str]


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: SynthesisPayload
    repair: SynthesisRepairContext | None = None
