"""Fail-closed claim and synopsis provenance verification."""

from __future__ import annotations

import re

from research_agent.models import EvidenceItem, FinalReport, ValidationLabel


class ProvenanceError(ValueError):
    pass


def _decimal_tokens(text: str) -> set[str]:
    return set(re.findall(r"-?\d+\.\d+", text))


def verify_report(report: FinalReport, evidence: list[EvidenceItem]) -> None:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    if len(evidence_by_id) != len(evidence):
        raise ProvenanceError("evidence IDs must be unique")

    claim_numbers: set[str] = set()
    for claim in report.claims:
        if not claim.evidence_ids:
            raise ProvenanceError(f"claim {claim.claim_id} has no evidence IDs")
        try:
            cited = [evidence_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        except KeyError as exc:
            raise ProvenanceError(f"claim {claim.claim_id} cites unknown evidence {exc}") from exc
        if any(item.validation_label is not ValidationLabel.VALID for item in cited):
            raise ProvenanceError(f"claim {claim.claim_id} cites non-VALID evidence")
        numeric_tokens = _decimal_tokens(claim.text)
        allowed = {f"{item.value:.2f}" for item in cited} | {f"{abs(item.value):.2f}" for item in cited}
        unsupported = numeric_tokens - allowed
        if unsupported:
            raise ProvenanceError(
                f"claim {claim.claim_id} contains unsupported numeric values: {sorted(unsupported)}"
            )
        claim_numbers.update(numeric_tokens)

    novel_synopsis_numbers = _decimal_tokens(report.direct_answer) - claim_numbers
    if novel_synopsis_numbers:
        raise ProvenanceError(
            f"direct_answer contains numbers absent from claims: {sorted(novel_synopsis_numbers)}"
        )
