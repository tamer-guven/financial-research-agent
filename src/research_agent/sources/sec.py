"""SEC Company Facts adapter for Apple's annual revenue."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from typing import Any
from uuid import uuid4

from research_agent.models import (
    EvidenceItem,
    FailureClass,
    RawArtifact,
    SourceRequest,
    ValidationLabel,
    ValidationResult,
)
from research_agent.sources.base import BaseSourceAdapter


class SECAdapter(BaseSourceAdapter):
    name = "sec"
    endpoint = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    ttl = timedelta(hours=24)
    cik = "0000320193"
    concept = "RevenueFromContractWithCustomerExcludingAssessedTax"
    accepted_forms = {"10-K", "10-K/A"}

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent

    def http_request(self, request: SourceRequest) -> tuple[str, str, dict[str, Any]]:
        return "GET", self.endpoint, {"headers": {"User-Agent": self.user_agent}}

    def _selected(self, payload: dict[str, Any], years: list[int]) -> dict[int, dict[str, Any]]:
        facts = payload["facts"]["us-gaap"][self.concept]["units"]["USD"]
        selected: dict[int, dict[str, Any]] = {}
        for year in years:
            candidates = [
                fact
                for fact in facts
                if fact.get("form") in self.accepted_forms
                and fact.get("fp") == "FY"
                and fact.get("fy") == year
            ]
            if candidates:
                selected[year] = max(
                    candidates,
                    key=lambda fact: (
                        fact.get("end", ""),
                        fact.get("filed", ""),
                        fact.get("form") == "10-K/A",
                        fact.get("accn", ""),
                    ),
                )
        return selected

    def validate(self, artifact: RawArtifact, request: SourceRequest) -> ValidationResult:
        transport = self.transport_validation(artifact)
        if transport:
            return transport
        try:
            payload = json.loads(artifact.payload)
            if int(payload["cik"]) != 320193:
                return ValidationResult(
                    label=ValidationLabel.INVALID,
                    failure_class=FailureClass.SEMANTIC_INVALID,
                    reasons=["unexpected CIK"],
                )
            if "apple" not in str(payload.get("entityName", "")).lower():
                return ValidationResult(
                    label=ValidationLabel.INVALID,
                    failure_class=FailureClass.SEMANTIC_INVALID,
                    reasons=["unexpected company entity"],
                )
            expected = {
                "company": "Apple",
                "ticker": "AAPL",
                "cik": self.cik,
                "concept": self.concept,
            }
            if any(
                key in request.parameters and str(request.parameters[key]) != value
                for key, value in expected.items()
            ):
                return ValidationResult(
                    label=ValidationLabel.INVALID,
                    failure_class=FailureClass.SEMANTIC_INVALID,
                    reasons=["SEC request parameters do not identify Apple revenue"],
                )
            years = [int(year) for year in request.parameters["years"]]
            selected = self._selected(payload, years)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.INVALID_SCHEMA,
                reasons=[f"invalid SEC Company Facts schema: {exc}"],
            )
        if set(selected) != set(years):
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.SEMANTIC_INVALID,
                reasons=["requested fiscal-year revenue facts are missing"],
            )
        if any(
            not isinstance(fact.get("val"), (int, float))
            or not math.isfinite(float(fact["val"]))
            or float(fact["val"]) <= 0
            for fact in selected.values()
        ):
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.CORRUPT_DATA,
                reasons=["revenue values must be finite and positive"],
            )
        return ValidationResult(label=ValidationLabel.VALID)

    def normalize(
        self,
        artifact: RawArtifact,
        request: SourceRequest,
        validation: ValidationResult,
    ) -> list[EvidenceItem]:
        if validation.label is not ValidationLabel.VALID:
            raise ValueError("only VALID SEC artifacts can be normalized")
        payload = json.loads(artifact.payload)
        years = [int(year) for year in request.parameters["years"]]
        selected = self._selected(payload, years)
        return [
            EvidenceItem(
                evidence_id=f"ev_{uuid4().hex}",
                run_id=artifact.run_id,
                task_id=artifact.step_id,
                metric="apple_revenue",
                value=float(selected[year]["val"]),
                unit="USD",
                period=f"FY{year}",
                source_name="SEC EDGAR Company Facts",
                source_url=artifact.request_url,
                artifact_id=artifact.artifact_id,
                retrieved_at=artifact.fetched_at,
                transformation=(
                    f"selected latest fiscal-period {selected[year]['form']} fact for fy={year}; "
                    f"filed={selected[year].get('filed', '')}; accn={selected[year].get('accn', '')}"
                ),
            )
            for year in sorted(years)
        ]
