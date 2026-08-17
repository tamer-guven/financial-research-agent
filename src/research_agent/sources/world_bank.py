"""World Bank Indicators API adapter."""

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


class WorldBankAdapter(BaseSourceAdapter):
    name = "world_bank"
    endpoint = "https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG"
    ttl = timedelta(days=7)

    def http_request(self, request: SourceRequest) -> tuple[str, str, dict[str, Any]]:
        years = sorted(request.parameters["years"])
        return "GET", self.endpoint, {
            "params": {"date": f"{years[0]}:{years[-1]}", "format": "json", "per_page": 100}
        }

    def _records(self, artifact: RawArtifact) -> list[dict[str, Any]]:
        payload = json.loads(artifact.payload)
        if not isinstance(payload, list) or len(payload) != 2 or not isinstance(payload[1], list):
            raise TypeError("expected metadata and records arrays")
        return payload[1]

    def validate(self, artifact: RawArtifact, request: SourceRequest) -> ValidationResult:
        transport = self.transport_validation(artifact)
        if transport:
            return transport
        if request.parameters.get("country", "WLD") != "WLD" or request.parameters.get(
            "indicator", "NY.GDP.MKTP.KD.ZG"
        ) != "NY.GDP.MKTP.KD.ZG":
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.SEMANTIC_INVALID,
                reasons=["unexpected World Bank country or indicator"],
            )
        try:
            records = self._records(artifact)
            years = set(request.parameters["years"])
            selected = [
                record
                for record in records
                if record["countryiso3code"] == "WLD"
                and record["indicator"]["id"] == "NY.GDP.MKTP.KD.ZG"
                and record["date"] in years
            ]
            values = [float(record["value"]) for record in selected]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.INVALID_SCHEMA,
                reasons=[f"invalid World Bank schema: {exc}"],
            )
        if {record["date"] for record in selected} != years:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.SEMANTIC_INVALID,
                reasons=["requested world GDP years are missing"],
            )
        if any(not math.isfinite(value) or not -50 <= value <= 50 for value in values):
            return ValidationResult(
                label=ValidationLabel.AMBIGUOUS,
                failure_class=FailureClass.AMBIGUOUS_DATA,
                reasons=["GDP growth is outside the broad plausibility range"],
            )
        return ValidationResult(label=ValidationLabel.VALID)

    def normalize(
        self,
        artifact: RawArtifact,
        request: SourceRequest,
        validation: ValidationResult,
    ) -> list[EvidenceItem]:
        if validation.label is not ValidationLabel.VALID:
            raise ValueError("only VALID World Bank artifacts can be normalized")
        years = set(request.parameters["years"])
        records = self._records(artifact)
        return [
            EvidenceItem(
                evidence_id=f"ev_{uuid4().hex}",
                run_id=artifact.run_id,
                task_id=artifact.step_id,
                metric="world_gdp_growth",
                value=float(record["value"]),
                unit="percent",
                period=record["date"],
                source_name="World Bank Indicators API",
                source_url=artifact.request_url,
                artifact_id=artifact.artifact_id,
                retrieved_at=artifact.fetched_at,
            )
            for record in records
            if record.get("countryiso3code") == "WLD"
            and record.get("indicator", {}).get("id") == "NY.GDP.MKTP.KD.ZG"
            and record.get("date") in years
        ]
