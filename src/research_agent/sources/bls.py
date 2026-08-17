"""Preconfigured unregistered BLS V1 CPI fallback."""

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


class BLSAdapter(BaseSourceAdapter):
    name = "bls"
    endpoint = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
    ttl = timedelta(hours=24)
    series = "CUUR0000SA0"

    def http_request(self, request: SourceRequest) -> tuple[str, str, dict[str, Any]]:
        body = {
            "seriesid": [self.series],
            "startyear": str(request.parameters["startyear"]),
            "endyear": str(request.parameters["endyear"]),
        }
        return "POST", self.endpoint, {"json": body}

    def _observations(self, artifact: RawArtifact) -> list[dict[str, Any]]:
        payload = json.loads(artifact.payload)
        if payload["status"] != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS status was {payload['status']}")
        series = payload["Results"]["series"]
        if len(series) != 1 or series[0]["seriesID"] != self.series:
            raise KeyError("unexpected BLS series")
        return series[0]["data"]

    def validate(self, artifact: RawArtifact, request: SourceRequest) -> ValidationResult:
        transport = self.transport_validation(artifact)
        if transport:
            return transport
        try:
            observations = self._observations(artifact)
            years = {str(request.parameters["startyear"]), str(request.parameters["endyear"])}
            selected = [row for row in observations if row["period"] == "M09" and row["year"] in years]
            values = [float(row["value"]) for row in selected]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.INVALID_SCHEMA,
                reasons=[f"invalid BLS V1 schema: {exc}"],
            )
        if {row["year"] for row in selected} != years or len(selected) != len(years):
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.SEMANTIC_INVALID,
                reasons=["requested BLS September observations are missing or duplicated"],
            )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.CORRUPT_DATA,
                reasons=["BLS CPI values must be finite and positive"],
            )
        return ValidationResult(label=ValidationLabel.VALID)

    def normalize(
        self,
        artifact: RawArtifact,
        request: SourceRequest,
        validation: ValidationResult,
    ) -> list[EvidenceItem]:
        if validation.label is not ValidationLabel.VALID:
            raise ValueError("only VALID BLS artifacts can be normalized")
        years = {str(request.parameters["startyear"]), str(request.parameters["endyear"])}
        return [
            EvidenceItem(
                evidence_id=f"ev_{uuid4().hex}",
                run_id=artifact.run_id,
                task_id=artifact.step_id,
                metric="us_cpi_u_nsa",
                value=float(row["value"]),
                unit="index_1982_1984_100",
                period=f"{row['year']}-09",
                source_name="BLS CPI-U via unregistered V1 API",
                source_url=artifact.request_url,
                artifact_id=artifact.artifact_id,
                retrieved_at=artifact.fetched_at,
                transformation="selected M09 observation",
            )
            for row in self._observations(artifact)
            if row.get("period") == "M09" and row.get("year") in years
        ]
