"""FRED CSV adapter with date-header normalization."""

from __future__ import annotations

import csv
import io
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


class FREDAdapter(BaseSourceAdapter):
    name = "fred"
    endpoint = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    ttl = timedelta(hours=24)
    series = "CPIAUCNS"

    def http_request(self, request: SourceRequest) -> tuple[str, str, dict[str, Any]]:
        params = {
            "id": self.series,
            "cosd": request.parameters["dates"][0],
            "coed": request.parameters["dates"][-1],
        }
        return "GET", self.endpoint, {"params": params}

    def _rows(self, artifact: RawArtifact) -> tuple[str, list[dict[str, str]]]:
        text = artifact.payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        fields = reader.fieldnames or []
        date_field = "observation_date" if "observation_date" in fields else "DATE" if "DATE" in fields else ""
        if not date_field or self.series not in fields:
            raise KeyError(f"required columns absent; received {fields}")
        return date_field, list(reader)

    def validate(self, artifact: RawArtifact, request: SourceRequest) -> ValidationResult:
        transport = self.transport_validation(artifact)
        if transport:
            return transport
        if request.parameters.get("id") != self.series:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.SEMANTIC_INVALID,
                reasons=["unexpected FRED series"],
            )
        try:
            date_field, rows = self._rows(artifact)
            requested = set(request.parameters["dates"])
            matching = [row for row in rows if row[date_field] in requested]
            if {row[date_field] for row in matching} != requested:
                return ValidationResult(
                    label=ValidationLabel.INVALID,
                    failure_class=FailureClass.SEMANTIC_INVALID,
                    reasons=["requested CPI dates are missing"],
                )
            if len(matching) != len(requested):
                return ValidationResult(
                    label=ValidationLabel.INVALID,
                    failure_class=FailureClass.CORRUPT_DATA,
                    reasons=["duplicate CPI observations"],
                )
            values = [float(row[self.series]) for row in matching]
        except (UnicodeDecodeError, csv.Error, KeyError, TypeError, ValueError) as exc:
            failure = FailureClass.INVALID_SCHEMA if isinstance(exc, KeyError) else FailureClass.CORRUPT_DATA
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=failure,
                reasons=[f"invalid FRED CSV: {exc}"],
            )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.CORRUPT_DATA,
                reasons=["CPI values must be finite and positive"],
            )
        return ValidationResult(label=ValidationLabel.VALID)

    def normalize(
        self,
        artifact: RawArtifact,
        request: SourceRequest,
        validation: ValidationResult,
    ) -> list[EvidenceItem]:
        if validation.label is not ValidationLabel.VALID:
            raise ValueError("only VALID FRED artifacts can be normalized")
        date_field, rows = self._rows(artifact)
        requested = set(request.parameters["dates"])
        return [
            EvidenceItem(
                evidence_id=f"ev_{uuid4().hex}",
                run_id=artifact.run_id,
                task_id=artifact.step_id,
                metric="us_cpi_u_nsa",
                value=float(row[self.series]),
                unit="index_1982_1984_100",
                period=row[date_field][:7],
                source_name="FRED CPIAUCNS",
                source_url=artifact.request_url,
                artifact_id=artifact.artifact_id,
                retrieved_at=artifact.fetched_at,
                transformation=f"normalized {date_field} to date",
            )
            for row in rows
            if row[date_field] in requested
        ]
