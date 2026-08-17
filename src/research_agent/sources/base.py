"""Shared HTTP adapter behavior and deterministic request fingerprints."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import httpx

from research_agent.models import (
    EvidenceItem,
    RawArtifact,
    SourceRequest,
    TaskParameters,
    ValidationResult,
)


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def request_fingerprint(
    request: SourceRequest,
    *,
    adapter_version: str,
    validator_version: str,
) -> str:
    material = "|".join(
        [
            request.source,
            request.endpoint,
            canonical_json(request.parameters),
            request.as_of_date,
            adapter_version,
            validator_version,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class BaseSourceAdapter(ABC):
    name: str
    endpoint: str
    ttl: timedelta
    adapter_version = "1"
    validator_version = "1"

    def make_request(
        self,
        parameters: dict[str, Any] | TaskParameters,
        as_of_date: str,
    ) -> SourceRequest:
        values = (
            parameters.model_dump(mode="json")
            if isinstance(parameters, TaskParameters)
            else parameters
        )
        return SourceRequest(
            source=self.name,
            endpoint=self.endpoint,
            parameters=values,
            as_of_date=as_of_date,
        )

    @abstractmethod
    def http_request(self, request: SourceRequest) -> tuple[str, str, dict[str, Any]]:
        """Return method, URL, and HTTPX keyword arguments."""

    async def fetch(
        self,
        *,
        run_id: str,
        step_id: str,
        request: SourceRequest,
        client: httpx.AsyncClient,
        now: datetime,
    ) -> RawArtifact:
        method, url, kwargs = self.http_request(request)
        response = await client.request(method, url, **kwargs)
        return self.artifact_from_response(
            run_id=run_id,
            step_id=step_id,
            request=request,
            response=response,
            now=now,
        )

    def artifact_from_response(
        self,
        *,
        run_id: str,
        step_id: str,
        request: SourceRequest,
        response: httpx.Response,
        now: datetime,
    ) -> RawArtifact:
        payload = response.content
        return RawArtifact(
            artifact_id=f"art_{uuid4().hex}",
            run_id=run_id,
            step_id=step_id,
            source=self.name,
            request_url=str(response.request.url),
            request_fingerprint=request_fingerprint(
                request,
                adapter_version=self.adapter_version,
                validator_version=self.validator_version,
            ),
            fetched_at=now,
            valid_until=now + self.ttl,
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            response_headers=dict(response.headers),
            payload=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def transport_validation(self, artifact: RawArtifact) -> ValidationResult | None:
        from research_agent.models import FailureClass, ValidationLabel

        if artifact.status_code in {401, 403}:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.AUTHENTICATION,
                reasons=[f"HTTP {artifact.status_code}"],
            )
        if artifact.status_code == 429:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.RATE_LIMITED,
                reasons=["HTTP 429"],
            )
        if artifact.status_code is None or artifact.status_code >= 500:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.TRANSIENT_TRANSPORT,
                reasons=[f"HTTP {artifact.status_code}"],
            )
        if artifact.status_code >= 400:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.UNKNOWN,
                reasons=[f"HTTP {artifact.status_code}"],
            )
        if not artifact.payload:
            return ValidationResult(
                label=ValidationLabel.INVALID,
                failure_class=FailureClass.CORRUPT_DATA,
                reasons=["empty response"],
            )
        return None

    @abstractmethod
    def validate(self, artifact: RawArtifact, request: SourceRequest) -> ValidationResult:
        ...

    @abstractmethod
    def normalize(
        self,
        artifact: RawArtifact,
        request: SourceRequest,
        validation: ValidationResult,
    ) -> list[EvidenceItem]:
        ...
