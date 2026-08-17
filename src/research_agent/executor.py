"""Checkpoint-aware source execution with deterministic recovery."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import httpx

from research_agent.clock import Clock, Sleeper
from research_agent.fault_injection import FaultInjector
from research_agent.models import (
    EvidenceItem,
    FailureClass,
    RawArtifact,
    RecoveryAction,
    RecoveryDecision,
    ResearchTask,
    SourceRequest,
    ValidationLabel,
    ValidationResult,
)
from research_agent.recovery import RecoveryPolicy
from research_agent.sources.base import BaseSourceAdapter
from research_agent.state.sqlite import SQLiteRepository


@dataclass
class StepOutcome:
    evidence: list[EvidenceItem]
    decision: RecoveryDecision | None = None


class SourceExecutor:
    def __init__(
        self,
        *,
        repository: SQLiteRepository,
        adapters: dict[str, BaseSourceAdapter],
        client: httpx.AsyncClient,
        clock: Clock,
        sleeper: Sleeper,
        recovery_policy: RecoveryPolicy,
        fault_injector: FaultInjector | None = None,
    ) -> None:
        self.repository = repository
        self.adapters = adapters
        self.client = client
        self.clock = clock
        self.sleeper = sleeper
        self.recovery_policy = recovery_policy
        self.fault_injector = fault_injector or FaultInjector()

    def _validate_committed(
        self,
        adapter: BaseSourceAdapter,
        artifact: RawArtifact,
        request: SourceRequest,
    ) -> tuple[ValidationResult, list[EvidenceItem]]:
        committed = self.repository.get_artifact(artifact.artifact_id)
        validation = adapter.validate(committed, request)
        evidence = adapter.normalize(committed, request, validation) if validation.label is ValidationLabel.VALID else []
        self.repository.persist_validation(committed, validation, evidence, self.clock.now())
        return validation, evidence

    def _decision(
        self,
        *,
        task: ResearchTask,
        failure_class: FailureClass,
        attempt_count: int,
        observation: str,
    ) -> RecoveryDecision:
        return self.recovery_policy.decide(
            observation=observation,
            failure_class=failure_class,
            attempt_count=attempt_count,
            fallback_source=task.fallback_source,
            critical=task.importance == "critical",
        )

    async def execute(self, run_id: str, task: ResearchTask) -> StepOutcome:
        if task.source not in self.adapters:
            raise KeyError(f"unregistered source: {task.source}")
        adapter = self.adapters[task.source]
        request = adapter.make_request(
            task.parameters.model_dump(mode="json"),
            self.clock.now().date().isoformat(),
        )

        existing_step = self.repository.get_step(run_id, task.id)
        if (
            existing_step["status"] == "FAILED"
            and existing_step["recovery_action"] == RecoveryAction.SUBSTITUTE.value
            and task.fallback_source
        ):
            decision = self._decision(
                task=task,
                failure_class=FailureClass(existing_step["failure_class"] or FailureClass.UNKNOWN.value),
                attempt_count=existing_step["attempt_count"],
                observation="reuse previously committed failure and recovery decision",
            )
            self.repository.record_event(
                run_id,
                task.id,
                "FAILED_CHECKPOINT_RETAINED",
                {"recovery_action": decision.selected_action.value},
                self.clock.now(),
            )
            return StepOutcome([], decision)

        if self.repository.validated_step_is_fresh(run_id, task.id, self.clock.now()):
            self.repository.record_event(
                run_id,
                task.id,
                "CHECKPOINT_RETAINED",
                {"network_calls_added": 0},
                self.clock.now(),
            )
            return StepOutcome(self.repository.evidence_for_step(run_id, task.id))

        pending_artifact = self.repository.latest_unvalidated_artifact(run_id, task.id)
        if pending_artifact is not None:
            validation, evidence = self._validate_committed(adapter, pending_artifact, request)
            if validation.label is ValidationLabel.VALID:
                return StepOutcome(evidence)
            step = self.repository.get_step(run_id, task.id)
            decision = self._decision(
                task=task,
                failure_class=validation.failure_class or FailureClass.UNKNOWN,
                attempt_count=step["attempt_count"],
                observation="persisted raw artifact failed validation during resume",
            )
            self.repository.record_recovery(run_id, task.id, decision, self.clock.now())
            if decision.selected_action is not RecoveryAction.RETRY:
                return StepOutcome([], decision)

        while True:
            self.repository.mark_fetching(run_id, task.id, self.clock.now())
            try:
                artifact = await adapter.fetch(
                    run_id=run_id,
                    step_id=task.id,
                    request=request,
                    client=self.client,
                    now=self.clock.now(),
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                step = self.repository.get_step(run_id, task.id)
                decision = self._decision(
                    task=task,
                    failure_class=FailureClass.TRANSIENT_TRANSPORT,
                    attempt_count=step["attempt_count"],
                    observation=f"transport exception: {type(exc).__name__}",
                )
                self.repository.record_recovery(run_id, task.id, decision, self.clock.now())
                if decision.selected_action is RecoveryAction.RETRY:
                    delay = self.recovery_policy.retry_delay(step["attempt_count"])
                    self.repository.record_event(run_id, task.id, "RETRY_SCHEDULED", {"delay_seconds": delay}, self.clock.now())
                    await self.sleeper.sleep(delay)
                    continue
                return StepOutcome([], decision)

            payload = self.fault_injector.apply(task.source, artifact.payload)
            if payload != artifact.payload:
                artifact = artifact.model_copy(
                    update={"payload": payload, "payload_sha256": hashlib.sha256(payload).hexdigest()}
                )
            self.repository.persist_raw_artifact(artifact)
            validation, evidence = self._validate_committed(adapter, artifact, request)
            if validation.label is ValidationLabel.VALID:
                return StepOutcome(evidence)

            step = self.repository.get_step(run_id, task.id)
            decision = self._decision(
                task=task,
                failure_class=validation.failure_class or FailureClass.UNKNOWN,
                attempt_count=step["attempt_count"],
                observation="; ".join(validation.reasons),
            )
            self.repository.record_recovery(run_id, task.id, decision, self.clock.now())
            if decision.selected_action is RecoveryAction.RETRY:
                retry_after = artifact.response_headers.get("retry-after")
                delay = self.recovery_policy.retry_delay(step["attempt_count"], retry_after)
                self.repository.record_event(run_id, task.id, "RETRY_SCHEDULED", {"delay_seconds": delay}, self.clock.now())
                await self.sleeper.sleep(delay)
                continue
            return StepOutcome([], decision)
