"""Deterministic recovery decisions and bounded full-jitter retries."""

from __future__ import annotations

from email.utils import parsedate_to_datetime

from research_agent.clock import Clock, RandomSource
from research_agent.models import FailureClass, RecoveryAction, RecoveryDecision


class RecoveryPolicy:
    def __init__(self, *, random_source: RandomSource, clock: Clock, max_retries: int = 2) -> None:
        self.random_source = random_source
        self.clock = clock
        self.max_retries = max_retries

    def decide(
        self,
        *,
        observation: str,
        failure_class: FailureClass,
        attempt_count: int,
        fallback_source: str | None,
        critical: bool,
    ) -> RecoveryDecision:
        transient = failure_class in {FailureClass.TRANSIENT_TRANSPORT, FailureClass.RATE_LIMITED}
        considered: list[RecoveryAction] = []
        if transient:
            considered.append(RecoveryAction.RETRY)
        if fallback_source:
            considered.append(RecoveryAction.SUBSTITUTE)
        considered.append(RecoveryAction.STOP if critical else RecoveryAction.PROCEED_PARTIAL)

        if transient and attempt_count <= self.max_retries:
            selected = RecoveryAction.RETRY
            replacement = None
            reason = f"{failure_class.value} is transient and retry budget remains"
        elif fallback_source:
            selected = RecoveryAction.SUBSTITUTE
            replacement = fallback_source
            reason = "unchanged retry is not justified; a preconfigured fallback is available"
        elif critical:
            selected = RecoveryAction.STOP
            replacement = None
            reason = "critical evidence is unavailable and cannot be fabricated"
        else:
            selected = RecoveryAction.PROCEED_PARTIAL
            replacement = None
            reason = "supporting evidence is unavailable; retain validated evidence and disclose the gap"

        return RecoveryDecision(
            observation=observation,
            failure_class=failure_class,
            transient=transient,
            considered_actions=considered,
            selected_action=selected,
            replacement_source=replacement,
            justification=reason,
        )

    def retry_delay(self, retry_number: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(retry_after)
                    return max(0.0, (parsed - self.clock.now()).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        cap = min(30.0, 0.5 * (2 ** max(0, retry_number - 1)))
        return self.random_source.uniform(0.0, cap)
