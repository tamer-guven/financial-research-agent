"""SQLite repository with an explicit raw-artifact commit boundary."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from research_agent.models import (
    EvidenceItem,
    FinalReport,
    InvestigationPlan,
    RawArtifact,
    RecoveryDecision,
    ResearchTask,
    StepStatus,
    ValidationLabel,
    ValidationResult,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    final_report_json TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    task_json TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    network_call_count INTEGER NOT NULL DEFAULT 0,
    failure_class TEXT,
    recovery_action TEXT,
    decision_reason TEXT,
    substitute_for_step_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    source TEXT NOT NULL,
    request_url TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status_code INTEGER,
    content_type TEXT,
    response_headers_json TEXT NOT NULL,
    payload BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL,
    validation_label TEXT,
    validation_json TEXT,
    FOREIGN KEY (run_id, step_id) REFERENCES steps(run_id, step_id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_lookup
    ON artifacts(run_id, request_fingerprint, valid_until);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    validation_label TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(artifact_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step_id TEXT,
    event_type TEXT NOT NULL,
    event_payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


def _json(model: Any) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    return json.dumps(model, sort_keys=True)


class SQLiteRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _event(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        step_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (f"evt_{uuid4().hex}", run_id, step_id, event_type, _json(payload), created_at.isoformat()),
        )

    def create_run(self, run_id: str, plan: InvestigationPlan, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (run_id, plan.question, plan.model_dump_json(), "RUNNING", now.isoformat(), now.isoformat()),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=None,
                event_type="RUN_CREATED",
                payload={"question": plan.question},
                created_at=now,
            )

    def get_plan(self, run_id: str) -> InvestigationPlan:
        with self._connect() as connection:
            row = connection.execute("SELECT plan_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return InvestigationPlan.model_validate_json(row["plan_json"])

    def add_step(
        self,
        run_id: str,
        task: ResearchTask,
        now: datetime,
        substitute_for: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO steps
                (run_id, step_id, task_json, status, created_at, updated_at, substitute_for_step_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    task.id,
                    task.model_dump_json(),
                    StepStatus.PENDING.value,
                    now.isoformat(),
                    now.isoformat(),
                    substitute_for,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=task.id,
                event_type="STEP_ADDED",
                payload={"source": task.source, "substitute_for": substitute_for},
                created_at=now,
            )

    def get_step(self, run_id: str, step_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM steps WHERE run_id = ? AND step_id = ?", (run_id, step_id)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown step: {run_id}/{step_id}")
        return dict(row)

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY created_at, rowid", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_fetching(self, run_id: str, step_id: str, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE steps SET status = ?, attempt_count = attempt_count + 1,
                network_call_count = network_call_count + 1, updated_at = ?
                WHERE run_id = ? AND step_id = ?""",
                (StepStatus.FETCHING.value, now.isoformat(), run_id, step_id),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=step_id,
                event_type="FETCH_STARTED",
                payload={},
                created_at=now,
            )

    def persist_raw_artifact(self, artifact: RawArtifact) -> None:
        """Transaction A: commit raw bytes before any validation starts."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO artifacts
                (artifact_id, run_id, step_id, source, request_url, request_fingerprint,
                 fetched_at, valid_until, status_code, content_type, response_headers_json,
                 payload, payload_sha256, validation_label, validation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.step_id,
                    artifact.source,
                    artifact.request_url,
                    artifact.request_fingerprint,
                    artifact.fetched_at.isoformat(),
                    artifact.valid_until.isoformat(),
                    artifact.status_code,
                    artifact.content_type,
                    _json(artifact.response_headers),
                    artifact.payload,
                    artifact.payload_sha256,
                ),
            )
            connection.execute(
                "UPDATE steps SET status = ?, updated_at = ? WHERE run_id = ? AND step_id = ?",
                (StepStatus.FETCHED.value, artifact.fetched_at.isoformat(), artifact.run_id, artifact.step_id),
            )
            self._event(
                connection,
                run_id=artifact.run_id,
                step_id=artifact.step_id,
                event_type="RAW_ARTIFACT_COMMITTED",
                payload={"artifact_id": artifact.artifact_id, "sha256": artifact.payload_sha256},
                created_at=artifact.fetched_at,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_artifact(self, artifact_id: str) -> RawArtifact:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return RawArtifact(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            source=row["source"],
            request_url=row["request_url"],
            request_fingerprint=row["request_fingerprint"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            valid_until=datetime.fromisoformat(row["valid_until"]),
            status_code=row["status_code"],
            content_type=row["content_type"],
            response_headers=json.loads(row["response_headers_json"]),
            payload=bytes(row["payload"]),
            payload_sha256=row["payload_sha256"],
        )

    def latest_unvalidated_artifact(self, run_id: str, step_id: str) -> RawArtifact | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT artifact_id FROM artifacts
                WHERE run_id = ? AND step_id = ? AND validation_label IS NULL
                ORDER BY fetched_at DESC LIMIT 1""",
                (run_id, step_id),
            ).fetchone()
        return self.get_artifact(row["artifact_id"]) if row else None

    def persist_validation(
        self,
        artifact: RawArtifact,
        validation: ValidationResult,
        evidence: list[EvidenceItem],
        now: datetime,
    ) -> None:
        """Transaction B: persist validation and evidence after raw commit."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE artifact_id = ?", (artifact.artifact_id,)
            ).fetchone()
            if exists is None:
                raise RuntimeError("raw artifact must be committed before validation")
            connection.execute(
                "UPDATE artifacts SET validation_label = ?, validation_json = ? WHERE artifact_id = ?",
                (validation.label.value, validation.model_dump_json(), artifact.artifact_id),
            )
            for item in evidence:
                connection.execute(
                    "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item.evidence_id,
                        item.run_id,
                        item.task_id,
                        item.artifact_id,
                        item.model_dump_json(),
                        item.validation_label.value,
                    ),
                )
            status = StepStatus.VALIDATED if validation.label is ValidationLabel.VALID else StepStatus.FAILED
            connection.execute(
                """UPDATE steps SET status = ?, failure_class = ?, updated_at = ?
                WHERE run_id = ? AND step_id = ?""",
                (
                    status.value,
                    validation.failure_class.value if validation.failure_class else None,
                    now.isoformat(),
                    artifact.run_id,
                    artifact.step_id,
                ),
            )
            self._event(
                connection,
                run_id=artifact.run_id,
                step_id=artifact.step_id,
                event_type="VALIDATION_COMMITTED",
                payload={"artifact_id": artifact.artifact_id, "label": validation.label.value},
                created_at=now,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def evidence_for_step(self, run_id: str, step_id: str) -> list[EvidenceItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM evidence WHERE run_id = ? AND step_id = ? ORDER BY rowid",
                (run_id, step_id),
            ).fetchall()
        return [EvidenceItem.model_validate_json(row["evidence_json"]) for row in rows]

    def list_evidence(self, run_id: str) -> list[EvidenceItem]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM evidence WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [EvidenceItem.model_validate_json(row["evidence_json"]) for row in rows]

    def validated_step_is_fresh(self, run_id: str, step_id: str, now: datetime) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT a.valid_until FROM artifacts a JOIN steps s
                ON a.run_id = s.run_id AND a.step_id = s.step_id
                WHERE s.run_id = ? AND s.step_id = ? AND s.status = ?
                AND a.validation_label = ? ORDER BY a.fetched_at DESC LIMIT 1""",
                (run_id, step_id, StepStatus.VALIDATED.value, ValidationLabel.VALID.value),
            ).fetchone()
        return bool(row and datetime.fromisoformat(row["valid_until"]) > now)

    def record_recovery(
        self,
        run_id: str,
        step_id: str,
        decision: RecoveryDecision,
        now: datetime,
    ) -> None:
        with self._connect() as connection:
            status = (
                StepStatus.PENDING.value
                if decision.selected_action.value == "RETRY"
                else StepStatus.FAILED.value
            )
            connection.execute(
                """UPDATE steps SET status = ?, recovery_action = ?, decision_reason = ?, updated_at = ?
                WHERE run_id = ? AND step_id = ?""",
                (
                    status,
                    decision.selected_action.value,
                    decision.justification,
                    now.isoformat(),
                    run_id,
                    step_id,
                ),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=step_id,
                event_type="RECOVERY_DECISION",
                payload=decision.model_dump(mode="json"),
                created_at=now,
            )

    def record_event(
        self,
        run_id: str,
        step_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        with self._connect() as connection:
            self._event(
                connection,
                run_id=run_id,
                step_id=step_id,
                event_type=event_type,
                payload=payload,
                created_at=now,
            )

    def persist_derived_evidence(
        self,
        run_id: str,
        step_id: str,
        artifact_id: str,
        evidence: list[EvidenceItem],
        now: datetime,
    ) -> None:
        payload = _json(
            {
                "formula_version": "financial_comparison_v1",
                "input_evidence_ids": sorted(
                    {evidence_id for item in evidence for evidence_id in item.input_evidence_ids}
                ),
            }
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO artifacts
                (artifact_id, run_id, step_id, source, request_url, request_fingerprint,
                 fetched_at, valid_until, status_code, content_type, response_headers_json,
                 payload, payload_sha256, validation_label, validation_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    run_id,
                    step_id,
                    "deterministic_calculation",
                    "",
                    digest,
                    now.isoformat(),
                    now.isoformat(),
                    200,
                    "application/json",
                    "{}",
                    payload,
                    digest,
                    ValidationLabel.VALID.value,
                    ValidationResult(label=ValidationLabel.VALID).model_dump_json(),
                ),
            )
            for item in evidence:
                connection.execute(
                    "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item.evidence_id,
                        run_id,
                        step_id,
                        artifact_id,
                        item.model_dump_json(),
                        ValidationLabel.VALID.value,
                    ),
                )
            connection.execute(
                "UPDATE steps SET status = ?, updated_at = ? WHERE run_id = ? AND step_id = ?",
                (StepStatus.VALIDATED.value, now.isoformat(), run_id, step_id),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=step_id,
                event_type="CALCULATION_COMMITTED",
                payload={"artifact_id": artifact_id, "evidence_count": len(evidence)},
                created_at=now,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_skipped(self, run_id: str, step_id: str, reason: str, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE steps SET status = ?, decision_reason = ?, updated_at = ?
                WHERE run_id = ? AND step_id = ?""",
                (StepStatus.SKIPPED.value, reason, now.isoformat(), run_id, step_id),
            )
            self._event(
                connection,
                run_id=run_id,
                step_id=step_id,
                event_type="STEP_SKIPPED",
                payload={"reason": reason},
                created_at=now,
            )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY rowid", (run_id,)
            ).fetchall()
        return [dict(row) | {"event_payload": json.loads(row["event_payload_json"])} for row in rows]

    def save_report(self, run_id: str, report: FinalReport, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, final_report_json = ?, updated_at = ? WHERE run_id = ?",
                ("COMPLETE", report.model_dump_json(), now.isoformat(), run_id),
            )
            report_step = connection.execute(
                "SELECT step_id FROM steps WHERE run_id = ? AND json_extract(task_json, '$.kind') = 'report'",
                (run_id,),
            ).fetchone()
            if report_step:
                connection.execute(
                    "UPDATE steps SET status = ?, updated_at = ? WHERE run_id = ? AND step_id = ?",
                    (StepStatus.VALIDATED.value, now.isoformat(), run_id, report_step["step_id"]),
                )
                self._event(
                    connection,
                    run_id=run_id,
                    step_id=report_step["step_id"],
                    event_type="REPORT_COMMITTED",
                    payload={"overall_confidence": report.overall_confidence.value},
                    created_at=now,
                )

    def get_report(self, run_id: str) -> FinalReport | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT final_report_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if not row or not row["final_report_json"]:
            return None
        return FinalReport.model_validate_json(row["final_report_json"])
