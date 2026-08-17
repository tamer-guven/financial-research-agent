"""Rich/Typer command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_agent.application import ApplicationResult, ResearchApplication
from research_agent.clock import AsyncioSleeper, SystemClock, SystemRandomSource
from research_agent.config import Settings
from research_agent.executor import SourceExecutor
from research_agent.fault_injection import FaultInjector
from research_agent.intent import CANONICAL_QUESTION, UnsupportedQuestionError
from research_agent.providers import (
    DeterministicDemoProvider,
    ResilientModelProvider,
    StructuredLLMProvider,
)
from research_agent.recovery import RecoveryPolicy
from research_agent.scheduler import SequentialScheduler
from research_agent.sources import BLSAdapter, FREDAdapter, SECAdapter, WorldBankAdapter
from research_agent.state import SQLiteRepository


app = typer.Typer(
    name="research-agent",
    help="Checkpointed corrective financial research agent.",
    no_args_is_help=False,
)
console = Console(width=140)


class UnavailableOpenAIProvider:
    def plan(self, question: str, *, year_override: int | None = None):  # type: ignore[no-untyped-def]
        raise RuntimeError("OPENAI_API_KEY is not configured")

    def synthesize(self, question: str, bundle):  # type: ignore[no-untyped-def]
        raise RuntimeError("OPENAI_API_KEY is not configured")


def _provider(settings: Settings, provider_name: str) -> ResilientModelProvider:
    deterministic = DeterministicDemoProvider()
    if provider_name == "deterministic":
        return ResilientModelProvider(
            deterministic=deterministic,
            requested_provider="deterministic",
        )
    if provider_name != "openai":
        raise typer.BadParameter("provider must be deterministic or openai")
    primary = (
        StructuredLLMProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        if settings.openai_api_key
        else UnavailableOpenAIProvider()
    )
    return ResilientModelProvider(
        deterministic=deterministic,
        primary=primary,
        requested_provider="openai",
        requested_model=settings.openai_model,
    )


async def _application(
    *,
    settings: Settings,
    provider_name: str,
    fault: str | None,
) -> tuple[ResearchApplication, httpx.AsyncClient]:
    clock = SystemClock()
    repository = SQLiteRepository(settings.database_path)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
        headers={"Accept": "application/json, text/csv;q=0.9, */*;q=0.1"},
    )
    executor = SourceExecutor(
        repository=repository,
        adapters={
            "sec": SECAdapter(settings.sec_user_agent),
            "world_bank": WorldBankAdapter(),
            "fred": FREDAdapter(),
            "bls": BLSAdapter(),
        },
        client=client,
        clock=clock,
        sleeper=AsyncioSleeper(),
        recovery_policy=RecoveryPolicy(random_source=SystemRandomSource(), clock=clock),
        fault_injector=FaultInjector(fault),
    )
    scheduler = SequentialScheduler(repository=repository, source_executor=executor, clock=clock)
    return (
        ResearchApplication(
            repository=repository,
            scheduler=scheduler,
            provider=_provider(settings, provider_name),
            clock=clock,
        ),
        client,
    )


def _render(repository: SQLiteRepository, result: ApplicationResult) -> None:
    run_id = result.engine.run_id
    console.print(Panel(f"[bold]Run ID:[/bold] {run_id}", title="Checkpointed Corrective DAG"))

    usage = result.model_usage
    usage_table = Table(title="Model usage", show_header=False)
    usage_table.add_column("Field", style="bold")
    usage_table.add_column("Value")
    usage_table.add_row("Requested provider", usage.requested_provider)
    usage_table.add_row("Requested model", usage.requested_model or "not requested")
    usage_table.add_row("Planner actually used", usage.planner_used)
    usage_table.add_row("Synthesizer actually used", usage.synthesizer_used)
    console.print(usage_table)

    for event in repository.list_events(run_id):
        if event["event_type"] == "MODEL_FALLBACK":
            payload = event["event_payload"]
            console.print(
                Panel(
                    f"[bold]Stage:[/bold] {payload['stage']}\n"
                    f"[bold]Reason:[/bold] {payload['reason']}",
                    title="[bold red]MODEL_FALLBACK — deterministic output used[/bold red]",
                    border_style="red",
                )
            )

    steps = Table(title="Task state and retained work")
    for column in ("Step", "Source", "Status", "Attempts", "Network calls", "Recovery"):
        steps.add_column(column)
    for row in repository.list_steps(run_id):
        task = json.loads(row["task_json"])
        steps.add_row(
            row["step_id"],
            task["source"],
            row["status"],
            str(row["attempt_count"]),
            str(row["network_call_count"]),
            row["recovery_action"] or "-",
        )
    console.print(steps)

    evidence_table = Table(title="Validated evidence and provenance")
    for column in ("Evidence ID", "Metric", "Period", "Value", "Unit", "Source"):
        evidence_table.add_column(column, overflow="fold")
    for item in result.engine.evidence:
        evidence_table.add_row(
            item.evidence_id,
            item.metric,
            item.period,
            f"{item.value:.2f}",
            item.unit,
            item.source_name,
        )
    console.print(evidence_table)

    if result.engine.recovery_decisions:
        for decision in result.engine.recovery_decisions:
            console.print(
                Panel(
                    f"Observation: {decision.observation}\n"
                    f"Classification: {decision.failure_class.value}\n"
                    f"Decision: {decision.selected_action.value}"
                    + (f" with {decision.replacement_source}" if decision.replacement_source else "")
                    + f"\nReason: {decision.justification}",
                    title="Recovery decision",
                )
            )

    report = result.report
    console.print(Panel(report.direct_answer, title=f"Final answer - {report.overall_confidence.value}"))
    for claim in report.claims:
        console.print(f"[bold]{claim.claim_id}[/bold]: {claim.text}")
        console.print(f"  evidence: {', '.join(claim.evidence_ids)}")
        if claim.limitation:
            console.print(f"  limitation: {claim.limitation}")
    console.print(f"[bold]Recovery summary:[/bold] {report.recovery_summary}")
    if report.missing_evidence:
        console.print(f"[bold]Missing evidence:[/bold] {', '.join(report.missing_evidence)}")


async def _run_command(
    *,
    question: str,
    database: Path,
    provider_name: str,
    fault: str | None,
    year_override: int | None = None,
) -> None:
    settings = Settings.from_environment(database)
    application, client = await _application(
        settings=settings,
        provider_name=provider_name,
        fault=fault,
    )
    try:
        result = await application.run(question, year_override=year_override)
        _render(application.repository, result)
    finally:
        await client.aclose()


@app.command()
def demo(
    fault: Annotated[str, typer.Option(help="Fault mode; use 'none' to disable.")] = "fred:corrupt_once",
    provider: Annotated[str, typer.Option(help="deterministic or openai")] = "deterministic",
    database: Annotated[Path, typer.Option("--db", help="SQLite state path.")] = Path("data/research-agent.sqlite3"),
) -> None:
    """Run the canonical, reproducible recovery demonstration."""
    fault_mode = None if fault.lower() == "none" else fault
    asyncio.run(
        _run_command(
            question=CANONICAL_QUESTION,
            database=database,
            provider_name=provider,
            fault=fault_mode,
            year_override=None,
        )
    )


@app.command("run")
def run_question(
    question: Annotated[str, typer.Option("--question", help="Supported financial-research question.")],
    year: Annotated[int | None, typer.Option("--year", help="Optional FY2022, FY2023, or FY2024 override.")] = None,
    fault: Annotated[str, typer.Option(help="Fault mode; use 'fred:corrupt_once' or 'none'.")] = "none",
    provider: Annotated[str, typer.Option(help="deterministic or openai")] = "deterministic",
    database: Annotated[Path, typer.Option("--db")] = Path("data/research-agent.sqlite3"),
) -> None:
    """Run a supported question or reasonable paraphrase."""
    try:
        asyncio.run(
            _run_command(
                question=question,
                database=database,
                provider_name=provider,
                fault=None if fault.lower() == "none" else fault,
                year_override=year,
            )
        )
    except UnsupportedQuestionError as exc:
        console.print(f"[red]Rejected:[/red] {exc}")
        raise typer.Exit(2) from exc


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Existing run ID.")],
    provider: Annotated[str, typer.Option(help="deterministic or openai")] = "deterministic",
    database: Annotated[Path, typer.Option("--db")] = Path("data/research-agent.sqlite3"),
) -> None:
    """Resume from durable checkpoints without repeating validated work."""

    async def execute() -> None:
        settings = Settings.from_environment(database)
        application, client = await _application(
            settings=settings,
            provider_name=provider,
            fault=None,
        )
        try:
            result = await application.resume(run_id)
            _render(application.repository, result)
        finally:
            await client.aclose()

    asyncio.run(execute())


@app.command("inspect")
def inspect_run(
    run_id: Annotated[str, typer.Argument(help="Existing run ID.")],
    database: Annotated[Path, typer.Option("--db")] = Path("data/research-agent.sqlite3"),
) -> None:
    """Inspect persisted steps, events, evidence, and the final report."""
    repository = SQLiteRepository(database)
    console.print(Panel(run_id, title="Persisted run"))
    for step in repository.list_steps(run_id):
        console.print(
            f"{step['step_id']}: {step['status']} | attempts={step['attempt_count']} | "
            f"network_calls={step['network_call_count']} | recovery={step['recovery_action'] or 'NONE'}"
        )
    for event in repository.list_events(run_id):
        console.print(f"{event['created_at']} {event['event_type']} {event['event_payload']}")
    report = repository.get_report(run_id)
    if report:
        console.print(Panel(report.direct_answer, title=f"Report - {report.overall_confidence.value}"))


@app.callback(invoke_without_command=True)
def default_demo(ctx: typer.Context) -> None:
    """Run the network-backed deterministic demo when no subcommand is supplied."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(demo)
