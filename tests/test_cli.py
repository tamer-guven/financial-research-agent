from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import research_agent.cli as cli_module
from research_agent.cli import app


runner = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def mock_official_apis(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Keep CLI tests hermetic without adding a production offline mode."""

    payloads = {
        "data.sec.gov": ("sec_companyfacts_trimmed.json", "application/json"),
        "api.worldbank.org": ("world_bank_trimmed.json", "application/json"),
        "fred.stlouisfed.org": ("fred_observation_date.csv", "text/csv"),
        "api.bls.gov": ("bls_v1_trimmed.json", "application/json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        filename, content_type = payloads[request.url.host]
        return httpx.Response(
            200,
            content=(FIXTURES / filename).read_bytes(),
            headers={"content-type": content_type},
        )

    real_async_client = httpx.AsyncClient

    def test_client(*args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(cli_module.httpx, "AsyncClient", test_client)


def test_network_demo_is_secret_free_and_shows_recovery(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite3"
    result = runner.invoke(app, ["demo", "--db", str(database)])
    assert result.exit_code == 0, result.output
    assert "INVALID_SCHEMA" in result.output
    assert "SUBSTITUTE with bls" in result.output
    assert "Final answer - MEDIUM" in result.output
    assert "sec_revenue" in result.output
    assert "world_bank_gdp" in result.output
    assert "bls_cpi" in result.output


def test_cli_has_no_offline_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["demo", "--offline", "--db", str(tmp_path / "forbidden-offline.sqlite3")],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_openai_failure_is_prominent_and_shows_actual_fallback_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "demo",
            "--provider",
            "openai",
            "--fault",
            "none",
            "--db",
            str(tmp_path / "openai-fallback.sqlite3"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Requested provider" in result.output
    assert "openai" in result.output
    assert "Requested model" in result.output
    assert "Planner actually used" in result.output
    assert "Synthesizer actually used" in result.output
    assert result.output.count("MODEL_FALLBACK") == 2
    assert result.output.count("OPENAI_API_KEY is not configured") == 2
    assert "deterministic (fallback)" in result.output


def test_cli_accepts_paraphrase_and_rejects_unrelated_question(tmp_path: Path) -> None:
    accepted = runner.invoke(
        app,
        [
            "run",
            "--question",
            "Compare AAPL FY24 sales growth with CPI inflation and give the world GDP growth backdrop.",
            "--db",
            str(tmp_path / "accepted.sqlite3"),
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert "Final answer - MEDIUM" in accepted.output

    rejected = runner.invoke(
        app,
        [
            "run",
            "--question",
            "Should I buy Tesla?",
            "--db",
            str(tmp_path / "rejected.sqlite3"),
        ],
    )
    assert rejected.exit_code == 2
    assert "Rejected" in rejected.output


def test_cli_supports_dynamic_year_and_rejects_override_conflict(tmp_path: Path) -> None:
    accepted = runner.invoke(
        app,
        [
            "run",
            "--question",
            "Compare Apple FY2022 revenue growth with US CPI and provide global GDP context.",
            "--db",
            str(tmp_path / "fy2022.sqlite3"),
        ],
    )
    assert accepted.exit_code == 0, accepted.output
    assert "FY2022" in accepted.output
    assert "2021-09/2022-09" in accepted.output

    overridden = runner.invoke(
        app,
        [
            "run",
            "--question",
            "Did Apple's revenue growth beat CPI?",
            "--year",
            "2023",
            "--db",
            str(tmp_path / "override.sqlite3"),
        ],
    )
    assert overridden.exit_code == 0, overridden.output
    assert "FY2023" in overridden.output

    conflict = runner.invoke(
        app,
        [
            "run",
            "--question",
            "Did Apple's FY2023 revenue growth beat CPI?",
            "--year",
            "2024",
            "--db",
            str(tmp_path / "conflict.sqlite3"),
        ],
    )
    assert conflict.exit_code == 2
    assert "conflicts" in conflict.output
