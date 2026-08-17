import asyncio
import csv
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from research_agent.fault_injection import FaultInjector
from research_agent.models import FailureClass, RawArtifact, ValidationLabel
from research_agent.sources import BLSAdapter, FREDAdapter, SECAdapter, WorldBankAdapter


FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 8, 16, tzinfo=UTC)


def artifact(source: str, payload: bytes, step: str = "step") -> RawArtifact:
    return RawArtifact(
        artifact_id=f"art_{source}",
        run_id="run_test",
        step_id=step,
        source=source,
        request_url="https://example.com/source",
        request_fingerprint="a" * 64,
        fetched_at=NOW,
        valid_until=NOW + timedelta(days=1),
        status_code=200,
        content_type="application/json" if source != "fred" else "text/csv",
        payload=payload,
        payload_sha256="b" * 64,
    )


def read(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_sec_selects_current_fiscal_period_from_comparative_10k_facts() -> None:
    adapter = SECAdapter("test contact@example.com")
    request = adapter.make_request({"years": [2021, 2022, 2023, 2024]}, "2026-08-16")
    raw = artifact("sec", read("sec_companyfacts_trimmed.json"), "sec_revenue")
    validation = adapter.validate(raw, request)
    assert validation.label is ValidationLabel.VALID
    evidence = adapter.normalize(raw, request, validation)
    assert [item.value for item in evidence] == [
        365_817_000_000.0,
        394_328_000_000.0,
        383_285_000_000.0,
        391_035_000_000.0,
    ]


def test_sec_accepts_10k_amendment_and_rejects_unrelated_form() -> None:
    payload = json.loads(read("sec_companyfacts_trimmed.json"))
    facts = payload["facts"]["us-gaap"][SECAdapter.concept]["units"]["USD"]
    current = next(fact for fact in facts if fact["fy"] == 2023 and fact["end"] == "2023-09-30")
    facts.append(
        current
        | {
            "form": "10-K/A",
            "filed": "2023-11-10",
            "accn": "synthetic-amendment-for-validator-test",
        }
    )
    facts.append(
        current
        | {
            "form": "8-K",
            "filed": "2023-11-11",
            "accn": "synthetic-rejected-form",
            "val": 1,
        }
    )
    adapter = SECAdapter("test contact@example.com")
    request = adapter.make_request({"years": [2023]}, "2026-08-16")
    raw = artifact("sec", json.dumps(payload).encode(), "sec_revenue")
    validation = adapter.validate(raw, request)
    assert validation.label is ValidationLabel.VALID
    evidence = adapter.normalize(raw, request, validation)
    assert evidence[0].value == 383_285_000_000.0
    assert "10-K/A" in (evidence[0].transformation or "")


@pytest.mark.parametrize("filename", ["fred_observation_date.csv", "fred_DATE.csv"])
def test_fred_normalizes_both_date_headers(filename: str) -> None:
    adapter = FREDAdapter()
    request = adapter.make_request(
        {"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]},
        "2026-08-16",
    )
    raw = artifact("fred", read(filename), "fred_cpi")
    validation = adapter.validate(raw, request)
    assert validation.label is ValidationLabel.VALID
    evidence = adapter.normalize(raw, request, validation)
    assert [(item.period, item.value) for item in evidence] == [
        ("2023-09", 307.789),
        ("2024-09", 315.301),
    ]


def test_exact_fred_corruption_is_parseable_invalid_schema_and_one_shot() -> None:
    original = read("fred_observation_date.csv")
    injector = FaultInjector("fred:corrupt_once")
    corrupted = injector.apply("fred", original)
    rows = list(csv.DictReader(io.StringIO(corrupted.decode())))
    assert rows[0]["BROKEN_CPI"] == "274.310"
    assert "CPIAUCNS" not in rows[0]
    assert injector.apply("fred", original) == original

    adapter = FREDAdapter()
    request = adapter.make_request(
        {"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]},
        "2026-08-16",
    )
    validation = adapter.validate(artifact("fred", corrupted), request)
    assert validation.label is ValidationLabel.INVALID
    assert validation.failure_class is FailureClass.INVALID_SCHEMA


def test_world_bank_and_bls_normalize_expected_periods() -> None:
    world_bank = WorldBankAdapter()
    world_request = world_bank.make_request({"years": ["2023", "2024"]}, "2026-08-16")
    world_raw = artifact("world_bank", read("world_bank_trimmed.json"))
    world_validation = world_bank.validate(world_raw, world_request)
    assert world_validation.label is ValidationLabel.VALID
    assert {item.period for item in world_bank.normalize(world_raw, world_request, world_validation)} == {"2023", "2024"}

    bls = BLSAdapter()
    bls_request = bls.make_request({"startyear": "2023", "endyear": "2024"}, "2026-08-16")
    method, url, kwargs = bls.http_request(bls_request)
    assert (method, url) == ("POST", "https://api.bls.gov/publicAPI/v1/timeseries/data/")
    assert kwargs["json"] == {
        "seriesid": ["CUUR0000SA0"],
        "startyear": "2023",
        "endyear": "2024",
    }
    bls_raw = artifact("bls", read("bls_v1_trimmed.json"))
    bls_validation = bls.validate(bls_raw, bls_request)
    assert bls_validation.label is ValidationLabel.VALID
    assert {item.period for item in bls.normalize(bls_raw, bls_request, bls_validation)} == {"2023-09", "2024-09"}


def test_adapter_fetch_builds_artifact() -> None:
    adapter = FREDAdapter()
    request = adapter.make_request(
        {"id": "CPIAUCNS", "dates": ["2023-09-01", "2024-09-01"]},
        "2026-08-16",
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=read("fred_DATE.csv"), headers={"content-type": "text/csv"})

    async def fetch() -> RawArtifact:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await adapter.fetch(
                run_id="run_test",
                step_id="fred_cpi",
                request=request,
                client=client,
                now=NOW,
            )

    raw = asyncio.run(fetch())
    assert raw.payload == read("fred_DATE.csv")
    assert raw.request_fingerprint
