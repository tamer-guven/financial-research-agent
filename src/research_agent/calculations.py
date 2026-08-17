"""Deterministic calculations over validated evidence."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from research_agent.models import EvidenceItem, ValidationLabel


def percentage_growth(new: float, old: float) -> float:
    if old <= 0:
        raise ValueError("growth baseline must be positive")
    result = (Decimal(str(new)) / Decimal(str(old)) - Decimal(1)) * Decimal(100)
    return float(result)


def calculate_comparison(
    *,
    run_id: str,
    task_id: str,
    baseline_revenue: EvidenceItem,
    target_revenue: EvidenceItem,
    baseline_cpi: EvidenceItem,
    target_cpi: EvidenceItem,
    baseline_fiscal_year: int,
    target_fiscal_year: int,
    inflation_month: int,
    calculated_at: datetime,
    artifact_id: str = "calculation",
) -> list[EvidenceItem]:
    inputs = [baseline_revenue, target_revenue, baseline_cpi, target_cpi]
    if any(item.validation_label is not ValidationLabel.VALID for item in inputs):
        raise ValueError("calculations require VALID evidence")
    if target_fiscal_year != baseline_fiscal_year + 1:
        raise ValueError("target fiscal year must immediately follow the baseline")

    baseline_month = f"{baseline_fiscal_year}-{inflation_month:02d}"
    target_month = f"{target_fiscal_year}-{inflation_month:02d}"
    expected_periods = [
        f"FY{baseline_fiscal_year}",
        f"FY{target_fiscal_year}",
        baseline_month,
        target_month,
    ]
    if [item.period for item in inputs] != expected_periods:
        raise ValueError("calculation inputs do not match the requested periods")

    revenue_growth = percentage_growth(target_revenue.value, baseline_revenue.value)
    inflation = percentage_growth(target_cpi.value, baseline_cpi.value)
    difference = revenue_growth - inflation
    common = {
        "run_id": run_id,
        "task_id": task_id,
        "source_name": "deterministic_calculation",
        "source_url": "",
        "artifact_id": artifact_id,
        "retrieved_at": calculated_at,
        "validation_label": ValidationLabel.VALID,
    }
    return [
        EvidenceItem(
            evidence_id=f"ev_{uuid4().hex}",
            metric="apple_revenue_growth",
            value=revenue_growth,
            unit="percent",
            period=f"FY{baseline_fiscal_year}-FY{target_fiscal_year}",
            transformation="percentage_growth_decimal_v2",
            input_evidence_ids=[baseline_revenue.evidence_id, target_revenue.evidence_id],
            **common,
        ),
        EvidenceItem(
            evidence_id=f"ev_{uuid4().hex}",
            metric="us_cpi_inflation",
            value=inflation,
            unit="percent",
            period=f"{baseline_month}/{target_month}",
            transformation="percentage_growth_decimal_v2",
            input_evidence_ids=[baseline_cpi.evidence_id, target_cpi.evidence_id],
            **common,
        ),
        EvidenceItem(
            evidence_id=f"ev_{uuid4().hex}",
            metric="revenue_growth_minus_inflation",
            value=difference,
            unit="percentage_points",
            period=f"FY{target_fiscal_year} approximate alignment",
            transformation="difference_v1",
            input_evidence_ids=[item.evidence_id for item in inputs],
            **common,
        ),
    ]
