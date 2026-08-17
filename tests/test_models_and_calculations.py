from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from research_agent.calculations import calculate_comparison
from research_agent.models import (
    EvidenceItem,
    InvestigationPlan,
    ReportPeriodParameters,
    ResearchTask,
)


def evidence(metric: str, value: float, period: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev_{metric}_{period}",
        run_id="run_test",
        task_id="source",
        metric=metric,
        value=value,
        unit="index" if metric == "cpi" else "USD",
        period=period,
        source_name="fixture",
        source_url="https://example.com/source",
        artifact_id=f"artifact_{metric}_{period}",
    )


def test_calculation_expected_values() -> None:
    result = calculate_comparison(
        run_id="run_test",
        task_id="compare",
        baseline_revenue=evidence("revenue", 383_285_000_000, "FY2023"),
        target_revenue=evidence("revenue", 391_035_000_000, "FY2024"),
        baseline_cpi=evidence("cpi", 307.789, "2023-09"),
        target_cpi=evidence("cpi", 315.301, "2024-09"),
        baseline_fiscal_year=2023,
        target_fiscal_year=2024,
        inflation_month=9,
        calculated_at=datetime(2026, 8, 16, tzinfo=UTC),
    )
    values = {item.metric: item.value for item in result}
    assert values["apple_revenue_growth"] == pytest.approx(2.0219940775)
    assert values["us_cpi_inflation"] == pytest.approx(2.4406330311)
    assert values["revenue_growth_minus_inflation"] == pytest.approx(-0.4186389536)
    assert {item.period for item in result} == {
        "FY2023-FY2024",
        "2023-09/2024-09",
        "FY2024 approximate alignment",
    }


def test_plan_rejects_cycle() -> None:
    with pytest.raises(ValidationError, match="acyclic"):
        InvestigationPlan(
            question="test",
            tasks=[
                ResearchTask(
                    id="a",
                    objective="a",
                    source="report",
                    parameters=ReportPeriodParameters(
                        baseline_fiscal_year=2023,
                        target_fiscal_year=2024,
                        inflation_month=9,
                    ),
                    dependencies=["b"],
                    importance="critical",
                    kind="report",
                ),
                ResearchTask(
                    id="b",
                    objective="b",
                    source="report",
                    parameters=ReportPeriodParameters(
                        baseline_fiscal_year=2023,
                        target_fiscal_year=2024,
                        inflation_month=9,
                    ),
                    dependencies=["a"],
                    importance="critical",
                    kind="report",
                ),
            ],
        )


def test_plan_rejects_missing_dependency() -> None:
    with pytest.raises(ValidationError, match="missing dependencies"):
        InvestigationPlan(
            question="test",
            tasks=[
                ResearchTask(
                    id="calculation",
                    objective="calculate",
                    source="calculation",
                    parameters=ReportPeriodParameters(
                        baseline_fiscal_year=2023,
                        target_fiscal_year=2024,
                        inflation_month=9,
                    ),
                    dependencies=["missing_source"],
                    importance="critical",
                    kind="calculation",
                )
            ],
        )
