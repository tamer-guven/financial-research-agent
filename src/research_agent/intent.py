"""Bounded Apple intent parsing, deterministic planning, and semantic plan checks."""

from __future__ import annotations

import re

from research_agent.models import (
    FREDParameters,
    ReportPeriodParameters,
    SECParameters,
    SUPPORTED_YEARS,
    InvestigationPlan,
    ResearchIntent,
    ResearchTask,
    TaskParameters,
    WorldBankParameters,
    validate_task_graph,
)


CANONICAL_QUESTION = (
    "Did Apple's FY2024 revenue growth beat US inflation over a comparable period, "
    "and what was the global GDP-growth backdrop?"
)
APPLE_CIK = "0000320193"
APPLE_REVENUE_CONCEPT = "RevenueFromContractWithCustomerExcludingAssessedTax"
FRED_SERIES = "CPIAUCNS"
WORLD_BANK_COUNTRY = "WLD"
WORLD_BANK_INDICATOR = "NY.GDP.MKTP.KD.ZG"


class UnsupportedQuestionError(ValueError):
    pass


def _supported_scope() -> str:
    years = ", ".join(f"FY{year}" for year in SUPPORTED_YEARS)
    return (
        "Supported scope: Apple/AAPL revenue growth versus US CPI inflation, with world-GDP "
        f"context, for {years}."
    )


def _normalized(question: str) -> str:
    value = question.lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _year(value: str) -> int:
    parsed = int(value)
    return 2000 + parsed if parsed < 100 else parsed


def _extract_years(question: str) -> set[int]:
    text = question.lower().replace("’", "'")
    values: set[int] = set()
    for match in re.finditer(r"\bfy\s*'?((?:20)?\d{2})\b", text):
        values.add(_year(match.group(1)))
    for match in re.finditer(r"\bfiscal(?:\s+year)?\s+(20\d{2})\b", text):
        values.add(int(match.group(1)))
    for match in re.finditer(r"\b(20\d{2})\b", text):
        values.add(int(match.group(1)))
    return values


def parse_research_intent(
    question: str,
    *,
    year_override: int | None = None,
) -> ResearchIntent:
    """Extract critical parameters without delegating them to a model."""

    text = _normalized(question)
    if not text:
        raise UnsupportedQuestionError(_supported_scope())

    advice_terms = ("buy", "sell", "price target", "stock price", "forecast", "predict", "recommend")
    if any(term in text for term in advice_terms):
        raise UnsupportedQuestionError(
            "Investment advice, forecasts, and stock-price questions are not supported. " + _supported_scope()
        )

    words = set(text.split())
    supported_company = bool(words & {"apple", "aapl"})
    other_companies = {
        "tesla": "Tesla",
        "tsla": "Tesla",
        "microsoft": "Microsoft",
        "msft": "Microsoft",
        "amazon": "Amazon",
        "amzn": "Amazon",
        "google": "Google",
        "alphabet": "Alphabet",
        "googl": "Alphabet",
        "meta": "Meta",
        "nvidia": "NVIDIA",
        "nvda": "NVIDIA",
    }
    mentioned_other = sorted({name for marker, name in other_companies.items() if marker in words})
    if mentioned_other:
        raise UnsupportedQuestionError(
            f"Unsupported company: {', '.join(mentioned_other)}. This version supports only Apple/AAPL."
        )
    if not supported_company:
        raise UnsupportedQuestionError("The company must be Apple or AAPL. " + _supported_scope())

    revenue_terms = ("revenue", "sales", "top line")
    inflation_terms = ("inflation", "cpi", "consumer price")
    if not any(term in text for term in revenue_terms) or not any(term in text for term in inflation_terms):
        raise UnsupportedQuestionError(
            "Unsupported question type: ask for Apple revenue growth compared with US CPI inflation. "
            + _supported_scope()
        )

    years = _extract_years(question)
    if len(years) > 1:
        rendered = ", ".join(str(year) for year in sorted(years))
        raise UnsupportedQuestionError(
            f"Conflicting or ambiguous target years: {rendered}. Specify exactly one fiscal year."
        )
    question_year = next(iter(years), None)
    if year_override is not None and question_year is not None and year_override != question_year:
        raise UnsupportedQuestionError(
            f"The --year value FY{year_override} conflicts with FY{question_year} in the question."
        )
    target_year = year_override if year_override is not None else question_year
    if target_year is None:
        raise UnsupportedQuestionError("No unambiguous target fiscal year was found. " + _supported_scope())
    if target_year not in SUPPORTED_YEARS:
        raise UnsupportedQuestionError(f"FY{target_year} is outside the supported scope. " + _supported_scope())

    return ResearchIntent(
        company="Apple",
        ticker="AAPL",
        target_fiscal_year=target_year,
        baseline_fiscal_year=target_year - 1,
        inflation_month=9,
        question_kind="revenue_vs_inflation_with_gdp_context",
    )


def is_supported_question(question: str) -> bool:
    try:
        parse_research_intent(question)
    except (UnsupportedQuestionError, ValueError):
        return False
    return True


def build_plan(intent: ResearchIntent, *, question: str | None = None) -> InvestigationPlan:
    baseline = intent.baseline_fiscal_year
    target = intent.target_fiscal_year
    month = intent.inflation_month
    dates = [f"{baseline}-{month:02d}-01", f"{target}-{month:02d}-01"]
    question_text = question or (
        f"Did Apple's FY{target} revenue growth beat US inflation, and what was the global GDP-growth backdrop?"
    )
    return InvestigationPlan(
        question=question_text,
        intent=intent,
        tasks=[
            ResearchTask(
                id="sec_revenue",
                objective=f"Retrieve Apple FY{baseline} and FY{target} revenue",
                source="sec",
                parameters=SECParameters(
                    company=intent.company,
                    ticker=intent.ticker,
                    cik=APPLE_CIK,
                    concept=APPLE_REVENUE_CONCEPT,
                    years=[baseline, target],
                ),
                importance="critical",
            ),
            ResearchTask(
                id="world_bank_gdp",
                objective=f"Retrieve {baseline} and {target} world GDP growth",
                source="world_bank",
                parameters=WorldBankParameters(
                    country=WORLD_BANK_COUNTRY,
                    indicator=WORLD_BANK_INDICATOR,
                    years=[str(baseline), str(target)],
                ),
                importance="supporting",
            ),
            ResearchTask(
                id="fred_cpi",
                objective=f"Retrieve September {baseline} and September {target} US CPI",
                source="fred",
                parameters=FREDParameters(id=FRED_SERIES, dates=dates),
                importance="critical",
                fallback_source="bls",
            ),
            ResearchTask(
                id="compare",
                objective=f"Compare Apple FY{target} revenue growth with US inflation",
                source="calculation",
                parameters=ReportPeriodParameters(
                    baseline_fiscal_year=baseline,
                    target_fiscal_year=target,
                    inflation_month=month,
                ),
                dependencies=["sec_revenue", "fred_cpi"],
                importance="critical",
                kind="calculation",
            ),
            ResearchTask(
                id="final_report",
                objective=f"Produce a provenance-linked FY{target} final report",
                source="report",
                parameters=ReportPeriodParameters(
                    baseline_fiscal_year=baseline,
                    target_fiscal_year=target,
                    inflation_month=month,
                ),
                dependencies=["compare", "world_bank_gdp"],
                importance="critical",
                kind="report",
            ),
        ],
    )


def canonical_plan(question: str, *, year_override: int | None = None) -> InvestigationPlan:
    intent = parse_research_intent(question, year_override=year_override)
    return build_plan(intent, question=question)


def validate_plan_against_intent(plan: InvestigationPlan, intent: ResearchIntent) -> None:
    """Reject schema-valid plans whose meaning does not match the parsed intent."""

    validate_task_graph(plan.tasks)
    if plan.intent != intent:
        raise ValueError("plan intent does not match the deterministic question intent")
    if parse_research_intent(plan.question) != intent:
        raise ValueError("plan question does not match the deterministic question intent")

    supported_sources = {"sec", "fred", "world_bank"}
    source_tasks = [task for task in plan.tasks if task.kind == "source"]
    if {task.source for task in source_tasks} != supported_sources or len(source_tasks) != 3:
        raise ValueError("plan must contain exactly SEC, FRED, and World Bank source tasks")

    by_source = {task.source: task for task in source_tasks}
    calculations = [task for task in plan.tasks if task.kind == "calculation"]
    reports = [task for task in plan.tasks if task.kind == "report"]
    if len(calculations) != 1 or calculations[0].source != "calculation":
        raise ValueError("plan must contain exactly one deterministic calculation task")
    if len(reports) != 1 or reports[0].source != "report":
        raise ValueError("plan must contain exactly one report task")

    baseline = intent.baseline_fiscal_year
    target = intent.target_fiscal_year
    month = intent.inflation_month
    expected_dates = [f"{baseline}-{month:02d}-01", f"{target}-{month:02d}-01"]
    expected_sec = {
        "company": "Apple",
        "ticker": "AAPL",
        "cik": APPLE_CIK,
        "concept": APPLE_REVENUE_CONCEPT,
        "years": [baseline, target],
    }
    expected_world_bank = {
        "country": WORLD_BANK_COUNTRY,
        "indicator": WORLD_BANK_INDICATOR,
        "years": [str(baseline), str(target)],
    }
    def parameter_values(task: ResearchTask, label: str) -> dict[str, object]:
        if not isinstance(task.parameters, TaskParameters):
            raise ValueError(f"{label} task parameters are not a typed parameter contract")
        return task.parameters.model_dump(mode="json")

    if parameter_values(by_source["sec"], "SEC") != expected_sec:
        raise ValueError("SEC task parameters do not match the requested Apple fiscal years")
    if parameter_values(by_source["world_bank"], "World Bank") != expected_world_bank:
        raise ValueError("World Bank task parameters do not match the requested years")
    if parameter_values(by_source["fred"], "FRED") != {
        "id": FRED_SERIES,
        "dates": expected_dates,
    }:
        raise ValueError("FRED task parameters do not match the requested CPI dates")
    if by_source["fred"].fallback_source != "bls":
        raise ValueError("FRED task must configure the BLS fallback")

    calculation = calculations[0]
    expected_periods = {
        "baseline_fiscal_year": baseline,
        "target_fiscal_year": target,
        "inflation_month": month,
    }
    if parameter_values(calculation, "calculation") != expected_periods:
        raise ValueError("calculation periods do not match the requested intent")
    if set(calculation.dependencies) != {by_source["sec"].id, by_source["fred"].id}:
        raise ValueError("calculation must depend on SEC revenue and CPI evidence")

    report = reports[0]
    if parameter_values(report, "report") != expected_periods:
        raise ValueError("report periods do not match the requested intent")
    if set(report.dependencies) != {calculation.id, by_source["world_bank"].id}:
        raise ValueError("report must depend on the calculation and World Bank context")


def validate_supported_plan(plan: InvestigationPlan) -> None:
    intent = parse_research_intent(plan.question)
    validate_plan_against_intent(plan, intent)
