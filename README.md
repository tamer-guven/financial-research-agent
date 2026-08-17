# Resilient Financial Research Agent

A financial research agent designed around a simple idea: **when one part of an investigation fails, successful work should not be repeated.**

This project was built for the LEC AI Engineering Intern assessment. Instead of attempting to support every possible financial question, it keeps the research scope intentionally narrow and focuses on the harder engineering problem: reliable recovery from partial failures.

The current investigation answers questions such as:

> **Did Apple's FY2024 revenue growth beat US inflation over a comparable period, and what was the global GDP-growth backdrop?**

The agent retrieves Apple revenue from SEC Company Facts, US CPI from FRED, and world GDP growth from the World Bank. If the FRED branch fails validation, it substitutes the preconfigured unregistered BLS V1 source without repeating successful SEC or World Bank work.

Author: [Tamer Guven](https://github.com/tamer-guven)

## What this project demonstrates

- typed, dependency-aware investigation plans;
- live official-source retrieval with durable SQLite checkpoints;
- raw-response persistence before validation;
- structural, semantic, and plausibility validation;
- bounded retry and source-substitution policies;
- deterministic financial calculations;
- optional OpenAI planning and synthesis with deterministic fallback;
- fail-closed, claim-level provenance verification;
- resume without repeating validated source requests.

## Supported research scope

The planner supports Apple/AAPL revenue growth compared with September-to-September US CPI inflation, with world-GDP-growth context, for FY2022, FY2023, or FY2024.

Reasonable paraphrases are accepted, including:

```text
Did Apple's FY2024 revenue growth beat US inflation, and what was the global GDP-growth backdrop?
Compare AAPL FY24 sales growth with CPI and give the world GDP backdrop.
Did Apple top-line growth in fiscal 2023 outpace consumer price inflation?
```

Unrelated companies, unsupported metrics, conflicting years, forecasts, investment advice, and stock-price questions are rejected explicitly.

## Architecture

```text
Question
  → deterministic scope and intent parsing
  → typed InvestigationPlan
  → semantic plan validation
  → dependency-aware sequential scheduler
      ├─ SEC Company Facts: Apple revenue
      ├─ World Bank Indicators: world GDP growth
      └─ FRED: US CPI
           └─ BLS V1 substitution when required
  → deterministic Decimal-based calculations
  → optional model synthesis
  → fail-closed provenance verification
  → persisted final report
```

The investigation is represented as a dependency graph rather than a single sequence of tool calls. Each source and calculation is an explicit task with declared dependencies. The current executor is sequential, but graph semantics allow it to preserve successful branches while repairing a failed one.

## Failure recovery

Suppose SEC revenue and World Bank GDP data have already been collected, but the FRED response is malformed. The agent does not restart the investigation:

1. Completed SEC and World Bank steps remain validated in SQLite.
2. The invalid FRED artifact is classified by the recovery policy.
3. The scheduler adds the configured BLS substitute for the CPI branch.
4. The calculation continues after the substitute is validated.
5. The final report records the recovery decision and its limitation.

Transient transport failures can be retried with bounded jitter. Schema or semantic failures are not retried indefinitely because an unchanged request is unlikely to repair unchanged invalid data.

The clock, sleeper, and retry-jitter RNG are injected, keeping recovery behaviour testable and deterministic where required.

## Raw data, validation, and checkpointing

A successful HTTP response is not automatically treated as trustworthy evidence.

SQLite persistence deliberately uses two committed transactions:

1. persist the complete raw response and request metadata;
2. reload the committed artifact, validate it, and persist normalized evidence.

If the process stops between those transactions, resume validates the already-persisted response instead of fetching it again. Fresh validated steps from the same persisted run are also reused with zero additional network calls.

Production source data is always obtained from the official internet endpoints when a fresh checkpoint is unavailable. There is no production offline transport and no embedded runtime financial dataset.

## Provenance and hallucination protection

Every substantive report claim cites one or more evidence IDs. Before a report is accepted, the provenance verifier checks that:

- every cited evidence item exists;
- every cited item has a `VALID` label;
- each decimal number in a claim belongs to its cited evidence;
- the direct answer introduces no number absent from verified claims.

Raw Apple revenue remains in SQLite and in the transformation lineage, but the synthesizer receives only reportable calculated metrics:

- `apple_revenue_growth`;
- `us_cpi_inflation`;
- `revenue_growth_minus_inflation`;
- `world_gdp_growth`.

Values are supplied as exact two-decimal display strings with an allowed-values mapping by evidence ID. A rejected model report receives one repair attempt containing the precise validation failure. The repaired output must pass the same verifier; otherwise the system falls back deterministically.

## Failure demonstration

The canonical demo intentionally renames the expected FRED value column from `CPIAUCNS` to `BROKEN_CPI` after retrieval. This simulates a technically successful HTTP response containing unusable data.

The corrupted artifact is persisted, rejected as `INVALID_SCHEMA`, and replaced through BLS. Already-validated SEC and World Bank work is retained.

```cmd
uv run python -m research_agent demo --provider deterministic
```

Use `--fault none` when you want the normal investigation without injected corruption.

## Data sources

| Source | Purpose | Important validation |
|---|---|---|
| [SEC Company Facts](https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json) | Apple annual revenue | Apple CIK, explicit revenue concept, USD, fiscal year, `10-K` or `10-K/A`, latest fiscal-period end and filing |
| [FRED CPIAUCNS](https://fred.stlouisfed.org/series/CPIAUCNS) | US CPI-U, not seasonally adjusted | exact series, requested dates, numeric values; both `observation_date` and `DATE` normalize to the internal date field |
| [BLS Public Data API](https://www.bls.gov/developers/) | CPI fallback delivery path | V1 POST, series `CUUR0000SA0`, exact September periods |
| [World Bank Indicators](https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG?format=json) | World GDP growth | world geography, indicator code, requested years, numeric values |

FRED republishes the BLS-originated CPI series. BLS therefore improves availability but is not independent confirmation of the underlying inflation measurement.

## Research influences

The implementation is intentionally smaller than the systems in these papers, but several ideas influenced its design.

### [LLMCompiler — An LLM Compiler for Parallel Function Calling](https://arxiv.org/abs/2312.04511)

LLMCompiler influenced the dependency-aware task representation. This project does not currently execute branches in parallel, but the graph structure makes successful work independently retainable when another branch fails.

### [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884)

CRAG influenced the retrieve-evaluate-correct pattern. Because these sources expose structured financial data, this project uses deterministic validation rather than asking a model to decide whether a response looks reliable.

### [Resilient Distributed Datasets](https://www.usenix.org/conference/nsdi12/technical-sessions/presentation/zaharia)

RDDs influenced the decision to retain useful intermediate results so unrelated work does not need to be recomputed after failure. Here, successful source artifacts and normalized evidence are persisted as SQLite checkpoints.

### [ReAct — Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

ReAct provided background for the plan-action-observation cycle. This project adds persistent state, deterministic data validation, and enforceable recovery rather than relying exclusively on a model reasoning loop.

## Installation

Requirements:

- Python 3.12;
- [uv](https://docs.astral.sh/uv/);
- internet access for official financial sources.

Install the locked dependencies:

```cmd
uv sync --locked
```

## Configuration

Configuration is read from environment variables. The application does not load `.env` automatically.

| Variable | Required | Purpose |
|---|---|---|
| `SEC_USER_AGENT` | Yes | Identifies the application and provides a real contact address for SEC requests |
| `OPENAI_API_KEY` | Only for `--provider openai` | Enables OpenAI planning and synthesis |
| `OPENAI_MODEL` | No | Overrides the default OpenAI model |
| `RESEARCH_AGENT_DB` | No | Overrides the SQLite database path |

Windows CMD example:

```cmd
set "SEC_USER_AGENT=resilient-financial-research-agent/1.0 YOUR_REAL_EMAIL"
set "OPENAI_API_KEY=YOUR_REAL_OPENAI_API_KEY"
set "OPENAI_MODEL=gpt-5.6-luna"
```

Never commit real credentials or personal contact values.

## Running the project

Run the supported question with deterministic planning and reporting:

```cmd
uv run python -m research_agent run --question "Did Apple's FY2024 revenue growth beat US inflation over a comparable period, and what was the global GDP-growth backdrop?" --provider deterministic --fault none
```

Run the same investigation with OpenAI planning and synthesis:

```cmd
uv run python -m research_agent run --question "Did Apple's FY2024 revenue growth beat US inflation over a comparable period, and what was the global GDP-growth backdrop?" --provider openai --fault none
```

The CLI displays the requested provider and model, the planner and synthesizer actually used, and a prominent warning for every `MODEL_FALLBACK` event. API keys are never displayed.

Inspect or resume a persisted run:

```cmd
uv run python -m research_agent inspect RUN_ID
uv run python -m research_agent resume RUN_ID --provider openai
```

## Testing

Run the complete unit and integration-style test suite:

```cmd
uv run pytest
```

The ordinary suite mocks external services. A live OpenAI contract test is opt-in and runs only when both the explicit flag and API key are present:

```cmd
set "RUN_OPENAI_LIVE_TEST=1"
uv run pytest -m live_openai
```

## Current limitations

- Only Apple/AAPL is supported.
- Supported target years are FY2022, FY2023, and FY2024.
- CPI uses September-to-September CPI-U as an approximate fiscal-period alignment.
- Revenue versus CPI is a nominal comparison, not a complete real-revenue calculation.
- World GDP growth is context, not evidence that explains Apple performance.
- BLS restores CPI availability but does not add independent lineage.
- Execution is sequential even though some graph branches are independent.
- The fallback catalogue is intentionally small.
- This project provides factual research, not investment advice.

## What I would do next

The next step would be to generalize the intent parser, typed plan contracts, and source registry so the same recovery engine could support more companies, financial metrics, and time periods.

After that, I would add genuinely independent fallback sources, execute independent graph branches concurrently, expand failure scenarios, and build a small interface for inspecting each claim's provenance chain.

The core boundary would remain unchanged: models can make the system more flexible, but data validation, calculations, recovery decisions, and provenance should remain enforceable by deterministic code.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
