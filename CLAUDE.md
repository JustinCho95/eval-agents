# CLAUDE.md — Knowledge-Grounded QA Agent Evaluation Framework

## Project Overview

This project extends the existing Knowledge-Grounded QA Agent evaluation framework with:
1. A **custom knowledge base component** using Vertex AI Search (`vertex_search`)
2. New **offline evaluation metrics** for agentic behaviour
3. New **online evaluation metrics** for production monitoring
4. **Source validation** — system prompt updated to prioritize high-authority sources,
   with an offline grader that evaluates source credibility against a per-category hierarchy

The agent is built on Google ADK with PlanReAct planning and Gemini. All evaluation is
reported to Langfuse. Always reuse existing infrastructure before building new components.

---

## Repository Structure (relevant files)

```
implementations/knowledge_qa/
├── agent.py                  # KnowledgeGroundedAgent, AgentResponse, StepExecution
├── event_extraction.py       # extract_tool_calls, extract_final_response, etc.
├── plan_parsing.py           # ResearchPlan, ResearchStep, StepStatus, tag constants
├── system_instructions.py    # build_system_instructions()
├── token_tracker.py          # TokenTracker
├── retry.py                  # is_retryable_api_error, is_context_overflow_error
├── notebook.py               # run_with_display, display_response
└── evaluation/
    ├── deepsearchqa_grader.py # EXISTING: Outcome/F1/Precision/Recall
    ├── online.py              # EXISTING: report_final_response_score()
    ├── offline.py             # NEW: offline evaluation runner
    └── graders/
        ├── plan_quality.py     # NEW
        ├── tool_selection.py   # NEW
        ├── source_validation.py # NEW
        └── replanning.py       # NEW

aieng/agent_evals/
├── async_client_manager.py   # AsyncClientManager singleton
├── evaluation/
│   ├── types.py              # Evaluation dataclass
│   └── graders/
│       ├── _utils.py         # run_structured_parse_call()
│       └── config.py         # LLMRequestConfig
└── tools/
    └── vertex_search.py      # NEW: vertex_search + create_vertex_search_tool()
```

---

## Phase 1 — Custom Knowledge Base (vertex_search)

### What to build

Add `vertex_search` as an optional tool alongside the existing five tools. The agent
uses it when the question can be answered from a private data store rather than the
public web. This replaces the `google_search → web_fetch` pipeline for grounded
private-document queries.

### Key files to read first

- `agent.py` — `KnowledgeGroundedAgent.__init__()` to see how tools are wired in
- `system_instructions.py` — `SYSTEM_INSTRUCTIONS_TEMPLATE` to understand tool descriptions
- Notebook `04_custom_knowledge_base.ipynb` — reference implementation of `vertex_search`

### Implementation

**`aieng/agent_evals/tools/vertex_search.py`**

```python
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.adk.tools import FunctionTool

async def vertex_search(query: str) -> dict:
    """Query the private Vertex AI Search data store.

    Use this tool when the question is about internal documents,
    policies, or proprietary data stored in the knowledge base.
    Do NOT use for public web information — use google_search instead.

    Returns a grounded answer directly from retrieved document chunks.
    No separate fetch step is needed.

    Parameters
    ----------
    query : str
        The search query.

    Returns
    -------
    dict
        Keys: status, summary, sources (list of {title, uri}), source_count
    """
    # Use discoveryengine client with ADC (automatic on GCE)
    # Config read from environment: VERTEX_AI_DATASTORE_ID, GOOGLE_CLOUD_LOCATION
    ...

def create_vertex_search_tool() -> FunctionTool:
    """Wrap vertex_search as an ADK FunctionTool."""
    return FunctionTool(func=vertex_search)
```

**`agent.py` changes — add `vertex_search` as optional tool**

```python
# In KnowledgeGroundedAgent.__init__()
if config.vertex_datastore_id:
    self._vertex_search_tool = create_vertex_search_tool()
    tools.append(self._vertex_search_tool)
```

**`system_instructions.py` changes — add vertex_search description and source prioritization**

Add to `SYSTEM_INSTRUCTIONS_TEMPLATE` under `## Tools`:

```
**vertex_search**: Query the private knowledge base for internal documents,
policies, or proprietary data. Returns a grounded answer directly — no
separate fetch step needed. Use this BEFORE google_search when the question
may be answerable from internal documents.
```

Add a new `## Source Quality` section to `SYSTEM_INSTRUCTIONS_TEMPLATE`:

```
## Source Quality

**Always prefer high-authority sources.** When multiple sources are available,
prioritize them in this order:

1. **Peer-reviewed research** — academic journals, arXiv preprints, conference
   proceedings (ACL, NeurIPS, ICML, Nature, Science, NEJM, etc.)
2. **Government and institutional databases** — Statistics Canada, World Bank,
   IMF, CDC, WHO, national statistics offices, central banks
3. **Official organizational sources** — CMHC, regulatory bodies, standards
   organizations, established NGOs with primary data
4. **Reputable news and analysis** — Reuters, AP, BBC, Financial Times,
   established broadsheet newspapers — for factual reporting only
5. **Reference works** — encyclopedias, official documentation, technical
   standards (ISO, IEEE, RFC)

**Avoid as primary sources:** blogs, forums (Reddit, Quora), content farms,
aggregator sites, Wikipedia (acceptable as a pointer to primary sources only),
social media, and press releases without underlying data.

**When a high-authority source is unavailable:** state the limitation explicitly
in REASONING rather than substituting a low-authority source without disclosure.
Adjust confidence language accordingly.
```

### Environment variables required

```
VERTEX_AI_DATASTORE_ID=projects/.../locations/.../collections/.../dataStores/...
GOOGLE_CLOUD_LOCATION=us-central1
```

### Comparison: vertex_search vs google_search + web_fetch

| | google_search + web_fetch | vertex_search |
|---|---|---|
| Data source | Public web | Private data store |
| Steps to answer | Search → Fetch → Verify | Single call |
| Grounding | Agent reads fetched HTML | Vertex AI Search retrieves chunks |
| Out-of-scope | Searches web anyway | Returns "I don't know" |
| Verification compliance rule | Applies | Does not apply |

---

## Phase 2 — Offline Evaluation

Offline evaluation runs after agent execution against a labeled dataset.
All evaluations are reported to Langfuse via `Evaluation` objects.

### Always reuse

- `AsyncClientManager.get_instance()` — for LLM client access
- `run_structured_parse_call()` from `aieng/agent_evals/evaluation/graders/_utils.py`
- `Evaluation` from `aieng/agent_evals/evaluation/types.py`
- `LLMRequestConfig` from `aieng/agent_evals/evaluation/graders/config.py`
- `extract_tool_calls()`, `extract_event_text()` from `event_extraction.py`
- `extract_plan_text()`, `parse_plan_steps_from_text()` from `plan_parsing.py`
- `REPLANNING_TAG`, `PLANNING_TAG` constants from `plan_parsing.py`

---

### 2.1 Existing: Outcome / F1 / Precision / Recall

**Status:** Already implemented in `deepsearchqa_grader.py`. Do not modify.

**How it works:**
- `evaluate_deepsearchqa_async()` builds `DEEPSEARCHQA_GRADER_PROMPT` with the question,
  agent answer, ground truth, and answer type
- Evaluator LLM returns `Correctness Details` (bool per GT item) and `Excessive Answers`
- Deterministic math computes Precision, Recall, F1 from those counts
- Outcome category derived from set relationships
- Four `Evaluation` objects pushed to Langfuse: Outcome, F1, Precision, Recall

**Ground truth:** `DSQAExample.answer` + `DSQAExample.answer_type` — existing dataset,
no new annotation required.

**Error handling:** `error_evaluations()` returns four zero-value `Evaluation` objects
maintaining consistent shape if the grader LLM call fails.

---

### 2.2 New: Plan Quality

**File:** `implementations/knowledge_qa/evaluation/graders/plan_quality.py`

**What it evaluates:** Quality of the `ResearchPlan` produced before any tool is called.

**Input:**
- `ResearchPlan` object captured from the `/*PLANNING*/` block in the event stream
- `plan_rubric` from the dataset item (see Ground truth below)
- Original question

**Judge dimensions (each scored 1–5):**
- `Plan/Coverage` — do step descriptions address all `must_cover` concepts?
- `Plan/Decomposition` — appropriate breakdown for the stated complexity?
- `Plan/DependencyLogic` — do step `expected_output` values feed logically into dependents?
- `Plan/Synthesis` — is the synthesis step specific and actionable?
- `Plan/Overall` — overall plan quality

**Implementation pattern:**

```python
from pydantic import BaseModel
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation

class PlanQualityScore(BaseModel):
    coverage:         ScoreDimension
    decomposition:    ScoreDimension
    dependency_logic: ScoreDimension
    synthesis:        ScoreDimension
    overall:          ScoreDimension

class ScoreDimension(BaseModel):
    score:   int    # 1–5
    comment: str

async def evaluate_plan_quality(
    question:   str,
    plan:       ResearchPlan,
    plan_rubric: dict,
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    ...
```

**Langfuse scores pushed:** `Plan/Coverage`, `Plan/Decomposition`, `Plan/DependencyLogic`,
`Plan/Synthesis`, `Plan/Overall`

**Ground truth required per dataset item:**

```json
{
  "plan_rubric": {
    "min_steps": 3,
    "max_steps": 6,
    "must_cover": ["price trends", "verified source", "2020-2022 timeframe"],
    "must_have_synthesis": true,
    "complexity": "high"
  }
}
```

`must_have_synthesis` and `complexity` are auto-derivable:
- `must_have_synthesis`: always `true` for `Set Answer`, `true` for multi-part `Single Answer`
- `complexity`: `"low"` for Single Answer, `"medium"` for short Set Answer,
  `"high"` for Statistics & Data or multi-source categories

---

### 2.3 New: Tool Selection Appropriateness and Efficiency

**File:** `implementations/knowledge_qa/evaluation/graders/tool_selection.py`

**What it evaluates:** Whether the agent used the right tools in the right pattern,
and whether the total call count was reasonable. Combined into one evaluator since
both consume the same tool call sequence input.

**Input:**
- Full tool call sequence from `extract_tool_calls()` — name + args per call
- `tool_pattern` from the dataset item
- Original question and final answer

**Judge dimensions (each scored 1–5):**
- `ToolSelection/Appropriateness` — right tools for the `data_type`?
- `ToolSelection/SequenceLogic` — search before fetch, file pipeline in order?
- `ToolSelection/SourceQuality` — authoritative sources for this question type?
- `Efficiency/CallVolume` — total calls justified for question complexity?
- `Efficiency/Redundancy` — any duplicate or overly granular calls?
- `ToolSelection/Overall` — overall assessment

**Implementation pattern:**

```python
async def evaluate_tool_selection(
    question:     str,
    tool_calls:   list[dict],
    final_answer: str,
    tool_pattern: dict,
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    # Build formatted tool call sequence string for judge
    # Feed to run_structured_parse_call() with ToolSelectionScore schema
    # Return list of Evaluation objects
    ...
```

**Langfuse scores pushed:** `ToolSelection/Appropriateness`, `ToolSelection/SequenceLogic`,
`ToolSelection/SourceQuality`, `Efficiency/CallVolume`, `Efficiency/Redundancy`,
`ToolSelection/Overall`

**Ground truth required per dataset item:**

```json
{
  "tool_pattern": {
    "must_use": ["google_search", "web_fetch"],
    "may_use": ["fetch_file", "grep_file", "read_file", "vertex_search"],
    "requires_file_pipeline": false,
    "data_type": "mixed"
  },
  "reference_tool_call_count": 5
}
```

`must_use` and `data_type` are largely rule-derivable from `problem_category`.
`reference_tool_call_count` derived from `data_type`:
- `web` → 2–4, `structured` → 4–5, `mixed` → 5–8

**Note on vertex_search:** If `vertex_search` was used, `SequenceLogic` judge should
not penalise for missing `web_fetch` after it — grounding is built in. Include
`"vertex_search_used": true` in the judge prompt context when applicable.

---

### 2.4 New: Custom Knowledge Base Evaluation

**File:** `implementations/knowledge_qa/evaluation/graders/tool_selection.py`
(add as an additional evaluator alongside tool selection)

**What it evaluates:** When the question involves private/internal knowledge, did the
agent correctly reach for `vertex_search` rather than the public web?

**Input:**
- Full tool call sequence from `extract_tool_calls()`
- Question
- Flag indicating whether the question is knowledge-base-answerable (from dataset annotation)

**Judge checks:**
- Did the agent call `vertex_search` when it should have?
- Did it avoid `google_search` for questions that should come from the knowledge base?
- Was the `vertex_search` query well-formed for the question?
- Did the final answer reflect content from the knowledge base sources?

**Langfuse scores pushed:** `KnowledgeBase/ToolUsed (0 or 1)`,
`KnowledgeBase/QueryQuality (1–5)`, `KnowledgeBase/AnswerGrounded (0 or 1)`

**Ground truth required per dataset item:**

```json
{
  "requires_knowledge_base": true,
  "kb_concepts": ["pricing tiers", "SLA uptime", "API rate limits"]
}
```

Only needed for questions where the answer lives in the private data store.
`requires_knowledge_base` defaults to `false` for all DeepSearchQA items —
only set `true` for items added from internal document evaluation sets.

---

### 2.5 New: Source Validation

**File:** `implementations/knowledge_qa/evaluation/graders/source_validation.py`

**What it evaluates:** Whether the sources the agent cited in `SOURCES:` are
high-authority for the question's domain. Enforces the source quality hierarchy
defined in the system prompt.

**Key files to read first:**
- `event_extraction.py` — `extract_sources_from_responses()`, `extract_grounding_sources()`
  already collect `GroundingChunk` objects (title + URI) during the run
- `notebook.py` — `_parse_response_sections()` extracts the `SOURCES` list from the
  final response text
- `agent.py` — `AgentResponse.sources` already holds resolved `GroundingChunk` objects

**Reuse:** `AgentResponse.sources` — already populated and URL-resolved by
`resolve_source_urls()` at the end of `answer_async()`. No additional extraction needed.

**Input:**
- `AgentResponse.sources` — list of `GroundingChunk(title, uri)` objects
- `source_rubric` from the dataset item (see Ground truth below)
- Original question and `problem_category`

**Judge dimensions (each scored 1–5):**
- `Source/AuthorityLevel` — are sources from tier 1–2 of the hierarchy (research,
  government, institutional)? Penalises blogs, forums, aggregators
- `Source/Relevance` — are the sources actually relevant to the specific question,
  or just topically adjacent?
- `Source/Diversity` — does the agent cite multiple independent sources, or rely
  on a single source for the entire answer?
- `Source/Overall` — overall source quality for this question

**Authority tier reference (included in judge prompt):**

```
Tier 1 — Research: academic journals, arXiv, conference proceedings
Tier 2 — Government/Institutional: national statistics offices, central banks,
          regulatory bodies, WHO, World Bank, CMHC, IMF
Tier 3 — Reputable reporting: Reuters, AP, BBC, Financial Times, established newspapers
Tier 4 — Reference: official documentation, technical standards
Tier 5 — Low authority: blogs, forums, Wikipedia, aggregators, press releases
```

**Implementation pattern:**

```python
from pydantic import BaseModel
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from aieng.agent_evals.tools import GroundingChunk

class SourceValidationScore(BaseModel):
    authority_level: ScoreDimension  # 1–5
    relevance:       ScoreDimension  # 1–5
    diversity:       ScoreDimension  # 1–5
    overall:         ScoreDimension  # 1–5

async def evaluate_source_validation(
    question:        str,
    problem_category: str,
    sources:         list[GroundingChunk],  # from AgentResponse.sources
    source_rubric:   dict,
    model_config:    LLMRequestConfig | None = None,
) -> list[Evaluation]:
    # Format sources as numbered list with title + URI for judge
    # Include authority tier reference and category-specific expectations
    # Feed to run_structured_parse_call() with SourceValidationScore schema
    ...

@staticmethod
def error_evaluations(error_msg: str) -> list[Evaluation]:
    """Four zero-value evaluations — maintains consistent Langfuse shape."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="Source/AuthorityLevel", value=0.0, comment=comment),
        Evaluation(name="Source/Relevance",      value=0.0, comment=comment),
        Evaluation(name="Source/Diversity",      value=0.0, comment=comment),
        Evaluation(name="Source/Overall",        value=0.0, comment=comment),
    ]
```

**Judge prompt structure:**

```
You are evaluating the quality of sources used by an AI research agent.

## Question
{question}

## Problem Category
{problem_category}

## Expected Source Types
{source_rubric["expected_tiers"]}  # e.g. ["Tier 1", "Tier 2"]
{source_rubric["domain_examples"]} # e.g. ["Statistics Canada", "CMHC", "Bank of Canada"]

## Authority Tier Reference
Tier 1 — Research: academic journals, arXiv, conference proceedings
Tier 2 — Government/Institutional: national statistics offices, central banks,
          regulatory bodies, WHO, World Bank
Tier 3 — Reputable reporting: Reuters, AP, BBC, Financial Times
Tier 4 — Reference: official documentation, technical standards
Tier 5 — Low authority: blogs, forums, Wikipedia, aggregators

## Sources Cited by Agent
{formatted_sources}  # numbered list: "1. [title] — uri"

## Evaluate on four dimensions: ...
```

**Langfuse scores pushed:** `Source/AuthorityLevel`, `Source/Relevance`,
`Source/Diversity`, `Source/Overall`

**Ground truth required per dataset item:**

```json
{
  "source_rubric": {
    "expected_tiers": ["Tier 1", "Tier 2"],
    "domain_examples": ["Statistics Canada", "CMHC", "Bank of Canada", "IMF"],
    "min_sources": 2,
    "allow_tier3_if_no_tier1_or_2": true
  }
}
```

`expected_tiers` and `domain_examples` are derivable from `problem_category`:

```python
CATEGORY_SOURCE_RUBRIC = {
    "Statistics & Data": {
        "expected_tiers": ["Tier 2"],
        "domain_examples": ["Statistics Canada", "World Bank", "IMF", "OECD", "national statistics offices"],
        "min_sources": 2,
        "allow_tier3_if_no_tier1_or_2": False,
    },
    "Science & Technology": {
        "expected_tiers": ["Tier 1", "Tier 2"],
        "domain_examples": ["arXiv", "Nature", "IEEE", "ACM", "government research agencies"],
        "min_sources": 2,
        "allow_tier3_if_no_tier1_or_2": True,
    },
    "Politics & Government": {
        "expected_tiers": ["Tier 2", "Tier 3"],
        "domain_examples": ["official government sites", "Reuters", "AP", "BBC"],
        "min_sources": 2,
        "allow_tier3_if_no_tier1_or_2": True,
    },
    "History & Culture": {
        "expected_tiers": ["Tier 1", "Tier 4"],
        "domain_examples": ["academic journals", "museum databases", "official archives"],
        "min_sources": 1,
        "allow_tier3_if_no_tier1_or_2": True,
    },
    # Default for uncategorized
    "_default": {
        "expected_tiers": ["Tier 1", "Tier 2", "Tier 3"],
        "domain_examples": [],
        "min_sources": 1,
        "allow_tier3_if_no_tier1_or_2": True,
    },
}
```

This means `source_rubric` can be auto-generated from `problem_category` for the
entire DeepSearchQA dataset with no manual annotation per item.

**Note on vertex_search:** Sources from `vertex_search` are already grounded in the
private data store — they should not be evaluated against the public authority tier
hierarchy. If `vertex_search` was the primary tool used, skip this evaluator or
score only the supplementary web sources.



**File:** `implementations/knowledge_qa/evaluation/graders/replanning.py`

**What it evaluates:** How many times did the agent replan mid-execution?
Fully deterministic — no LLM call required.

**Input:**
- Full event stream text via `extract_event_text()`
- `REPLANNING_TAG` constant from `plan_parsing.py`
- `max_replan_threshold` from dataset annotation (auto-derivable)

**What gets computed:**

```python
from .plan_parsing import REPLANNING_TAG, PLANNING_TAG
from .event_extraction import extract_event_text

def evaluate_replanning_rate(
    events: list,
    plan_steps: int,
    max_replan_threshold: int,
) -> list[Evaluation]:
    replan_count = sum(
        1 for event in events
        if REPLANNING_TAG in extract_event_text(event)
    )
    replan_flag  = int(replan_count > max_replan_threshold)
    replan_ratio = replan_count / max(plan_steps, 1)

    # Extract reasons from text after each REPLANNING tag
    replan_reasons = _extract_replan_reasons(events)

    return [
        Evaluation(name="Replanning/Count", value=replan_count),
        Evaluation(name="Replanning/Flag",  value=replan_flag),
        Evaluation(name="Replanning/Ratio", value=replan_ratio),
    ]
```

**Langfuse scores pushed:** `Replanning/Count`, `Replanning/Flag`, `Replanning/Ratio`

**Replan reasons** logged as trace metadata, not scores — qualitative only.

**Ground truth required:**

```
max_replan_threshold — fully auto-derivable:
  DSQAExample.answer_type == "Single Answer"  → 0
  DSQAExample.problem_category == "Statistics & Data" → 2
  All others → 1
```

No manual annotation required.

---

### 2.7 Offline Evaluation Runner

**File:** `implementations/knowledge_qa/evaluation/offline.py`

Orchestrates all offline evaluators against a Langfuse dataset.
Follow the same pattern as `implementations/report_generation/evaluation/offline.py`.

```python
async def evaluate(
    dataset_name: str,
    max_concurrency: int = 5,
) -> None:
    client_manager = AsyncClientManager.get_instance()
    langfuse_client = client_manager.langfuse_client
    dataset = langfuse_client.get_dataset(dataset_name)

    task = KnowledgeQATask()

    result = dataset.run_experiment(
        name="Evaluate Knowledge QA Agent",
        task=task.run,
        evaluators=[
            deepsearchqa_evaluator,    # existing — reuse as-is
            plan_quality_evaluator,    # new
            tool_selection_evaluator,  # new (includes efficiency + KB)
            source_validation_evaluator, # new
            replanning_evaluator,      # new
        ],
        max_concurrency=max_concurrency,
    )
```

Each evaluator follows the `async def evaluator(*, input, output, expected_output, **kwargs)`
signature required by Langfuse's `run_experiment`.

---

## Phase 3 — Online Evaluation

Online evaluation runs in production against live traffic.
All scores are pushed to Langfuse immediately via `create_score()` linked to `trace_id`.

### Always reuse

- `AsyncClientManager.get_instance().langfuse_client` for score reporting
- `extract_tool_calls()`, `extract_event_text()` from `event_extraction.py`
- `is_retryable_api_error()`, `is_context_overflow_error()` from `retry.py`
- `TokenTracker` — already instantiated in `KnowledgeGroundedAgent`
- `_parse_response_sections()` from `notebook.py` for coherence evaluation
- `REPLANNING_TAG` from `plan_parsing.py`

---

### 3.1 Existing: Valid Final Response

**Status:** Already implemented in `online.py` as `report_final_response_score()`.
Do not modify. Called within an active Langfuse trace when `event.is_final_response()`.

**How it works:**
- `EventParser.parse(event)` classifies the event
- Checks for `FINAL_RESPONSE` type and optional `string_match`
- Pushes `score(name="Valid Final Response", value=0/1)` to Langfuse

---

### 3.2 Existing: Latency and Token Usage

**Status:** `TokenTracker` is already instrumented in `KnowledgeGroundedAgent`.
`add_from_event()` is called in `_process_event()` on every event.

**Latency** — add timestamps around `runner.run_async()` in `_run_agent_once_inner()`.
Store per-phase durations in the `AgentResponse` or push directly to Langfuse.

**Reuse pattern:**
```python
# Already in AgentResponse
total_duration_ms: int  # already populated

# Already in TokenTracker
self._token_tracker.usage.latest_prompt_tokens
self._token_tracker.usage.context_used_percent
self._token_tracker.usage.total_tokens
```

---

### 3.3 New: Tool Call Volume and Errors

**File:** `implementations/knowledge_qa/evaluation/online.py`

**Reuse:** `extract_tool_calls()` and `extract_sources_from_responses()` already
called in `_process_event()`. Results are in `AgentResponse.tool_calls`.

```python
def report_tool_call_metrics(
    response: AgentResponse,
    trace_id: str,
    langfuse_client,
) -> None:
    tool_calls = response.tool_calls
    calls_by_tool = Counter(tc["name"] for tc in tool_calls)

    langfuse_client.create_score(
        name="ToolCalls/Total", value=len(tool_calls), trace_id=trace_id
    )
    # Error rate already detectable from tool responses with "error" key
    # in extract_sources_from_responses() — reuse existing error flag logic
```

---

### 3.4 New: Retry Rate

**Reuse:** `is_retryable_api_error()` and `is_context_overflow_error()` from `retry.py`
already used in `_run_agent_once()`. Add counters alongside existing retry logic.

```python
# Alongside existing retry logic in _run_agent_once()
self._rate_limit_retries += 1   # on is_retryable_api_error
self._overflow_resets += 1      # on is_context_overflow_error
```

Push to Langfuse at end of `answer_async()`.

---

### 3.5 New: Replanning Rate (Online)

**Reuse:** `REPLANNING_TAG` from `plan_parsing.py`. Already processed in
`_process_event_text_for_plan()`. Add a counter:

```python
# In _process_event_text_for_plan()
if REPLANNING_TAG in text:
    self._replan_count += 1   # add to KnowledgeGroundedAgent
```

Push to Langfuse at end of `answer_async()`.

---

### 3.6 New: Verification Compliance

**Reuse:** `AgentResponse.tool_calls` — already populated list of tool call dicts.

```python
def check_verification_compliance(tool_calls: list[dict]) -> tuple[int, list[int]]:
    """Return (compliant 0/1, list of violation positions)."""
    names = [tc["name"] for tc in tool_calls]
    violations = []
    for i, name in enumerate(names):
        if name == "google_search":
            subsequent = names[i+1:]
            next_search = next(
                (j for j, n in enumerate(subsequent) if n == "google_search"),
                len(subsequent)
            )
            if "web_fetch" not in subsequent[:next_search]:
                violations.append(i)
    return (0 if violations else 1, violations)
```

**Note:** `vertex_search` calls are exempt from this rule — grounding is built in.

---

### 3.7 New: Response Coherence (Async LLM Judge)

**Reuse:** `_parse_response_sections()` from `notebook.py` already splits the final
response into `answer`, `sources`, `reasoning`.

Run asynchronously after `answer_async()` returns — do not block the response path.

```python
async def evaluate_coherence_async(
    response_text: str,
    trace_id: str,
    langfuse_client,
) -> None:
    answer, sources, reasoning = _parse_response_sections(response_text)
    # Feed to evaluator LLM via run_structured_parse_call()
    # Push score(name="Coherence", value=0/1) to Langfuse
```

---

### 3.8 New: Follow-up Question Rate

**File:** `implementations/knowledge_qa/evaluation/online.py`

Track at session level. No ground truth required.

```python
def check_follow_up(
    time_since_response_ms: int,
    next_message: str,
    trace_id: str,
    langfuse_client,
    threshold_ms: int = 60_000,
) -> None:
    if time_since_response_ms < threshold_ms and _is_clarification(next_message):
        langfuse_client.create_score(
            name="UserFeedback/FollowUpRequired",
            value=1,
            trace_id=trace_id,
            comment=next_message[:100],
        )
```

---

## Evaluation Summary

### Offline

| Evaluation | Status | Method | Ground truth |
|---|---|---|---|
| Outcome / F1 / Precision / Recall | Existing | LLM-as-a-judge | `DSQAExample.answer` + `answer_type` |
| Plan quality | New | LLM-as-a-judge | `plan_rubric` — mostly auto-derivable |
| Tool selection and efficiency | New | LLM-as-a-judge | `tool_pattern` + `reference_tool_call_count` |
| Custom knowledge base | New | LLM-as-a-judge | `requires_knowledge_base` + `kb_concepts` |
| Source validation | New | LLM-as-a-judge | `source_rubric` — fully auto-derivable from `problem_category` |
| Replanning rate | New | Deterministic | `max_replan_threshold` — fully auto-derivable |

### Online

| Evaluation | Status | Method | Ground truth |
|---|---|---|---|
| Valid final response | Existing | Deterministic | None |
| Latency | Existing | Instrumentation | None |
| Token usage and cost | Existing | Instrumentation | None |
| Tool call volume and errors | New | Instrumentation | None |
| Retry rate | New | Instrumentation | None |
| Replanning rate | New | Deterministic | None |
| Verification compliance | New | Deterministic | None |
| Response coherence | New | LLM judge (async) | None |
| Follow-up question rate | New | Implicit session tracking | None |

---

## Success Metrics

### Minimum Goals
- F1 ≥ 0.70 on DeepSearchQA benchmark
- Replanning rate ≤ 1 per run on average
- Verification compliance ≥ 90%
- End-to-end latency ≤ 30s per question
- Context overflow rate ≤ 1% of production runs

### Stretch Goals
- F1 ≥ 0.85 on DeepSearchQA benchmark
- Replanning rate ≤ 0.3 per run on average
- Plan quality overall score ≥ 4/5 on high-complexity questions
- Tool call efficiency within 1.2x of reference count
- Follow-up question rate ≤ 10% per session

---

## Implementation Order

1. **`vertex_search` tool** — add to `tools/`, wire into `KnowledgeGroundedAgent`,
   update `system_instructions.py` (tools section + new Source Quality section)
2. **Offline runner** — scaffold `evaluation/offline.py` following report generation pattern
3. **Replanning rate** — deterministic, no LLM, lowest effort new offline metric
4. **Plan quality grader** — LLM judge on `ResearchPlan` object
5. **Tool selection and efficiency grader** — LLM judge on tool call sequence
6. **Custom knowledge base grader** — extend tool selection grader
7. **Source validation grader** — LLM judge on `AgentResponse.sources`;
   build `CATEGORY_SOURCE_RUBRIC` auto-derivation first to avoid manual annotation
8. **Online instrumentation** — tool call volume, retry rate, replanning rate, verification compliance
   (reuse existing `_process_event()` hooks — add counters only)
9. **Response coherence** — async LLM judge, reuse `_parse_response_sections()`
10. **Follow-up question rate** — session-level tracking
11. **Offline evaluation notebook** — `implementations/knowledge_qa/05_offline_evaluation.ipynb`;
    walks through each offline evaluator in isolation (single `AgentResponse`), then runs a full
    `run_experiment` with all offline evaluators wired up; follows the structure of `03_evaluation.ipynb`
12. **Online evaluation notebook** — `implementations/knowledge_qa/06_online_evaluation.ipynb`;
    demonstrates each online metric (verification compliance, tool call volume, replanning rate,
    retry rate, response coherence, follow-up rate) against a live agent run; shows how scores
    are pushed to Langfuse in real time

---

## Key Design Principles

- **Reuse before building.** `extract_tool_calls()`, `extract_event_text()`,
  `run_structured_parse_call()`, `AsyncClientManager`, `Evaluation` types — all exist.
- **LLM judge for quality, deterministic for counts.** Never use an LLM where
  counting tags or checking a rule suffices.
- **Async for online judges.** Response coherence runs after the response is
  delivered — never on the critical path.
- **vertex_search exemptions.** Verification compliance rule, file pipeline checks,
  and public source authority hierarchy do not apply to `vertex_search` calls —
  adjust all relevant evaluators.
- **Source rubric is auto-derivable.** `CATEGORY_SOURCE_RUBRIC` maps `problem_category`
  to expected authority tiers and domain examples — no per-item annotation needed.
  Only override for items where the category default is wrong.
- **Consistent Langfuse shape.** Every evaluator must return the same number of
  `Evaluation` objects on success and failure (follow `error_evaluations()` pattern
  from `deepsearchqa_grader.py`).
- **`reset()` between eval items.** Call `agent.reset()` between each dataset item
  to clear session history, plan state, and token tracker.
