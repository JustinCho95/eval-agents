# Customer Complaint Triage & Response Agent

## Project overview

Agentic AI system for financial institutions that handles incoming customer complaints
end-to-end. A 4-step pipeline: classify → RAG retrieval → decision reasoning → response
generation, with a human-in-the-loop (HITL) escalation gate when confidence falls below
threshold.

Built as part of Vector Institute's Agentic AI Bootcamp, adapting the Investigation,
Triage and Case Management reference implementation.

**LLM backbone:** Google Gemini 2.5 via Google ADK (`google.adk.agents`)
**Orchestrator / Decision model:** `gemini-2.5-pro`
**Classifier / Retriever / Responder model:** `gemini-2.5-flash`
**Vector store:** Vertex AI Search — `aieng/agent_evals/tools/vertex_search.py`
**Knowledge base:** Scotia Momentum Terms & Conditions PDF

---

## Repo structure

```
/
├── CLAUDE.md                   ← you are here
├── data/
│   ├── raw/                    ← source datasets (CFPB, Harper Valley)
│   ├── processed/              ← cleaned, labelled complaint records
│   └── synthetic/              ← generated edge-case samples
├── knowledge_base/
│   ├── scotia_tnc.pdf          ← T&C source document
│   └── ingest.py               ← chunk, embed, and index T&C PDF into Vertex AI Search
├── pipeline/
│   ├── base.py                 ← abstract base class all steps implement
│   ├── classifier.py           ← Step 1: complaint classification
│   ├── retriever.py            ← Step 2: Vertex AI Search RAG retrieval
│   ├── decision.py             ← Step 3: resolve vs escalate logic
│   ├── responder.py            ← Step 4: response generation
│   └── orchestrator.py         ← end-to-end pipeline runner
├── hitl/
│   └── escalation_queue.py     ← HITL gate + human review queue
├── tests/
│   └── ...
└── .env                        ← API keys (never commit)
```

---

## Build instruction: phase by phase

When I say **"build Phase N"**, implement only the components listed under that phase.
Do not scaffold future phases. Ask for confirmation before moving to the next phase.

---

## Phase 1 — Foundations & architecture (Weeks 1–2)

**Goal:** Repo skeleton, environment, config, and HITL gate design. No LLM calls yet.

### Components to build

| Component | File | What it does |
|-----------|------|--------------|
| Environment config | `.env.example` + extend `aieng/agent_evals/configs.py` | Add complaint-specific settings (confidence threshold, knowledge base path) to existing `pydantic BaseSettings` |
| Project scaffold | All directories + `__init__.py` files | Repo structure as above |
| HITL gate schema | `hitl/escalation_queue.py` | Dataclass for escalated complaints; queue interface stub |
| Logging | reuse `aieng/agent_evals/logging_config.py` | Import directly; do not create a new logger module |
| Base pipeline interface | `pipeline/base.py` | Abstract base class each pipeline step must implement |

### Conventions for this phase
- Extend `aieng/agent_evals/configs.py` (`pydantic BaseSettings`) for all new env vars — do not use raw `python-dotenv`
- Confidence threshold default: `0.75` (add as `CONFIDENCE_THRESHOLD` field)
- All pipeline steps must implement `run(input: dict) -> dict` from the base class
- Log every step input/output at DEBUG level for traceability
- No hardcoded credentials anywhere

### Done when
- Config loads without errors; directory structure matches the repo layout above
- HITL dataclass has fields: `complaint_id`, `complaint_text`, `category`, `confidence_score`, `reason`, `timestamp`

---

## Phase 2 — Data preparation & RAG setup (Weeks 3–4)

**Goal:** Ingest datasets, chunk and index the T&C PDF into Vertex AI Search.

### Components to build

| Component | File | What it does |
|-----------|------|--------------|
| CFPB data loader | `data/loaders/cfpb_loader.py` | Load + clean Kaggle/CFPB complaint CSVs |
| Synthetic data generator | `data/loaders/synthetic_generator.py` | Gemini-powered edge-case generation |
| T&C ingestion | `knowledge_base/ingest.py` | Chunk PDF → embed → index into Vertex AI Search |

### RAG retrieval
- Use `aieng/agent_evals/tools/vertex_search.py` for all T&C clause queries — do not implement a new retrieval client
- Chunk size: 512 tokens, overlap: 64 tokens
- Preserve section headers in each chunk's metadata
- Do not split mid-sentence
- Index metadata fields: `clause_text`, `section`, `category[]`, `page_number`, `source`

### Done when
- Vertex AI Search index has all T&C clauses; `knowledge_base/ingest.py --verify` reports 0 errors
- CFPB loader returns a clean DataFrame with at least `complaint_text`, `category`, `source` columns

---

## Phase 3 — Core agent implementation (Weeks 5–7)

**Goal:** Build and wire all 4 pipeline steps + HITL gate into a working end-to-end agent.

Use `google.adk.agents.LlmAgent` for single-step agents (classifier, retriever, responder)
and `google.adk.agents.Agent` with `PlanReActPlanner` for the orchestrator, consistent with
`implementations/aml_investigation/agent.py` and `implementations/knowledge_qa/agent.py`.

Wrap all pipeline steps with Langfuse spans using the tracing pattern in `aieng/agent_evals/langfuse.py`.

### Step 1 — Classifier (`pipeline/classifier.py`)

**Model:** `gemini-2.5-flash`
**Input:** `{ complaint_text: str }`
**Output (Pydantic model):** `{ category: str, severity: str, confidence: float, raw_response: str }`

- Use `google.adk.agents.LlmAgent` with structured output parsed via Pydantic model
- Categories: `billing | service_failure | fraud | account_access | product_dispute | other`
- Severity: `low | medium | high | critical`
- If structured parse fails, retry once, then return `category: "other"` with `confidence: 0.0`

### Step 2 — Retriever (`pipeline/retriever.py`)

**Input:** `{ complaint_text: str, category: str }`
**Output:** `{ clauses: list[dict], retrieval_scores: list[float] }`

- Use `aieng/agent_evals/tools/vertex_search.py` directly — do not re-implement retrieval
- Return top-3 clauses with `clause_text`, `section`, `score`
- Filter by `category` metadata when available
- Log retrieval scores at DEBUG level

### Step 3 — Decision reasoner (`pipeline/decision.py`)

**Model:** `gemini-2.5-pro` (enable `ThinkingConfig` for extended reasoning, same pattern as `implementations/knowledge_qa/agent.py`)
**Input:** `{ category: str, severity: str, confidence: float, clauses: list[dict] }`
**Output (Pydantic model):** `{ route: "resolve" | "escalate", reason: str, decision_confidence: float }`

- Escalate if classifier `confidence < CONFIDENCE_THRESHOLD`
- Escalate if severity is `critical` regardless of confidence
- Escalate if no relevant clauses retrieved (empty list)
- Otherwise resolve; pass to Step 4

### Step 4 — Responder (`pipeline/responder.py`)

**Model:** `gemini-2.5-flash`
**Input:** `{ complaint_text: str, category: str, clauses: list[dict] }`
**Output (Pydantic model):** `{ response_text: str, cited_clauses: list[str], grounded: bool }`

- Use `google.adk.agents.LlmAgent` to draft a customer-facing response
- Must cite at least one clause by section name
- `grounded: False` if no clause citations present in output
- Response must not exceed 300 words

### HITL gate (`hitl/escalation_queue.py`)

- Intercept after Step 3 if `route == "escalate"`
- Write escalated complaint to `escalation_queue` (list in memory, or file-backed for persistence)
- Return a holding response: `"Your complaint has been escalated to a specialist who will respond within 2 business days."`

### Orchestrator (`pipeline/orchestrator.py`)

```python
def run_pipeline(complaint_text: str) -> PipelineResult:
    # Step 1: classify
    # Step 2: retrieve
    # Step 3: decide → HITL gate if escalate
    # Step 4: respond (only if resolve)
    # Return full result with all intermediate outputs
```

- Use `aieng/agent_evals/async_client_manager.py` singleton for async client lifecycle
- `PipelineResult` Pydantic model must include all intermediate outputs for auditability
- Instrument total latency; log warning if > 15 seconds
- Catch and log exceptions per step; never crash silently

### Done when
- `python pipeline/orchestrator.py --complaint "I was double charged last month"` returns a full result
- An obviously fraudulent complaint triggers HITL escalation
- All steps log structured output at DEBUG level with Langfuse spans visible

---

## Phase 4 — Demo & documentation (Weeks 8–9)

**Goal:** Runnable demo, risk documentation, and production path.

### Components to build

| Component | File | What it does |
|-----------|------|--------------|
| CLI demo | `demo.py` | Interactive complaint input → full pipeline result printed |
| Risk & limitations doc | `docs/risks.md` | Regulatory exposure, edge cases, threshold sensitivity |
| Production notes | `docs/production_roadmap.md` | Monitoring, retraining, compliance review checklist |

### Demo script behaviour
- Accept complaint text from stdin or `--complaint` flag
- Print each step's output clearly labelled
- Show whether complaint was resolved or escalated
- Show cited T&C clauses (section names only)
- Show end-to-end latency

### Done when
- `python demo.py --complaint "..."` runs without errors on a fresh clone
- `docs/risks.md` covers at minimum: confidence threshold sensitivity, ambiguous T&C gaps,
  regulatory exposure from misclassification, hallucination risk

---

## Always-on conventions

- **Never commit** `.env` or any file containing API keys
- **Never call LLM steps** with unvalidated or unsanitized user input
- **Always return** intermediate step outputs in `PipelineResult` for auditability
- **Always log** at DEBUG level: step name, input summary, output summary, latency
- **Instrument all pipeline steps** with Langfuse spans (`aieng/agent_evals/langfuse.py`)
- **Use Pydantic models** for all structured step inputs/outputs (consistent with existing repo patterns)
- **Use type hints** on all function signatures
- **Docstring every public function** with Args / Returns / Raises
- Python version: **3.12+**
- Dependency management: **`pyproject.toml` + `uv`** (pin versions)
- Test files go in `/tests/` mirroring the source structure

---

## Key datasets

| Dataset | Path | Purpose |
|---------|------|---------|
| Bank Customer Complaint Analysis (Kaggle/CFPB) | `data/raw/cfpb_main.csv` | Primary complaint corpus |
| Consumer Complaints Dataset (CFPB) | `data/raw/cfpb_supplementary.csv` | Edge-case coverage |
| Scotia Momentum T&C PDF | `knowledge_base/scotia_tnc.pdf` | RAG knowledge base |
| Harper Valley Bank Call Corpus | `data/raw/harper_valley/` | Realistic banking language for synthetic generation |

---

## Success metrics (reference)

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 85% vs ground truth labels |
| T&C retrieval faithfulness | ≥ 80% faithfulness score |
| Routing accuracy | ≥ 85% correct resolve/escalate |
| Hallucination rate | < 5% ungrounded claims |
| HITL escalation precision | ≥ 90% correct low-confidence flagging |
| End-to-end latency | < 15 seconds per complaint |
