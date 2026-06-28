"""Tool selection, efficiency, and knowledge base evaluators."""

import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from langfuse.experiment import Evaluation
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScoreDimension(BaseModel):
    score: int = Field(ge=1, le=5)
    comment: str


class ToolSelectionScore(BaseModel):
    appropriateness: ScoreDimension
    sequence_logic: ScoreDimension
    source_quality: ScoreDimension
    call_volume: ScoreDimension
    redundancy: ScoreDimension
    overall: ScoreDimension


_TOOL_SELECTION_SYSTEM_PROMPT = """\
You are evaluating the tool usage strategy of an AI research agent.

Score each dimension from 1 (very poor) to 5 (excellent).

## Scoring Dimensions

**Appropriateness (1-5):** Did the agent use the right tools for the question type?
- Web questions → should use google_search + web_fetch
- Structured data questions → may use fetch_file, grep_file, read_file
- Internal knowledge questions → should prefer vertex_search over google_search
- Never penalise vertex_search for skipping web_fetch — its grounding is built in

**SequenceLogic (1-5):** Are tools called in a sensible order?
- google_search should precede web_fetch for the same topic
- File pipeline (fetch_file → grep_file/read_file) must be in order
- vertex_search is self-contained — no follow-up fetch is expected or required

**SourceQuality (1-5):** Are the sources retrieved appropriate for this question type?
- Statistics/Data questions → government/institutional sources preferred
- Science questions → academic sources preferred
- General questions → reputable news/reference sources acceptable

**CallVolume (1-5):** Was the total number of tool calls appropriate for complexity?
- Simple questions: 2-4 calls efficient; 8+ wasteful
- Complex multi-part questions: 5-8 appropriate; 1-2 insufficient

**Redundancy (1-5):** Did the agent avoid duplicate or unnecessary calls?
- Fetching the same URL twice → low score
- Repeated searches on the same topic → lower score

**Overall (1-5):** Holistic assessment of tool strategy.
"""

_TOOL_SELECTION_USER_TEMPLATE = """\
## Question
{question}

## Tool Call Sequence ({total_calls} total calls)
{tool_sequence}

## Expected Tool Pattern
- Must use: {must_use}
- May use: {may_use}
- Data type: {data_type}
- Reference call count: {reference_count}
- vertex_search used: {vertex_search_used}

## Final Answer (excerpt)
{answer_excerpt}

Evaluate tool strategy on the six dimensions.
"""


async def evaluate_tool_selection(
    question: str,
    tool_calls: list[dict[str, Any]],
    final_answer: str,
    tool_pattern: dict[str, Any],
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    """Evaluate tool selection, sequence logic, and efficiency.

    Parameters
    ----------
    question : str
        The original question.
    tool_calls : list[dict]
        Sequence of tool calls with 'name' and 'args' keys.
    final_answer : str
        The agent's final answer text.
    tool_pattern : dict
        Expected tool pattern from dataset annotation (or auto-derived).
    model_config : LLMRequestConfig, optional
        LLM configuration.

    Returns
    -------
    list[Evaluation]
        Six Langfuse evaluations.
    """
    if model_config is None:
        model_config = LLMRequestConfig(temperature=0.0)

    tool_names = [tc.get("name", "unknown") for tc in tool_calls]
    vertex_search_used = "vertex_search" in tool_names

    sequence_lines = []
    for i, tc in enumerate(tool_calls, 1):
        name = tc.get("name", "unknown")
        args_str = str(tc.get("args", {}))
        args_preview = args_str[:80] + "..." if len(args_str) > 80 else args_str
        sequence_lines.append(f"{i}. {name}({args_preview})")
    tool_sequence = "\n".join(sequence_lines) if sequence_lines else "No tool calls"

    user_prompt = _TOOL_SELECTION_USER_TEMPLATE.format(
        question=question,
        total_calls=len(tool_calls),
        tool_sequence=tool_sequence,
        must_use=", ".join(tool_pattern.get("must_use", [])),
        may_use=", ".join(tool_pattern.get("may_use", [])),
        data_type=tool_pattern.get("data_type", "mixed"),
        reference_count=tool_pattern.get("reference_tool_call_count", "unknown"),
        vertex_search_used=vertex_search_used,
        answer_excerpt=final_answer[:300] + "..." if len(final_answer) > 300 else final_answer,
    )

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=model_config,
            system_prompt=_TOOL_SELECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=ToolSelectionScore,
        )
        score = completion.choices[0].message.parsed
        if score is None:
            return error_evaluations("Judge returned null response")

        return [
            Evaluation(name="ToolSelection/Appropriateness", value=float(score.appropriateness.score), comment=score.appropriateness.comment),
            Evaluation(name="ToolSelection/SequenceLogic", value=float(score.sequence_logic.score), comment=score.sequence_logic.comment),
            Evaluation(name="ToolSelection/SourceQuality", value=float(score.source_quality.score), comment=score.source_quality.comment),
            Evaluation(name="Efficiency/CallVolume", value=float(score.call_volume.score), comment=score.call_volume.comment),
            Evaluation(name="Efficiency/Redundancy", value=float(score.redundancy.score), comment=score.redundancy.comment),
            Evaluation(name="ToolSelection/Overall", value=float(score.overall.score), comment=score.overall.comment),
        ]
    except Exception as e:
        logger.exception("Tool selection evaluation failed: %s", e)
        return error_evaluations(str(e))


# ── Knowledge Base evaluator ──────────────────────────────────────────────────

class KnowledgeBaseScore(BaseModel):
    query_quality: ScoreDimension
    answer_grounded: int = Field(ge=0, le=1, description="1 if answer reflects KB content, 0 otherwise")
    answer_grounded_comment: str


_KB_SYSTEM_PROMPT = """\
You are evaluating whether an AI agent correctly used a private knowledge base tool (vertex_search).

Assess two things:
1. QueryQuality (1-5): Were the vertex_search queries specific and well-formed
   for retrieving the KB concepts needed to answer the question?
2. AnswerGrounded (0 or 1): Does the final answer reflect content from the
   knowledge base (based on the expected KB concepts)?

Return scores and brief comments for each.
"""


async def evaluate_knowledge_base_usage(
    question: str,
    tool_calls: list[dict[str, Any]],
    final_answer: str,
    kb_concepts: list[str],
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    """Evaluate whether the agent correctly used vertex_search for KB questions.

    Parameters
    ----------
    question : str
        The original question.
    tool_calls : list[dict]
        Sequence of tool calls from the agent.
    final_answer : str
        The agent's final answer.
    kb_concepts : list[str]
        Key concepts that should appear from the knowledge base.
    model_config : LLMRequestConfig, optional
        LLM configuration.

    Returns
    -------
    list[Evaluation]
        Three Langfuse evaluations: ToolUsed, QueryQuality, AnswerGrounded.
    """
    if model_config is None:
        model_config = LLMRequestConfig(temperature=0.0)

    tool_names = [tc.get("name", "unknown") for tc in tool_calls]
    vertex_calls = [tc for tc in tool_calls if tc.get("name") == "vertex_search"]
    tool_used = int(bool(vertex_calls))

    if not vertex_calls:
        return [
            Evaluation(name="KnowledgeBase/ToolUsed", value=0.0,
                       comment="Agent did not call vertex_search for a KB question"),
            Evaluation(name="KnowledgeBase/QueryQuality", value=0.0,
                       comment="Cannot evaluate — vertex_search was not called"),
            Evaluation(name="KnowledgeBase/AnswerGrounded", value=0.0,
                       comment="Cannot evaluate — vertex_search was not called"),
        ]

    vs_calls_text = "\n".join(
        f"- query: {tc.get('args', {}).get('query', str(tc.get('args', {})))}"
        for tc in vertex_calls
    )

    user_prompt = f"""\
## Question
{question}

## Expected KB Concepts
{", ".join(kb_concepts) if kb_concepts else "not specified"}

## vertex_search Calls Made
{vs_calls_text}

## Final Answer (excerpt)
{final_answer[:400]}

Evaluate query quality (1-5) and whether the answer is grounded in KB content (0 or 1).
"""

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=model_config,
            system_prompt=_KB_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=KnowledgeBaseScore,
        )
        score = completion.choices[0].message.parsed
        if score is None:
            return kb_error_evaluations("Judge returned null response")

        return [
            Evaluation(name="KnowledgeBase/ToolUsed", value=float(tool_used)),
            Evaluation(name="KnowledgeBase/QueryQuality", value=float(score.query_quality.score),
                       comment=score.query_quality.comment),
            Evaluation(name="KnowledgeBase/AnswerGrounded", value=float(score.answer_grounded),
                       comment=score.answer_grounded_comment),
        ]
    except Exception as e:
        logger.exception("Knowledge base evaluation failed: %s", e)
        return kb_error_evaluations(str(e))


def derive_tool_pattern(problem_category: str, answer_type: str) -> dict[str, Any]:
    """Auto-derive the expected tool pattern from dataset metadata."""
    may_use = ["fetch_file", "grep_file", "read_file", "vertex_search"]

    if problem_category == "Statistics & Data":
        return {
            "must_use": ["google_search"],
            "may_use": may_use,
            "requires_file_pipeline": False,
            "data_type": "structured",
            "reference_tool_call_count": 4,
        }
    if problem_category in ("Science & Technology", "Finance & Economics"):
        return {
            "must_use": ["google_search", "web_fetch"],
            "may_use": may_use,
            "requires_file_pipeline": False,
            "data_type": "mixed",
            "reference_tool_call_count": 5,
        }
    return {
        "must_use": ["google_search", "web_fetch"],
        "may_use": may_use,
        "requires_file_pipeline": False,
        "data_type": "web",
        "reference_tool_call_count": 3,
    }


def error_evaluations(error_msg: str) -> list[Evaluation]:
    """Return zero-value evaluations on tool selection grader failure."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="ToolSelection/Appropriateness", value=0.0, comment=comment),
        Evaluation(name="ToolSelection/SequenceLogic", value=0.0, comment=comment),
        Evaluation(name="ToolSelection/SourceQuality", value=0.0, comment=comment),
        Evaluation(name="Efficiency/CallVolume", value=0.0, comment=comment),
        Evaluation(name="Efficiency/Redundancy", value=0.0, comment=comment),
        Evaluation(name="ToolSelection/Overall", value=0.0, comment=comment),
    ]


def kb_error_evaluations(error_msg: str) -> list[Evaluation]:
    """Return zero-value KB evaluations on grader failure."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="KnowledgeBase/ToolUsed", value=0.0, comment=comment),
        Evaluation(name="KnowledgeBase/QueryQuality", value=0.0, comment=comment),
        Evaluation(name="KnowledgeBase/AnswerGrounded", value=0.0, comment=comment),
    ]
