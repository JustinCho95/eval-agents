"""Online evaluation metrics for the Knowledge-Grounded QA Agent.

These functions run against live agent responses in production.
All scores are pushed to Langfuse immediately via create_score().

Usage
-----
After agent.answer_async() returns:

    from aieng.agent_evals.knowledge_qa.evaluation.online import report_all_online_metrics
    report_all_online_metrics(response, trace_id=trace_id, langfuse_client=lf)
    # Coherence runs as a background asyncio task — does not block the caller.
"""

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.knowledge_qa.agent import AgentResponse
from langfuse import Langfuse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Tool Call Volume ──────────────────────────────────────────────────────────

def report_tool_call_metrics(
    response: AgentResponse,
    trace_id: str,
    langfuse_client: Langfuse,
) -> None:
    """Push tool call volume metrics to Langfuse.

    Parameters
    ----------
    response : AgentResponse
        The completed agent response.
    trace_id : str
        Langfuse trace ID to attach scores to.
    langfuse_client : Langfuse
        Langfuse client instance.
    """
    tool_calls = response.tool_calls
    calls_by_tool = Counter(tc.get("name", "unknown") for tc in tool_calls)

    langfuse_client.create_score(
        name="ToolCalls/Total",
        value=float(len(tool_calls)),
        trace_id=trace_id,
        comment=", ".join(f"{name}×{count}" for name, count in calls_by_tool.most_common()),
    )
    langfuse_client.create_score(
        name="ToolCalls/UniqueTools",
        value=float(len(calls_by_tool)),
        trace_id=trace_id,
    )


# ── Retry Rate ────────────────────────────────────────────────────────────────

def report_retry_metrics(
    response: AgentResponse,
    trace_id: str,
    langfuse_client: Langfuse,
) -> None:
    """Push retry rate metrics to Langfuse.

    Parameters
    ----------
    response : AgentResponse
        The completed agent response (retry counters populated by agent).
    trace_id : str
        Langfuse trace ID.
    langfuse_client : Langfuse
        Langfuse client instance.
    """
    langfuse_client.create_score(
        name="Retries/RateLimitRetries",
        value=float(response.rate_limit_retries),
        trace_id=trace_id,
    )
    langfuse_client.create_score(
        name="Retries/ContextOverflowResets",
        value=float(response.overflow_resets),
        trace_id=trace_id,
    )


# ── Replanning Rate ───────────────────────────────────────────────────────────

def report_replanning_metrics(
    response: AgentResponse,
    trace_id: str,
    langfuse_client: Langfuse,
) -> None:
    """Push online replanning metrics to Langfuse.

    Parameters
    ----------
    response : AgentResponse
        The completed agent response.
    trace_id : str
        Langfuse trace ID.
    langfuse_client : Langfuse
        Langfuse client instance.
    """
    plan_steps = len(response.plan.steps) if response.plan else 1
    replan_ratio = response.replan_count / max(plan_steps, 1)

    langfuse_client.create_score(
        name="Online/ReplanCount",
        value=float(response.replan_count),
        trace_id=trace_id,
    )
    langfuse_client.create_score(
        name="Online/ReplanRatio",
        value=round(replan_ratio, 4),
        trace_id=trace_id,
    )


# ── Verification Compliance ───────────────────────────────────────────────────

def check_verification_compliance(
    tool_calls: list[dict[str, Any]],
) -> tuple[int, list[int]]:
    """Check that every google_search is followed by a web_fetch before the next search.

    vertex_search calls are exempt — grounding is built in.

    Parameters
    ----------
    tool_calls : list[dict]
        Ordered tool calls from AgentResponse.

    Returns
    -------
    tuple[int, list[int]]
        (compliant 0/1, list of violation positions in the tool_calls sequence)
    """
    names = [tc.get("name", "") for tc in tool_calls]
    violations: list[int] = []

    for i, name in enumerate(names):
        if name == "google_search":
            subsequent = names[i + 1:]
            next_search_idx = next(
                (j for j, n in enumerate(subsequent) if n == "google_search"),
                len(subsequent),
            )
            if "web_fetch" not in subsequent[:next_search_idx]:
                violations.append(i)

    return (0 if violations else 1, violations)


def report_verification_compliance(
    response: AgentResponse,
    trace_id: str,
    langfuse_client: Langfuse,
) -> None:
    """Push verification compliance score to Langfuse.

    Parameters
    ----------
    response : AgentResponse
        The completed agent response.
    trace_id : str
        Langfuse trace ID.
    langfuse_client : Langfuse
        Langfuse client instance.
    """
    compliant, violations = check_verification_compliance(response.tool_calls)
    comment = "Compliant" if compliant else f"Violations at positions: {violations}"
    langfuse_client.create_score(
        name="Online/VerificationCompliance",
        value=float(compliant),
        trace_id=trace_id,
        comment=comment,
    )


# ── Response Coherence ────────────────────────────────────────────────────────

def _parse_response_sections(response_text: str) -> tuple[str, str, str]:
    """Split a final response into ANSWER, SOURCES, REASONING sections."""
    answer = sources = reasoning = ""

    answer_match = re.search(r"ANSWER:\s*", response_text, re.IGNORECASE)
    sources_match = re.search(r"SOURCES:\s*", response_text, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*", response_text, re.IGNORECASE)

    all_matches = sorted(
        [m for m in [answer_match, sources_match, reasoning_match] if m],
        key=lambda m: m.start(),
    )

    def _extract(match: re.Match | None) -> str:
        if not match:
            return ""
        start = match.end()
        end = next(
            (m.start() for m in all_matches if m.start() > start),
            len(response_text),
        )
        return response_text[start:end].strip()

    answer = _extract(answer_match)
    sources = _extract(sources_match)
    reasoning = _extract(reasoning_match)
    return answer, sources, reasoning


class CoherenceScore(BaseModel):
    coherent: int = Field(ge=0, le=1, description="1 if coherent, 0 if not")
    reason: str


_COHERENCE_SYSTEM_PROMPT = """\
You are evaluating the coherence of an AI research agent's response.

A response is coherent (1) when:
- The ANSWER directly addresses what was asked
- The REASONING explains how sources support the answer
- The SOURCES cited are consistent with claims in ANSWER and REASONING
- There are no internal contradictions

Return 0 if the response is incoherent or sections contradict each other.
"""


async def evaluate_coherence_async(
    response_text: str,
    trace_id: str,
    langfuse_client: Langfuse,
    model_config: LLMRequestConfig | None = None,
) -> None:
    """Async LLM judge for response coherence. Does not block the response path.

    Parameters
    ----------
    response_text : str
        The agent's final response text.
    trace_id : str
        Langfuse trace ID.
    langfuse_client : Langfuse
        Langfuse client instance.
    model_config : LLMRequestConfig, optional
        LLM configuration.
    """
    if model_config is None:
        model_config = LLMRequestConfig(temperature=0.0)

    answer, sources, reasoning = _parse_response_sections(response_text)

    user_prompt = f"""\
## ANSWER
{answer or "(empty)"}

## SOURCES
{sources or "(empty)"}

## REASONING
{reasoning or "(empty)"}

Is this response coherent? Return 1 (coherent) or 0 (incoherent) with a brief reason.
"""

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=model_config,
            system_prompt=_COHERENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=CoherenceScore,
        )
        score = completion.choices[0].message.parsed
        if score is None:
            logger.warning("Coherence judge returned null response for trace %s", trace_id)
            return

        langfuse_client.create_score(
            name="Online/Coherence",
            value=float(score.coherent),
            trace_id=trace_id,
            comment=score.reason,
        )
    except Exception as e:
        logger.exception("Coherence evaluation failed for trace %s: %s", trace_id, e)


# ── Follow-up Question Rate ───────────────────────────────────────────────────

_CLARIFICATION_RE = re.compile(
    r"\b(what did you mean|can you clarify|could you explain|"
    r"i don't understand|what about|but what|why did you|"
    r"that's wrong|that seems incorrect)\b",
    re.IGNORECASE,
)


def _is_clarification(message: str) -> bool:
    """Heuristic: is this a follow-up clarification request?"""
    return bool(_CLARIFICATION_RE.search(message))


def check_follow_up(
    time_since_response_ms: int,
    next_message: str,
    trace_id: str,
    langfuse_client: Langfuse,
    threshold_ms: int = 60_000,
) -> None:
    """Record a follow-up question if it arrives quickly after a response.

    Parameters
    ----------
    time_since_response_ms : int
        Milliseconds elapsed since the agent's response was delivered.
    next_message : str
        The user's next message.
    trace_id : str
        Langfuse trace ID of the preceding response.
    langfuse_client : Langfuse
        Langfuse client instance.
    threshold_ms : int
        Time window in ms within which a message is considered an immediate follow-up.
    """
    if time_since_response_ms < threshold_ms and _is_clarification(next_message):
        langfuse_client.create_score(
            name="UserFeedback/FollowUpRequired",
            value=1.0,
            trace_id=trace_id,
            comment=next_message[:100],
        )


# ── Convenience: report all synchronous online metrics ───────────────────────

def report_all_online_metrics(
    response: AgentResponse,
    trace_id: str,
    langfuse_client: Langfuse,
    run_coherence: bool = True,
) -> None:
    """Push all synchronous online metrics and optionally schedule the coherence judge.

    Coherence runs as a background asyncio task so it does not block the caller.

    Parameters
    ----------
    response : AgentResponse
        The completed agent response.
    trace_id : str
        Langfuse trace ID.
    langfuse_client : Langfuse
        Langfuse client instance.
    run_coherence : bool
        If True, schedules the async coherence LLM judge as a background task.
    """
    report_tool_call_metrics(response, trace_id, langfuse_client)
    report_retry_metrics(response, trace_id, langfuse_client)
    report_replanning_metrics(response, trace_id, langfuse_client)
    report_verification_compliance(response, trace_id, langfuse_client)

    if run_coherence:
        asyncio.create_task(
            evaluate_coherence_async(response.text, trace_id, langfuse_client)
        )
