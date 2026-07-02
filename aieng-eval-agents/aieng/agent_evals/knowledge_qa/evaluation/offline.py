"""Offline evaluation runner for the Knowledge-Grounded QA Agent.

Orchestrates all offline evaluators against a Langfuse dataset.

Usage
-----
    from aieng.agent_evals.knowledge_qa.evaluation.offline import evaluate
    await evaluate(dataset_name="DeepSearchQA-Subset")
"""

import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.knowledge_qa.agent import KnowledgeGroundedAgent
from aieng.agent_evals.knowledge_qa.deepsearchqa_grader import (
    DeepSearchQAResult,
    evaluate_deepsearchqa_async,
)
from langfuse.experiment import Evaluation, ExperimentResult

from .graders.plan_quality import derive_plan_rubric
from .graders.plan_quality import error_evaluations as plan_error_evals
from .graders.plan_quality import evaluate_plan_quality
from .graders.replanning import error_evaluations as replan_error_evals
from .graders.replanning import evaluate_replanning_rate, get_max_replan_threshold
from .graders.source_validation import error_evaluations as source_error_evals
from .graders.source_validation import evaluate_source_validation, get_source_rubric
from .graders.tool_selection import derive_tool_pattern
from .graders.tool_selection import error_evaluations as tool_error_evals
from .graders.tool_selection import evaluate_knowledge_base_usage
from .graders.tool_selection import evaluate_tool_selection
from .graders.tool_selection import kb_error_evaluations

logger = logging.getLogger(__name__)

_MODEL_CONFIG = LLMRequestConfig(temperature=0.0)


class KnowledgeQATask:
    """Task that runs the Knowledge Agent and returns a rich output dict."""

    async def run(self, *, item: Any, **kwargs: Any) -> dict[str, Any]:
        question = item.input if not isinstance(item, dict) else item["input"]
        logger.info("Running agent on: %s...", question[:80])

        try:
            agent = KnowledgeGroundedAgent(enable_planning=True)  # type: ignore[call-arg]
            response = await agent.answer_async(question)
            logger.info(
                "Agent completed: %d chars, %d tool calls",
                len(response.text),
                len(response.tool_calls),
            )

            client_manager = AsyncClientManager.get_instance()
            client_manager.langfuse_client.update_current_span(
                metadata=response.model_dump(exclude={"text"}),
            )

            return {
                "text": response.text,
                "plan_steps": len(response.plan.steps),
                "plan_descriptions": [s.description for s in response.plan.steps],
                "tool_calls": response.tool_calls,
                "sources": [{"title": s.title, "uri": s.uri} for s in response.sources],
                "replan_count": response.replan_count,
                "total_duration_ms": response.total_duration_ms,
            }
        except Exception as e:
            logger.error("Agent failed: %s", e)
            return {
                "text": f"Error: {e}",
                "plan_steps": 0,
                "plan_descriptions": [],
                "tool_calls": [],
                "sources": [],
                "replan_count": 0,
                "total_duration_ms": 0,
            }


async def deepsearchqa_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Outcome / F1 / Precision / Recall (existing grader — reused as-is)."""
    answer_text = output["text"] if isinstance(output, dict) else str(output)
    answer_type = (metadata or {}).get("answer_type", "Set Answer")
    try:
        result = await evaluate_deepsearchqa_async(
            question=input,
            answer=answer_text,
            ground_truth=expected_output,
            answer_type=answer_type,
            model_config=_MODEL_CONFIG,
        )
        logger.info("DeepSearchQA: %s (F1: %.2f)", result.outcome, result.f1_score)
        return result.to_evaluations()
    except Exception as e:
        logger.error("DeepSearchQA evaluation failed: %s", e)
        return DeepSearchQAResult.error_evaluations(str(e))


async def replanning_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """Deterministic replanning rate evaluator."""
    try:
        replan_count = output.get("replan_count", 0) if isinstance(output, dict) else 0
        plan_steps = output.get("plan_steps", 1) if isinstance(output, dict) else 1
        answer_type = (metadata or {}).get("answer_type", "Set Answer")
        problem_category = (metadata or {}).get("category", "")
        threshold = get_max_replan_threshold(answer_type, problem_category)
        return evaluate_replanning_rate(replan_count, plan_steps, threshold)
    except Exception as e:
        logger.error("Replanning evaluation failed: %s", e)
        return replan_error_evals(str(e))


async def plan_quality_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM judge for research plan quality."""
    try:
        plan_descriptions = output.get("plan_descriptions", []) if isinstance(output, dict) else []
        if not plan_descriptions:
            return plan_error_evals("No plan steps in output")

        answer_type = (metadata or {}).get("answer_type", "Set Answer")
        problem_category = (metadata or {}).get("category", "")
        example_id = (metadata or {}).get("example_id")

        plan_rubric = (metadata or {}).get("plan_rubric") or derive_plan_rubric(answer_type, problem_category)

        return await evaluate_plan_quality(
            question=input,
            plan_steps=plan_descriptions,
            plan_rubric=plan_rubric,
            model_config=_MODEL_CONFIG,
        )
    except Exception as e:
        logger.error("Plan quality evaluation failed: %s", e)
        return plan_error_evals(str(e))


async def tool_selection_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM judge for tool selection appropriateness and efficiency."""
    try:
        tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []
        answer_text = output.get("text", "") if isinstance(output, dict) else str(output)
        problem_category = (metadata or {}).get("category", "")
        answer_type = (metadata or {}).get("answer_type", "Set Answer")
        example_id = (metadata or {}).get("example_id")

        tool_pattern = (metadata or {}).get("tool_pattern") or derive_tool_pattern(problem_category, answer_type)

        return await evaluate_tool_selection(
            question=input,
            tool_calls=tool_calls,
            final_answer=answer_text,
            tool_pattern=tool_pattern,
            model_config=_MODEL_CONFIG,
        )
    except Exception as e:
        logger.error("Tool selection evaluation failed: %s", e)
        return tool_error_evals(str(e))


async def knowledge_base_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM judge for knowledge base tool usage (only runs on KB-flagged questions)."""
    requires_kb = (metadata or {}).get("requires_knowledge_base", False)
    if not requires_kb:
        return [
            Evaluation(name="KnowledgeBase/ToolUsed", value=0.0,
                       comment="Not a KB question — skipped"),
            Evaluation(name="KnowledgeBase/QueryQuality", value=0.0,
                       comment="Not a KB question — skipped"),
            Evaluation(name="KnowledgeBase/AnswerGrounded", value=0.0,
                       comment="Not a KB question — skipped"),
        ]
    try:
        tool_calls = output.get("tool_calls", []) if isinstance(output, dict) else []
        answer_text = output.get("text", "") if isinstance(output, dict) else str(output)
        kb_concepts = (metadata or {}).get("kb_concepts", [])
        return await evaluate_knowledge_base_usage(
            question=input,
            tool_calls=tool_calls,
            final_answer=answer_text,
            kb_concepts=kb_concepts,
            model_config=_MODEL_CONFIG,
        )
    except Exception as e:
        logger.error("Knowledge base evaluation failed: %s", e)
        return kb_error_evaluations(str(e))


async def source_validation_evaluator(
    *,
    input: str,  # noqa: A002
    output: Any,
    expected_output: str,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[Evaluation]:
    """LLM judge for source authority and relevance."""
    try:
        sources = output.get("sources", []) if isinstance(output, dict) else []
        problem_category = (metadata or {}).get("category", "")
        source_rubric = (metadata or {}).get("source_rubric") or get_source_rubric(problem_category)

        return await evaluate_source_validation(
            question=input,
            problem_category=problem_category,
            sources=sources,
            source_rubric=source_rubric,
            model_config=_MODEL_CONFIG,
        )
    except Exception as e:
        logger.error("Source validation evaluation failed: %s", e)
        return source_error_evals(str(e))


async def evaluate(
    dataset_name: str,
    experiment_name: str = "Knowledge QA — Full Offline Evaluation",
    max_concurrency: int = 5,
) -> ExperimentResult:
    """Run all offline evaluators against a Langfuse dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run in Langfuse.
    max_concurrency : int
        Maximum concurrent agent + evaluator runs.

    Returns
    -------
    ExperimentResult
        Langfuse experiment result.
    """
    client_manager = AsyncClientManager.get_instance()
    try:
        logger.info(
            "Starting experiment '%s' on dataset '%s'",
            experiment_name,
            dataset_name,
        )
        task = KnowledgeQATask()
        result = run_experiment(
            dataset_name,
            name=experiment_name,
            description=(
                "Knowledge QA Agent — plan quality, tool selection, "
                "source validation, replanning"
            ),
            task=task.run,
            evaluators=[
                deepsearchqa_evaluator,
                replanning_evaluator,
                plan_quality_evaluator,
                tool_selection_evaluator,
                knowledge_base_evaluator,
                source_validation_evaluator,
            ],
            max_concurrency=max_concurrency,
        )
        logger.info("Experiment complete")
        return result
    finally:
        try:
            await client_manager.close()
        except Exception as e:
            logger.warning("Cleanup warning: %s", e)
