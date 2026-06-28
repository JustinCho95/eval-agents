"""Plan quality grader — LLM judge for research plan quality."""

import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from langfuse.experiment import Evaluation
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScoreDimension(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 (poor) to 5 (excellent)")
    comment: str = Field(description="Brief justification for the score")


class PlanQualityScore(BaseModel):
    coverage: ScoreDimension
    decomposition: ScoreDimension
    dependency_logic: ScoreDimension
    synthesis: ScoreDimension
    overall: ScoreDimension


_SYSTEM_PROMPT = """\
You are an expert evaluator assessing the quality of research plans created by an AI agent.

Score each dimension from 1 (very poor) to 5 (excellent). Be strict but fair.

## Scoring Dimensions

**Coverage (1-5):** Do the step descriptions collectively address the key concepts needed
to answer the question? A score of 5 means all critical aspects are covered with no gaps.

**Decomposition (1-5):** Is the question broken into an appropriate number of steps
with the right granularity for the stated complexity?
- Too few steps for a complex question → low score
- Unnecessary steps for a simple question → low score

**DependencyLogic (1-5):** Do later steps logically build on earlier ones? Expected
outputs of early steps should feed naturally into the requirements of dependent steps.

**Synthesis (1-5):** Is there a final synthesis step that specifically combines findings
into a complete answer? A vague "summarize" step scores lower than a targeted synthesis step.

**Overall (1-5):** Holistic quality of the plan for answering this specific question.
"""

_USER_PROMPT_TEMPLATE = """\
## Question
{question}

## Research Plan Steps
{plan_steps}

## Expected Plan Characteristics
- Minimum steps: {min_steps}
- Maximum steps: {max_steps}
- Complexity: {complexity}
- Must cover: {must_cover}
- Requires synthesis step: {must_have_synthesis}

Evaluate the plan on the five dimensions. Provide a score (1-5) and a brief comment for each.
"""


async def evaluate_plan_quality(
    question: str,
    plan_steps: list[str],
    plan_rubric: dict[str, Any],
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    """Evaluate the quality of a research plan using an LLM judge.

    Parameters
    ----------
    question : str
        The original question the plan was created for.
    plan_steps : list[str]
        Step descriptions from the plan (in order).
    plan_rubric : dict
        Expected plan characteristics. Fields: min_steps, max_steps, complexity,
        must_have_synthesis, must_cover (optional list of key concepts).
    model_config : LLMRequestConfig, optional
        LLM configuration. Defaults to temperature=0.

    Returns
    -------
    list[Evaluation]
        Five Langfuse evaluations: Coverage, Decomposition, DependencyLogic,
        Synthesis, Overall.
    """
    if model_config is None:
        model_config = LLMRequestConfig(temperature=0.0)

    if not plan_steps:
        return error_evaluations("No plan steps found in agent response")

    steps_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan_steps))
    must_cover = plan_rubric.get("must_cover", [])

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        plan_steps=steps_text,
        min_steps=plan_rubric.get("min_steps", 2),
        max_steps=plan_rubric.get("max_steps", 8),
        complexity=plan_rubric.get("complexity", "medium"),
        must_cover=", ".join(must_cover) if must_cover else "not specified",
        must_have_synthesis=plan_rubric.get("must_have_synthesis", True),
    )

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=model_config,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=PlanQualityScore,
        )
        score = completion.choices[0].message.parsed
        if score is None:
            return error_evaluations("Judge returned null response")

        return [
            Evaluation(name="Plan/Coverage", value=float(score.coverage.score), comment=score.coverage.comment),
            Evaluation(name="Plan/Decomposition", value=float(score.decomposition.score), comment=score.decomposition.comment),
            Evaluation(name="Plan/DependencyLogic", value=float(score.dependency_logic.score), comment=score.dependency_logic.comment),
            Evaluation(name="Plan/Synthesis", value=float(score.synthesis.score), comment=score.synthesis.comment),
            Evaluation(name="Plan/Overall", value=float(score.overall.score), comment=score.overall.comment),
        ]
    except Exception as e:
        logger.exception("Plan quality evaluation failed: %s", e)
        return error_evaluations(str(e))


def derive_plan_rubric(answer_type: str, problem_category: str) -> dict[str, Any]:
    """Auto-derive a plan rubric from dataset metadata.

    Parameters
    ----------
    answer_type : str
        DeepSearchQA answer type.
    problem_category : str
        DeepSearchQA problem category.

    Returns
    -------
    dict
        Plan rubric with auto-derived fields.
    """
    if answer_type == "Single Answer":
        complexity, min_steps, max_steps, must_have_synthesis = "low", 2, 4, False
    elif problem_category in ("Statistics & Data", "Science & Technology"):
        complexity, min_steps, max_steps, must_have_synthesis = "high", 3, 7, True
    else:
        complexity, min_steps, max_steps, must_have_synthesis = "medium", 2, 6, True

    return {
        "min_steps": min_steps,
        "max_steps": max_steps,
        "complexity": complexity,
        "must_have_synthesis": must_have_synthesis,
        "must_cover": [],
    }


def error_evaluations(error_msg: str) -> list[Evaluation]:
    """Return zero-value evaluations on grader failure."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="Plan/Coverage", value=0.0, comment=comment),
        Evaluation(name="Plan/Decomposition", value=0.0, comment=comment),
        Evaluation(name="Plan/DependencyLogic", value=0.0, comment=comment),
        Evaluation(name="Plan/Synthesis", value=0.0, comment=comment),
        Evaluation(name="Plan/Overall", value=0.0, comment=comment),
    ]
