"""Source validation grader — evaluates source authority, relevance, and diversity."""

import logging
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from langfuse.experiment import Evaluation
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


CATEGORY_SOURCE_RUBRIC: dict[str, dict[str, Any]] = {
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
    "Finance & Economics": {
        "expected_tiers": ["Tier 2", "Tier 3"],
        "domain_examples": ["central banks", "IMF", "World Bank", "Reuters", "Financial Times"],
        "min_sources": 2,
        "allow_tier3_if_no_tier1_or_2": True,
    },
    "_default": {
        "expected_tiers": ["Tier 1", "Tier 2", "Tier 3"],
        "domain_examples": [],
        "min_sources": 1,
        "allow_tier3_if_no_tier1_or_2": True,
    },
}

_AUTHORITY_TIER_REFERENCE = """\
Tier 1 — Research: academic journals, arXiv, conference proceedings (ACL, NeurIPS, Nature, etc.)
Tier 2 — Government/Institutional: national statistics offices, central banks, regulatory bodies,
          WHO, World Bank, CMHC, IMF
Tier 3 — Reputable reporting: Reuters, AP, BBC, Financial Times, established newspapers
Tier 4 — Reference: official documentation, technical standards (ISO, IEEE, RFC)
Tier 5 — Low authority: blogs, forums, Wikipedia, aggregators, social media, press releases
"""

_SYSTEM_PROMPT = f"""\
You are evaluating the quality of sources used by an AI research agent.

Score each dimension from 1 (very poor) to 5 (excellent).

## Authority Tier Reference

{_AUTHORITY_TIER_REFERENCE}

## Scoring Dimensions

**AuthorityLevel (1-5):** What tier are the sources? Tier 1-2 sources score highest.
Penalise blogs, forums, Wikipedia, and aggregators. A score of 5 requires Tier 1-2 sources.

**Relevance (1-5):** Are the sources directly relevant to the specific question,
or just topically adjacent? A topically adjacent but off-point source scores 2-3.

**Diversity (1-5):** Does the agent cite multiple independent sources, or rely entirely
on one source? 3+ independent sources score 4-5; a single source scores 1-2.

**Overall (1-5):** Holistic source quality for this question.
"""

_USER_PROMPT_TEMPLATE = """\
## Question
{question}

## Problem Category
{problem_category}

## Expected Source Types
- Expected tiers: {expected_tiers}
- Domain examples: {domain_examples}
- Minimum sources: {min_sources}
- Allow Tier 3 if no Tier 1/2 available: {allow_tier3}

## Sources Cited by Agent
{formatted_sources}

Evaluate source quality on the four dimensions.
"""


class ScoreDimension(BaseModel):
    score: int = Field(ge=1, le=5)
    comment: str


class SourceValidationScore(BaseModel):
    authority_level: ScoreDimension
    relevance: ScoreDimension
    diversity: ScoreDimension
    overall: ScoreDimension


def _is_vertex_source(uri: str) -> bool:
    """Return True if the URI is from the private knowledge base (vertex_search)."""
    return any(marker in uri for marker in (
        "aiplatform.googleapis.com",
        "discoveryengine",
        "datastore",
    )) or uri.startswith("gs://")


async def evaluate_source_validation(
    question: str,
    problem_category: str,
    sources: list[dict[str, str]],
    source_rubric: dict[str, Any] | None = None,
    model_config: LLMRequestConfig | None = None,
) -> list[Evaluation]:
    """Evaluate the authority and relevance of sources cited by the agent.

    Sources from vertex_search are already grounded in the private data store
    and are not evaluated against the public authority tier hierarchy.

    Parameters
    ----------
    question : str
        The original question.
    problem_category : str
        DeepSearchQA problem category (used to auto-derive rubric if not provided).
    sources : list[dict]
        Sources from AgentResponse — each with 'title' and 'uri' keys.
    source_rubric : dict, optional
        Expected source types. Auto-derived from problem_category if not provided.
    model_config : LLMRequestConfig, optional
        LLM configuration.

    Returns
    -------
    list[Evaluation]
        Four Langfuse evaluations: AuthorityLevel, Relevance, Diversity, Overall.
    """
    if model_config is None:
        model_config = LLMRequestConfig(temperature=0.0)

    vertex_sources = [s for s in sources if _is_vertex_source(s.get("uri", ""))]
    web_sources = [s for s in sources if not _is_vertex_source(s.get("uri", ""))]

    if not web_sources:
        if vertex_sources:
            return [
                Evaluation(name="Source/AuthorityLevel", value=5.0,
                           comment="All sources from vertex_search (private KB — grounded)"),
                Evaluation(name="Source/Relevance", value=5.0,
                           comment="vertex_search sources are directly grounded to the query"),
                Evaluation(name="Source/Diversity", value=float(min(len(vertex_sources), 5)),
                           comment=f"{len(vertex_sources)} KB document(s) cited"),
                Evaluation(name="Source/Overall", value=5.0,
                           comment="Knowledge base sources used correctly"),
            ]
        return error_evaluations("No sources found in agent response")

    if source_rubric is None:
        source_rubric = get_source_rubric(problem_category)

    formatted_sources = "\n".join(
        f"{i + 1}. [{s.get('title', 'Untitled')}] — {s.get('uri', 'no URI')}"
        for i, s in enumerate(web_sources)
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        question=question,
        problem_category=problem_category,
        expected_tiers=", ".join(source_rubric.get("expected_tiers", [])),
        domain_examples=", ".join(source_rubric.get("domain_examples", [])) or "not specified",
        min_sources=source_rubric.get("min_sources", 1),
        allow_tier3=source_rubric.get("allow_tier3_if_no_tier1_or_2", True),
        formatted_sources=formatted_sources,
    )

    try:
        client_manager = AsyncClientManager.get_instance()
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            model_config=model_config,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_format=SourceValidationScore,
        )
        score = completion.choices[0].message.parsed
        if score is None:
            return error_evaluations("Judge returned null response")

        return [
            Evaluation(name="Source/AuthorityLevel", value=float(score.authority_level.score),
                       comment=score.authority_level.comment),
            Evaluation(name="Source/Relevance", value=float(score.relevance.score),
                       comment=score.relevance.comment),
            Evaluation(name="Source/Diversity", value=float(score.diversity.score),
                       comment=score.diversity.comment),
            Evaluation(name="Source/Overall", value=float(score.overall.score),
                       comment=score.overall.comment),
        ]
    except Exception as e:
        logger.exception("Source validation evaluation failed: %s", e)
        return error_evaluations(str(e))


def get_source_rubric(problem_category: str) -> dict[str, Any]:
    """Get the source rubric for a problem category.

    Parameters
    ----------
    problem_category : str
        The problem category from the dataset.

    Returns
    -------
    dict
        Source rubric with expected tiers and domain examples.
    """
    return CATEGORY_SOURCE_RUBRIC.get(problem_category, CATEGORY_SOURCE_RUBRIC["_default"])


def error_evaluations(error_msg: str) -> list[Evaluation]:
    """Return zero-value evaluations on source validation failure."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="Source/AuthorityLevel", value=0.0, comment=comment),
        Evaluation(name="Source/Relevance", value=0.0, comment=comment),
        Evaluation(name="Source/Diversity", value=0.0, comment=comment),
        Evaluation(name="Source/Overall", value=0.0, comment=comment),
    ]
