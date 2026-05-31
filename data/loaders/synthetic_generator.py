"""Generate synthetic edge-case complaint samples using Gemini."""

import json
import logging

from aieng.agent_evals.async_client_manager import AsyncClientManager


logger = logging.getLogger(__name__)

_CATEGORIES = [
    "billing",
    "service_failure",
    "fraud",
    "account_access",
    "product_dispute",
    "other",
]

_SYSTEM_PROMPT = """\
You are a synthetic data generator for a financial complaint triage system.
Generate realistic but fictional customer complaints for edge cases that are
hard to classify or involve ambiguous severity.

Return a JSON array of objects, each with:
- complaint_text: realistic customer complaint (2-4 sentences)
- category: one of billing|service_failure|fraud|account_access|product_dispute|other
- severity: one of low|medium|high|critical
- edge_case_type: brief label describing why this is an edge case

Return only valid JSON with no markdown fences."""


async def generate_synthetic_complaints(
    categories: list[str] | None = None,
    count_per_category: int = 3,
) -> list[dict]:
    """Generate synthetic edge-case complaints using Gemini.

    Parameters
    ----------
    categories : list[str] | None
        Categories to generate for. Defaults to all six.
    count_per_category : int
        Number of samples per category.

    Returns
    -------
    list[dict]
        Generated complaint records with complaint_text, category, severity, edge_case_type.
    """
    categories = categories or _CATEGORIES
    manager = AsyncClientManager.get_instance()

    user_message = (
        f"Generate {count_per_category} edge-case complaints for each of these categories: "
        f"{', '.join(categories)}. "
        f"Focus on cases that are ambiguous, borderline, or likely to be misclassified."
    )

    logger.info("Generating synthetic complaints for categories: %s", categories)

    try:
        response = await manager.openai_client.chat.completions.create(
            model=manager.configs.default_worker_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=1.0,
        )
        # Model may wrap the array in a key
        raw = json.loads(response.choices[0].message.content)
        records: list[dict] = raw if isinstance(raw, list) else next(
            (v for v in raw.values() if isinstance(v, list)), []
        )
        logger.info("Generated %d synthetic complaints", len(records))
        return records
    except Exception as exc:
        logger.error("Synthetic generation failed: %s", exc)
        return []
