"""Step 1: classify complaint category, severity, and confidence."""

import json
import logging

from aieng.agent_evals.async_client_manager import AsyncClientManager

from .base import PipelineStep
from .models import ClassifierOutput


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a complaint classifier for a financial institution.
Given a customer complaint, return a JSON object with exactly these fields:
- category: one of billing|service_failure|fraud|account_access|product_dispute|other
- severity: one of low|medium|high|critical
- confidence: float 0.0-1.0 indicating classification certainty
- raw_response: one-sentence summary of the complaint

Return only valid JSON with no markdown fences."""


class Classifier(PipelineStep):
    """Step 1: classify complaint category, severity, and confidence."""

    async def run(self, input: dict) -> dict:
        """Classify the complaint.

        Parameters
        ----------
        input : dict
            Must contain ``complaint_text``.

        Returns
        -------
        dict
            ClassifierOutput fields.
        """
        complaint_text = str(input["complaint_text"])
        logger.debug("Classifier input: %d chars", len(complaint_text))

        manager = AsyncClientManager.get_instance()

        for attempt in range(2):
            try:
                response = await manager.openai_client.chat.completions.create(
                    model=manager.configs.default_worker_model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": complaint_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(response.choices[0].message.content)
                result = ClassifierOutput(
                    category=data.get("category", "other"),
                    severity=data.get("severity", "medium"),
                    confidence=float(data.get("confidence", 0.5)),
                    raw_response=data.get("raw_response", ""),
                )
                logger.debug(
                    "Classifier: category=%s severity=%s confidence=%.2f",
                    result.category,
                    result.severity,
                    result.confidence,
                )
                return result.model_dump()
            except Exception as exc:
                logger.warning("Classifier attempt %d failed: %s", attempt + 1, exc)

        logger.warning("Classifier returning fallback result after 2 attempts")
        return ClassifierOutput(
            category="other", severity="medium", confidence=0.0, raw_response=""
        ).model_dump()
