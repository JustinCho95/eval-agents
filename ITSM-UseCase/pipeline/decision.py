"""Step 3: resolve vs escalate routing decision with extended reasoning."""

import json
import logging

from google.genai import Client
from google.genai import types as genai_types

from aieng.agent_evals.async_client_manager import AsyncClientManager

from .base import PipelineStep
from .models import DecisionOutput


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a complaint routing agent for a financial institution.
Given complaint classification and retrieved T&C clauses, decide whether to resolve
the complaint automatically or escalate to a human specialist.

Return a JSON object with exactly these fields:
- route: "resolve" or "escalate"
- reason: one-sentence explanation of the routing decision
- decision_confidence: float 0.0-1.0

Return only valid JSON with no markdown fences."""


class Decision(PipelineStep):
    """Step 3: decide whether to resolve the complaint or escalate to a specialist.

    Uses rule-based fast paths for clear cases and gemini-2.5-pro with extended
    thinking for nuanced situations.
    """

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        """Initialise with configurable confidence threshold.

        Parameters
        ----------
        confidence_threshold : float
            Classifier confidence below this triggers escalation.
        """
        self.confidence_threshold = confidence_threshold

    async def run(self, input: dict) -> dict:
        """Make the routing decision.

        Parameters
        ----------
        input : dict
            Must contain ``category``, ``severity``, ``confidence``, ``clauses``.

        Returns
        -------
        dict
            DecisionOutput fields.
        """
        confidence = float(input.get("confidence", 0.0))
        severity = str(input.get("severity", "medium"))
        clauses = input.get("clauses", [])

        # Rule-based fast paths — no LLM call needed
        if confidence < self.confidence_threshold:
            logger.debug("Escalating: confidence %.2f < threshold %.2f", confidence, self.confidence_threshold)
            return DecisionOutput(
                route="escalate",
                reason=f"Classifier confidence {confidence:.2f} is below threshold {self.confidence_threshold}.",
                decision_confidence=1.0,
            ).model_dump()

        if severity == "critical":
            logger.debug("Escalating: critical severity")
            return DecisionOutput(
                route="escalate",
                reason="Critical severity requires human review.",
                decision_confidence=1.0,
            ).model_dump()

        if not clauses:
            logger.debug("Escalating: no T&C clauses retrieved")
            return DecisionOutput(
                route="escalate",
                reason="No relevant T&C clauses found to support an automated response.",
                decision_confidence=1.0,
            ).model_dump()

        # LLM with extended thinking for nuanced cases
        manager = AsyncClientManager.get_instance()
        user_content = json.dumps({
            "category": input.get("category"),
            "severity": severity,
            "confidence": confidence,
            "retrieved_sections": [c.get("section", "") for c in clauses[:3]],
        })

        try:
            client = Client(api_key=manager.configs.google_api_key.get_secret_value())
            try:
                response = client.models.generate_content(
                    model=manager.configs.default_planner_model,
                    contents=[genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=_SYSTEM_PROMPT + "\n\n" + user_content)],
                    )],
                    config=genai_types.GenerateContentConfig(
                        thinking_config=genai_types.ThinkingConfig(include_thoughts=True),
                        temperature=0.0,
                    ),
                )
            finally:
                client.close()

            # Extract non-thought text parts
            text = "".join(
                p.text
                for p in response.candidates[0].content.parts
                if hasattr(p, "text") and p.text and not getattr(p, "thought", False)
            )
            data = json.loads(text)
            result = DecisionOutput(
                route=data.get("route", "escalate"),
                reason=data.get("reason", ""),
                decision_confidence=float(data.get("decision_confidence", 0.5)),
            )
            logger.debug("Decision: route=%s confidence=%.2f", result.route, result.decision_confidence)
            return result.model_dump()

        except Exception as exc:
            logger.warning("Decision LLM failed, defaulting to escalate: %s", exc)
            return DecisionOutput(
                route="escalate",
                reason=f"Decision step encountered an error: {exc}",
                decision_confidence=0.0,
            ).model_dump()
