"""Step 4: generate a grounded customer-facing response citing T&C clauses."""

import json
import logging

from aieng.agent_evals.async_client_manager import AsyncClientManager

from .base import PipelineStep
from .models import ResponderOutput


logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a customer service agent for a financial institution.
Draft a clear, professional response to the customer complaint below.

Rules:
- Cite at least one T&C clause by its section name (e.g. "Section 4.2 — Billing Disputes").
- Keep the response under 300 words.
- Do not invent facts or promises not supported by the provided clauses.

Return a JSON object with exactly these fields:
- response_text: the customer-facing response (plain text, max 300 words)
- cited_clauses: list of section names cited (e.g. ["Section 4.2 — Billing Disputes"])

Return only valid JSON with no markdown fences."""


class Responder(PipelineStep):
    """Step 4: draft a grounded customer-facing response citing T&C clauses."""

    async def run(self, input: dict) -> dict:
        """Generate the customer response.

        Parameters
        ----------
        input : dict
            Must contain ``complaint_text``, ``category``, ``clauses``.

        Returns
        -------
        dict
            ResponderOutput fields.
        """
        complaint_text = str(input["complaint_text"])
        category = str(input.get("category", ""))
        clauses = input.get("clauses", [])

        clause_context = "\n".join(
            f"- {c.get('section', 'Unknown')}: {c.get('clause_text', '')[:300]}"
            for c in clauses[:3]
        )
        user_message = (
            f"Complaint category: {category}\n\n"
            f"Customer complaint:\n{complaint_text}\n\n"
            f"Relevant T&C clauses:\n{clause_context}"
        )

        logger.debug("Responder input: category=%s, %d clauses", category, len(clauses))

        manager = AsyncClientManager.get_instance()

        try:
            response = await manager.openai_client.chat.completions.create(
                model=manager.configs.default_worker_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            data = json.loads(response.choices[0].message.content)
            response_text = data.get("response_text", "")
            cited_clauses = data.get("cited_clauses", [])
            grounded = bool(cited_clauses)

            logger.debug(
                "Responder: %d words, %d citations, grounded=%s",
                len(response_text.split()),
                len(cited_clauses),
                grounded,
            )
            return ResponderOutput(
                response_text=response_text,
                cited_clauses=cited_clauses,
                grounded=grounded,
            ).model_dump()

        except Exception as exc:
            logger.warning("Responder failed: %s", exc)
            return ResponderOutput(
                response_text="We have received your complaint and will follow up shortly.",
                cited_clauses=[],
                grounded=False,
            ).model_dump()
