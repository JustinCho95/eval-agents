"""Step 2: retrieve relevant T&C clauses from Vertex AI Search."""

import logging

from aieng.agent_evals.tools.vertex_search import vertex_search

from .base import PipelineStep
from .models import RetrieverOutput


logger = logging.getLogger(__name__)


class Retriever(PipelineStep):
    """Step 2: retrieve T&C clauses relevant to the complaint via Vertex AI Search."""

    async def run(self, input: dict) -> dict:
        """Retrieve relevant T&C clauses.

        Parameters
        ----------
        input : dict
            Must contain ``complaint_text`` and ``category``.

        Returns
        -------
        dict
            RetrieverOutput fields.
        """
        complaint_text = str(input["complaint_text"])
        category = str(input.get("category", ""))
        query = f"{category} complaint: {complaint_text}".strip()

        logger.debug("Retriever query (first 120 chars): %s", query[:120])

        result = await vertex_search(query)

        if result["status"] == "error":
            logger.warning("Vertex AI Search failed: %s", result.get("error"))
            return RetrieverOutput(clauses=[], retrieval_scores=[]).model_dump()

        clauses = [
            {
                "clause_text": result["summary"],
                "section": src.get("title", "Unknown Section"),
                "score": 1.0,
            }
            for src in result["sources"]
        ]
        scores = [float(c["score"]) for c in clauses]

        logger.debug("Retriever: returned %d clause(s)", len(clauses))
        return RetrieverOutput(clauses=clauses, retrieval_scores=scores).model_dump()
