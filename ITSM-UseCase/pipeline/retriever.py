"""Step 2: retrieve relevant T&C clauses from ChromaDB."""

import logging

from knowledge_base.chroma_store import query as chroma_query

from .base import PipelineStep
from .models import RetrieverOutput


logger = logging.getLogger(__name__)


class Retriever(PipelineStep):
    """Step 2: retrieve T&C clauses relevant to the complaint via ChromaDB."""

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

        try:
            clauses = chroma_query(query, n_results=3)
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return RetrieverOutput(clauses=[], retrieval_scores=[]).model_dump()

        scores = [float(c["score"]) for c in clauses]

        logger.debug("Retriever: returned %d clause(s)", len(clauses))
        return RetrieverOutput(clauses=clauses, retrieval_scores=scores).model_dump()
