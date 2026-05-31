"""End-to-end complaint triage pipeline orchestrator."""

import argparse
import asyncio
import logging
import time

from aieng.agent_evals.langfuse import init_tracing
from aieng.agent_evals.logging_config import setup_logging

from hitl.escalation_queue import HOLDING_RESPONSE, enqueue
from pipeline.classifier import Classifier
from pipeline.decision import Decision
from pipeline.models import PipelineResult
from pipeline.responder import Responder
from pipeline.retriever import Retriever


logger = logging.getLogger(__name__)

_LATENCY_WARNING_SEC = 15.0


async def run_pipeline(complaint_text: str, enable_tracing: bool = True) -> PipelineResult:
    """Run the full complaint triage pipeline.

    Steps: classify → retrieve → decide → (respond | escalate)

    Parameters
    ----------
    complaint_text : str
        Raw customer complaint text.
    enable_tracing : bool
        Whether to initialise Langfuse tracing. Default True.

    Returns
    -------
    PipelineResult
        Full result including all intermediate step outputs.
    """
    if enable_tracing:
        init_tracing(service_name="complaint-triage")

    from config import TriageConfigs  # noqa: PLC0415

    configs = TriageConfigs()  # type: ignore[call-arg]
    result = PipelineResult(complaint_text=complaint_text)
    start = time.perf_counter()

    try:
        # Step 1: classify
        logger.debug("Step 1: classify")
        classifier_out = await Classifier().run({"complaint_text": complaint_text})
        result.classifier = classifier_out  # type: ignore[assignment]

        # Step 2: retrieve
        logger.debug("Step 2: retrieve")
        retriever_out = await Retriever().run({
            "complaint_text": complaint_text,
            "category": classifier_out["category"],
        })
        result.retriever = retriever_out  # type: ignore[assignment]

        # Step 3: decide
        logger.debug("Step 3: decide")
        decision_out = await Decision(
            confidence_threshold=configs.confidence_threshold
        ).run({
            "category": classifier_out["category"],
            "severity": classifier_out["severity"],
            "confidence": classifier_out["confidence"],
            "clauses": retriever_out["clauses"],
        })
        result.decision = decision_out  # type: ignore[assignment]

        if decision_out["route"] == "escalate":
            # HITL gate
            enqueue(
                complaint_text=complaint_text,
                category=classifier_out["category"],
                confidence_score=classifier_out["confidence"],
                reason=decision_out["reason"],
            )
            result.escalated = True
            result.final_response = HOLDING_RESPONSE
            logger.debug("Complaint escalated: %s", decision_out["reason"])
        else:
            # Step 4: respond
            logger.debug("Step 4: respond")
            responder_out = await Responder().run({
                "complaint_text": complaint_text,
                "category": classifier_out["category"],
                "clauses": retriever_out["clauses"],
            })
            result.responder = responder_out  # type: ignore[assignment]
            result.final_response = responder_out["response_text"]

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        result.error = str(exc)

    result.latency_sec = time.perf_counter() - start

    if result.latency_sec > _LATENCY_WARNING_SEC:
        logger.warning("Pipeline latency %.1fs exceeded %.0fs target", result.latency_sec, _LATENCY_WARNING_SEC)
    else:
        logger.debug("Pipeline completed in %.2fs", result.latency_sec)

    return result


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run the complaint triage pipeline.")
    parser.add_argument("--complaint", required=True, help="Customer complaint text.")
    parser.add_argument("--no-tracing", action="store_true", help="Disable Langfuse tracing.")
    args = parser.parse_args()

    result = asyncio.run(run_pipeline(args.complaint, enable_tracing=not args.no_tracing))
    print(result.model_dump_json(indent=2))
