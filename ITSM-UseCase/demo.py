"""CLI demo: run a complaint through the triage pipeline and print results."""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure ITSM-UseCase/ is on sys.path so pipeline, hitl, and config resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aieng.agent_evals.logging_config import setup_logging
from pipeline.orchestrator import run_pipeline


def print_result(result) -> None:
    """Print pipeline result in a human-readable format."""
    print("\n" + "=" * 60)
    print("COMPLAINT TRIAGE RESULT")
    print("=" * 60)
    print(f"Complaint: {result.complaint_text[:120]}{'...' if len(result.complaint_text) > 120 else ''}")
    print()

    if result.classifier:
        c = result.classifier
        print(f"Step 1 — Classification")
        print(f"  Category : {c['category']}")
        print(f"  Severity : {c['severity']}")
        print(f"  Confidence: {c['confidence']:.2f}")
        print()

    if result.retriever:
        r = result.retriever
        sections = [cl.get("section", "?") for cl in r["clauses"]]
        print(f"Step 2 — Retrieved {len(r['clauses'])} T&C clause(s)")
        for section in sections:
            print(f"  - {section}")
        print()

    if result.decision:
        d = result.decision
        print(f"Step 3 — Routing Decision: {d['route'].upper()}")
        print(f"  Reason: {d['reason']}")
        print()

    if result.escalated:
        print("Outcome : ESCALATED")
    elif result.responder:
        resp = result.responder
        print("Step 4 — Response Generated")
        print(f"  Grounded : {resp['grounded']}")
        print(f"  Citations: {', '.join(resp['cited_clauses']) or 'none'}")
        print()
        print("Outcome : RESOLVED")

    print()
    print(f"Response:\n{result.final_response}")
    print()
    print(f"Latency : {result.latency_sec:.2f}s")

    if result.error:
        print(f"Error   : {result.error}", file=sys.stderr)

    print("=" * 60)


async def main(complaint: str, no_tracing: bool) -> None:
    """Run the pipeline and print the result."""
    result = await run_pipeline(complaint, enable_tracing=not no_tracing)
    print_result(result)
    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Complaint triage demo.")
    parser.add_argument(
        "--complaint",
        required=True,
        help='Customer complaint text, e.g. --complaint "I was double charged last month."',
    )
    parser.add_argument("--no-tracing", action="store_true", help="Disable Langfuse tracing.")
    args = parser.parse_args()

    asyncio.run(main(args.complaint, args.no_tracing))
