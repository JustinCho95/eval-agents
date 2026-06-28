"""Evaluate the Knowledge Agent using Langfuse experiments.

Runs the Knowledge Agent against a Langfuse dataset with the full suite of
offline evaluators: DeepSearchQA (existing), plus plan quality, tool selection,
source validation, replanning rate, and knowledge base usage (new in Phase 2).

Optionally, trace-level groundedness evaluation can be enabled.

Usage:
    # Run full evaluation with all graders
    python evaluate.py

    # Custom dataset / experiment name
    python evaluate.py --dataset-name "MyDataset" --experiment-name "v2-test"

    # Enable trace groundedness
    ENABLE_TRACE_GROUNDEDNESS=true python evaluate.py
"""

import asyncio
import logging
import os

import click
from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation import run_experiment, run_experiment_with_trace_evals
from aieng.agent_evals.evaluation.graders import create_trace_groundedness_evaluator
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import EvaluationResult
from aieng.agent_evals.knowledge_qa.evaluation.offline import (
    KnowledgeQATask,
    deepsearchqa_evaluator,
    knowledge_base_evaluator,
    plan_quality_evaluator,
    replanning_evaluator,
    source_validation_evaluator,
    tool_selection_evaluator,
)
from aieng.agent_evals.logging_config import setup_logging
from dotenv import load_dotenv
from langfuse.experiment import ExperimentResult


load_dotenv(verbose=True)
setup_logging(level=logging.INFO, show_time=True, show_path=False)
logger = logging.getLogger(__name__)


DEFAULT_DATASET_NAME = "DeepSearchQA-Subset"
DEFAULT_EXPERIMENT_NAME = "Knowledge Agent Evaluation"

ENABLE_TRACE_GROUNDEDNESS = os.getenv("ENABLE_TRACE_GROUNDEDNESS", "false").lower() in ("true", "1", "yes")

_ALL_EVALUATORS = [
    deepsearchqa_evaluator,
    replanning_evaluator,
    plan_quality_evaluator,
    tool_selection_evaluator,
    knowledge_base_evaluator,
    source_validation_evaluator,
]


async def run_evaluation(
    dataset_name: str,
    experiment_name: str,
    max_concurrency: int = 1,
    enable_trace_groundedness: bool = False,
) -> ExperimentResult | EvaluationResult:
    """Run the full evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to evaluate against.
    experiment_name : str
        Name for this experiment run.
    max_concurrency : int, optional
        Maximum concurrent agent runs, by default 1.
    enable_trace_groundedness : bool, optional
        Whether to enable trace-level groundedness evaluation, by default False.
    """
    client_manager = AsyncClientManager.get_instance()
    task = KnowledgeQATask()

    try:
        logger.info("Starting experiment '%s' on dataset '%s'", experiment_name, dataset_name)
        logger.info("Max concurrency: %d", max_concurrency)
        logger.info("Trace groundedness: %s", "enabled" if enable_trace_groundedness else "disabled")

        result: ExperimentResult | EvaluationResult
        if enable_trace_groundedness:
            groundedness_evaluator = create_trace_groundedness_evaluator(
                name="trace_groundedness",
                model_config=LLMRequestConfig(temperature=0.0),
            )
            result = run_experiment_with_trace_evals(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Knowledge Agent — full offline evaluation suite + trace groundedness",
                task=task.run,
                evaluators=_ALL_EVALUATORS,
                trace_evaluators=[groundedness_evaluator],
                max_concurrency=max_concurrency,
            )
        else:
            result = run_experiment(
                dataset_name=dataset_name,
                name=experiment_name,
                description="Knowledge Agent — full offline evaluation suite",
                task=task.run,
                evaluators=_ALL_EVALUATORS,
                max_concurrency=max_concurrency,
            )

        logger.info("Experiment complete!")
        if isinstance(result, EvaluationResult):
            logger.info("Results: %s", result.experiment)
            if result.trace_evaluations:
                te = result.trace_evaluations
                logger.info(
                    "Trace evaluations: %d traces, %d skipped, %d failed",
                    len(te.evaluations_by_trace_id),
                    len(te.skipped_trace_ids),
                    len(te.failed_trace_ids),
                )
        else:
            logger.info("Results: %s", result)

        return result

    finally:
        logger.info("Closing client manager and flushing data...")
        try:
            await client_manager.close()
            await asyncio.sleep(0.1)
            logger.info("Cleanup complete")
        except Exception as e:
            logger.warning("Cleanup warning: %s", e)


@click.command()
@click.option(
    "--dataset-name",
    default=DEFAULT_DATASET_NAME,
    help="Name of the Langfuse dataset to evaluate against.",
)
@click.option(
    "--experiment-name",
    default=DEFAULT_EXPERIMENT_NAME,
    help="Name for this experiment run.",
)
@click.option(
    "--max-concurrency",
    default=1,
    type=int,
    help="Maximum concurrent agent runs (default: 1).",
)
@click.option(
    "--enable-trace-groundedness",
    is_flag=True,
    default=ENABLE_TRACE_GROUNDEDNESS,
    help="Enable trace-level groundedness evaluation.",
)
def cli(dataset_name: str, experiment_name: str, max_concurrency: int, enable_trace_groundedness: bool) -> None:
    """Run Knowledge Agent evaluation using Langfuse experiments."""
    asyncio.run(
        run_evaluation(
            dataset_name,
            experiment_name,
            max_concurrency,
            enable_trace_groundedness,
        )
    )


if __name__ == "__main__":
    cli()
