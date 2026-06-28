"""Replanning rate evaluator — deterministic, no LLM required."""

from langfuse.experiment import Evaluation


def evaluate_replanning_rate(
    replan_count: int,
    plan_steps: int,
    max_replan_threshold: int,
) -> list[Evaluation]:
    """Compute replanning metrics from accumulated counters.

    Parameters
    ----------
    replan_count : int
        Number of times the agent replanned during execution.
    plan_steps : int
        Total number of plan steps (denominator for ratio).
    max_replan_threshold : int
        Maximum acceptable replan count before flagging.

    Returns
    -------
    list[Evaluation]
        Three Langfuse evaluations: Count, Flag, Ratio.
    """
    replan_flag = int(replan_count > max_replan_threshold)
    replan_ratio = replan_count / max(plan_steps, 1)
    return [
        Evaluation(name="Replanning/Count", value=float(replan_count)),
        Evaluation(name="Replanning/Flag", value=float(replan_flag)),
        Evaluation(name="Replanning/Ratio", value=round(replan_ratio, 4)),
    ]


def get_max_replan_threshold(answer_type: str, problem_category: str) -> int:
    """Auto-derive the replanning threshold from dataset metadata.

    Parameters
    ----------
    answer_type : str
        DeepSearchQA answer type (e.g. "Single Answer", "Set Answer").
    problem_category : str
        DeepSearchQA problem category (e.g. "Statistics & Data").

    Returns
    -------
    int
        Maximum replannings before a run is flagged.
    """
    if answer_type == "Single Answer":
        return 0
    if problem_category == "Statistics & Data":
        return 2
    return 1


def error_evaluations(error_msg: str) -> list[Evaluation]:
    """Return zero-value evaluations when replanning metrics cannot be computed."""
    comment = f"Evaluation error: {error_msg}"
    return [
        Evaluation(name="Replanning/Count", value=0.0, comment=comment),
        Evaluation(name="Replanning/Flag", value=0.0, comment=comment),
        Evaluation(name="Replanning/Ratio", value=0.0, comment=comment),
    ]
