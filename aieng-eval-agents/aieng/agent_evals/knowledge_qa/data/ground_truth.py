"""Ground truth annotation loader for offline evaluation graders.

Annotations are stored in implementations/knowledge_qa/data/ground_truth_annotations.jsonl.
Each line is a JSON object keyed by example_id.

The loader is used by the offline evaluators to enrich auto-derived rubrics with
per-question annotations (primarily must_cover for Plan Quality and
reference_tool_call_count for Tool Selection).
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default path relative to the repo root
# parents[0]=data, [1]=knowledge_qa, [2]=agent_evals, [3]=aieng, [4]=aieng-eval-agents, [5]=repo root
_DEFAULT_ANNOTATIONS_PATH = Path(__file__).parents[5] / "implementations" / "knowledge_qa" / "data" / "ground_truth_annotations.jsonl"


class GroundTruthAnnotations:
    """Loads and provides per-question ground truth annotations.

    Parameters
    ----------
    path : Path, optional
        Path to the annotations JSONL file. Defaults to the repo-standard location.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _DEFAULT_ANNOTATIONS_PATH
        self._annotations: dict[int, dict[str, Any]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self._path.exists():
            logger.debug("No ground truth annotations file found at %s", self._path)
            return

        count = 0
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                    example_id = record.get("example_id")
                    if example_id is not None:
                        self._annotations[int(example_id)] = record
                        count += 1
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed annotation line: %s — %s", line[:80], e)

        if count:
            logger.info("Loaded %d ground truth annotations from %s", count, self._path)

    def get(self, example_id: int) -> dict[str, Any] | None:
        """Return the annotation for a given example_id, or None if not annotated.

        Parameters
        ----------
        example_id : int
            The example_id from DSQAExample / Langfuse item metadata.

        Returns
        -------
        dict or None
            The annotation record, or None if this example has no annotation.
        """
        self._load()
        return self._annotations.get(example_id)

    def get_plan_rubric_overrides(self, example_id: int) -> dict[str, Any]:
        """Return plan rubric overrides for a given example, or empty dict.

        Parameters
        ----------
        example_id : int
            The example_id from DSQAExample / Langfuse item metadata.

        Returns
        -------
        dict
            Fields to overlay on top of the auto-derived plan rubric.
            Empty dict if no annotation exists.
        """
        annotation = self.get(example_id)
        if not annotation:
            return {}
        return annotation.get("plan_rubric", {})

    def get_tool_pattern_overrides(self, example_id: int) -> dict[str, Any]:
        """Return tool pattern overrides for a given example, or empty dict.

        Parameters
        ----------
        example_id : int
            The example_id from DSQAExample / Langfuse item metadata.

        Returns
        -------
        dict
            Fields to overlay on top of the auto-derived tool pattern.
            Empty dict if no annotation exists.
        """
        annotation = self.get(example_id)
        if not annotation:
            return {}
        return annotation.get("tool_pattern", {})

    def __len__(self) -> int:
        self._load()
        return len(self._annotations)

    def __repr__(self) -> str:
        self._load()
        return f"GroundTruthAnnotations(path={self._path}, count={len(self._annotations)})"


# Module-level singleton so the file is only read once per process
_annotations = GroundTruthAnnotations()


def get_annotations() -> GroundTruthAnnotations:
    """Return the module-level annotations singleton."""
    return _annotations
