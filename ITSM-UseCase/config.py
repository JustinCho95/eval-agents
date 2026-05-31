"""Complaint triage configuration extending the shared eval agents config."""

from pydantic import Field

from aieng.agent_evals.configs import Configs as _BaseConfigs


class TriageConfigs(_BaseConfigs):
    """Configuration for the complaint triage pipeline.

    Inherits all base settings (API keys, model names, Langfuse, Vertex AI Search)
    and adds complaint-specific fields.
    """

    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to auto-resolve; below this triggers HITL escalation.",
    )
    knowledge_base_path: str = Field(
        default="knowledge_base/scotia_tnc.pdf",
        description="Path to the T&C source document.",
    )
