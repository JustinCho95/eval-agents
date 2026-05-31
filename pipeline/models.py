"""Pydantic models for all pipeline step inputs, outputs, and results."""

from typing import Literal

from pydantic import BaseModel


class ClassifierOutput(BaseModel):
    """Output from Step 1 — complaint classification."""

    category: Literal["billing", "service_failure", "fraud", "account_access", "product_dispute", "other"]
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float
    raw_response: str


class RetrieverOutput(BaseModel):
    """Output from Step 2 — T&C clause retrieval."""

    clauses: list[dict]
    retrieval_scores: list[float]


class DecisionOutput(BaseModel):
    """Output from Step 3 — resolve vs escalate routing decision."""

    route: Literal["resolve", "escalate"]
    reason: str
    decision_confidence: float


class ResponderOutput(BaseModel):
    """Output from Step 4 — customer-facing response generation."""

    response_text: str
    cited_clauses: list[str]
    grounded: bool


class PipelineResult(BaseModel):
    """Full pipeline result including all intermediate step outputs."""

    complaint_text: str
    classifier: ClassifierOutput | None = None
    retriever: RetrieverOutput | None = None
    decision: DecisionOutput | None = None
    responder: ResponderOutput | None = None
    final_response: str = ""
    escalated: bool = False
    latency_sec: float = 0.0
    error: str | None = None
