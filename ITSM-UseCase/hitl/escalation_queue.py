"""HITL escalation gate and in-memory complaint queue."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)

HOLDING_RESPONSE = (
    "Your complaint has been escalated to a specialist "
    "who will respond within 2 business days."
)

_queue: list["EscalatedComplaint"] = []


@dataclass
class EscalatedComplaint:
    """A complaint that has been routed to the HITL queue."""

    complaint_id: str
    complaint_text: str
    category: str
    confidence_score: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


def enqueue(
    complaint_text: str,
    category: str,
    confidence_score: float,
    reason: str,
    complaint_id: str | None = None,
) -> EscalatedComplaint:
    """Add a complaint to the escalation queue.

    Parameters
    ----------
    complaint_text : str
        The original complaint text.
    category : str
        Classified complaint category.
    confidence_score : float
        Classifier confidence at time of escalation.
    reason : str
        Why the complaint was escalated.
    complaint_id : str | None
        Optional ID; auto-generated if not provided.

    Returns
    -------
    EscalatedComplaint
        The queued complaint record.
    """
    complaint = EscalatedComplaint(
        complaint_id=complaint_id or str(uuid.uuid4()),
        complaint_text=complaint_text,
        category=category,
        confidence_score=confidence_score,
        reason=reason,
    )
    _queue.append(complaint)
    logger.debug(
        "Escalated complaint %s (category=%s reason=%s)",
        complaint.complaint_id,
        complaint.category,
        complaint.reason,
    )
    return complaint


def get_queue() -> list[EscalatedComplaint]:
    """Return a copy of the current escalation queue.

    Returns
    -------
    list[EscalatedComplaint]
        All complaints currently awaiting human review.
    """
    return list(_queue)


def clear_queue() -> None:
    """Clear the in-memory queue (useful for testing).

    Returns
    -------
    None
    """
    _queue.clear()
