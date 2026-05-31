"""Abstract base class for complaint triage pipeline steps."""

from abc import ABC, abstractmethod


class PipelineStep(ABC):
    """Base class all pipeline steps must implement."""

    @abstractmethod
    async def run(self, input: dict) -> dict:
        """Execute the step.

        Parameters
        ----------
        input : dict
            Step-specific input data.

        Returns
        -------
        dict
            Step output data.
        """
        ...
