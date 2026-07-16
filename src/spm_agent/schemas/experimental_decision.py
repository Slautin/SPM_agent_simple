from pydantic import BaseModel, Field
from typing import Literal, Optional

class ExperimentDecision(BaseModel, frozen=True):
    """Decide-node verdict: what to measure next in the PFM experiment, based on
    the digest of all results so far. Fields are generated in order — the model
    states its understanding before committing to an action."""

    understanding: str = Field(
        description="2-4 sentences: what the evidence collected so far says about "
                    "the experiment task — established findings, not plans")
    open_questions: str = Field(
        description="What remains unresolved or undersampled for the task; "
                    "'none' if the evidence already answers it")
    action: Literal["loop", "scan", "stop"] = Field(
        description="loop: measure one more hysteresis loop on the current scan; "
                    "scan: acquire a new image (current scan exhausted or compromised); "
                    "stop: task answered or further measurement adds nothing")
    target_criterion: Optional[str] = Field(
        default=None,
        description="Required when action='loop': the importance-map criterion the next "
                    "loop should serve, copied exactly from the CRITERIA list in the digest")
    reasoning: str = Field(
        description="2-3 sentences: why this action closes the gap, citing specific "
                    "loop numbers and values from the digest")