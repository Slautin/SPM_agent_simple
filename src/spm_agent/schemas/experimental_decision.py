from pydantic import BaseModel, Field, model_validator
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
    frame_action: Optional[Literal["hold", "zoom_in", "zoom_out", "relocate"]] = Field(
        default=None,
        description="Required when action='scan'. "

                    "hold: keep the current frame — the region is still informative, the "
                    "acquisition was not (noisy, drifted, etc.). "

                    "zoom_in: the frame is TOO LARGE for the structure of interest to be "
                    "resolved — its features approach the pixel size. "

                    "zoom_out: the frame is TOO SMALL — the structure of interest extends "
                    "past its edges, so what was sampled is not representative of it. "

                    "relocate: the region is exhausted, fatigued or damaged — move to "
                    "unexplored area.")
    target_criterion: Optional[str] = Field(
        default=None,
        description="Required when action='loop': the "
                    "importance-map criterion the next measurement should serve, copied "
                    "exactly from the CRITERIA list in the digest")
    reasoning: str = Field(
        description="2-3 sentences: why this action closes the gap, citing specific "
                    "loop numbers and values from the digest. "
                    "For zoom_in/zoom_out, state the "
                    "structure's apparent scale against the current pixel size or frame size")


    @model_validator(mode="after")
    def _check(self):
        if self.action == "scan":
            assert self.frame_action, "action='scan' requires frame_action"
        elif self.frame_action is not None:
            assert False, "frame_action applies only to action='scan'"

        if self.action == "loop":
            assert self.target_criterion, "action='loop' requires target_criterion"
        else:
            assert self.target_criterion is None, \
                "target_criterion applies only to action='loop'"
        return self