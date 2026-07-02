from pydantic import BaseModel, Field
from typing_extensions import Literal

AnalysisTask = Literal[
    "ferroelectric domain segmentation",
    "ferroelectric domain wall segmentation",
    "grain boundary segmentation",
    "crack scratch detection",
    "surface contamination detection",
    "scanning artifact identification"
]

class SingleTaskChannelReccomendation(BaseModel):
    task: AnalysisTask = Field(
        description="Specific downstream image-analysis task"
    )

    feasible: bool = Field(
        description="Whether this task appears feasible from the available channels."
    )

    primary_channel: str = Field(
        description="Best channel ID for this task. Empty if infeasible."
    )

    secondary_channels: list[str] = Field(
        description="Supporting/reference channel IDs. Empty if not needed or infeasible."
    )

    confidence: float = Field(
        ge = 0.0,
        le = 1.0,
        description = "Confidence in this task-specific channel recommendation."
    )

    reasoning: str = Field(
        description="Short explanation of why these channels are suitable or why the task is infeasible."
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Task-specific warnings or limitations."
    )

class TaskChannelReccomendationReport(BaseModel):
    task_recommendation: list[SingleTaskChannelReccomendation]

    summary: str = Field(
        description="Brief overall summary of the channel-selection logic."
    )

    global_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings that apply to the whole recommendation."
    )

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the recommendations."
    )