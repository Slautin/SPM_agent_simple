from typing import Literal
from pydantic import BaseModel, Field, model_validator


class TaskFinding(BaseModel, frozen=True):
    """One task's answer. Fields are generated in order — the evidence is stated
    before the claim it is supposed to support."""
    task: str = Field(
        description="The experiment task, copied exactly from the TASKS list")
    evidence: str = Field(
        description="2-4 sentences: the measurements bearing on this task, citing loop "
                    "labels and values from the digest")
    answer: str = Field(
        description="2-4 sentences: what those measurements establish about the task. "
                    "Say plainly if they do not settle it")
    confidence: Literal["high", "medium", "low"] = Field(
        description="high: several good-quality loops agree; "
                    "medium: the trend holds but is thinly sampled or partly poor quality; "
                    "low: too sparse, inconsistent or compromised to settle the task")
    caveats: str = Field(
        description="What weakens this answer — sampling gaps, loop quality, conflicting "
                    "loops; 'none' if nothing material")


class ExperimentSummary(BaseModel, frozen=True):
    """Closing report: what the campaign did, what it establishes per task, what it
    does not, and what to measure next."""
    campaign: str = Field(
        description="2-4 sentences: what was actually measured — how many scans, where "
                    "they went and why, how the loops were distributed. Narrative, not a list")
    task_findings: list[TaskFinding] = Field(
        description="One entry per task in the TASKS list, in that order")
    inconsistencies: str = Field(
        description="Measurements that disagree with each other or with the answers "
                    "above, cited by loop label; 'none' if the evidence is coherent")
    limitations: str = Field(
        description="What this campaign could not establish and why — region coverage, "
                    "parameter range, loop quality")

    @model_validator(mode="after")
    def _check(self):
        assert self.task_findings, "at least one task finding is required"
        return self


def validate_summary(summary: ExperimentSummary, tasks: list[str]) -> list[str]:
    """Every task answered exactly once, none invented — context the schema cannot see.
    Same shape as validate_plan: errors feed a retry, they do not raise."""
    errors: list[str] = []
    got = [f.task for f in summary.task_findings]
    errors += [f"task {t!r} has no finding" for t in tasks if t not in got]
    errors += [f"finding for {t!r}, which is not one of the tasks" for t in got if t not in tasks]
    if len(set(got)) != len(got):
        errors.append("a task has more than one finding")
    return errors