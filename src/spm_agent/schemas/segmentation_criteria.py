from pathlib import Path
import json
import numpy as np
from pydantic import BaseModel, model_validator

class SegClass(BaseModel, frozen=True):
    id: int
    name: str
    definition: str

class SegCriterion(BaseModel, frozen=True):
    name: str
    channels: list[str]
    rule: str          # human-readable decision rule actually used in code
    rationale: str     # physical mechanism linking signal to the task

class CriteriaMeta(BaseModel, frozen=True):
    """Metadata contract for the coding agent's criteria.json deliverable."""
    task: str
    classes: list[SegClass]
    criteria: list[SegCriterion]

    @model_validator(mode="after")
    def _check(self):
        ids = [c.id for c in self.classes]
        assert len(ids) == len(set(ids)), "duplicate class ids"
        assert all(i > 0 for i in ids), "class ids must be > 0 (0 = unassigned)"
        assert self.criteria, "at least one criterion required"
        return self

def validate_criteria(npy_path: Path, json_path: Path, expected_hw: tuple,
                      max_classes: int = 8,
                      min_cov: float = 0.005, max_cov: float = 0.995):
    """Deterministic post-hoc validation. Returns (meta | None, errors: list[str])."""
    errors = []
    if not npy_path.exists():  errors.append("labels.npy not found")
    if not json_path.exists(): errors.append("criteria.json not found")
    if errors:
        return None, errors
    try:
        meta = CriteriaMeta(**json.loads(json_path.read_text()))
    except Exception as e:
        return None, [f"criteria.json invalid: {e}"]

    lab = np.load(npy_path)
    if lab.shape != tuple(expected_hw):
        errors.append(f"bad shape {lab.shape}, expected {expected_hw}")
        return meta, errors
    if not np.issubdtype(lab.dtype, np.integer):
        errors.append(f"labels must be integer, got {lab.dtype}")
    declared = {c.id for c in meta.classes}
    present  = set(np.unique(lab).tolist()) - {0}
    if len(declared) > max_classes: errors.append(f"{len(declared)} classes exceeds max {max_classes}")
    if present - declared:  errors.append(f"labels {sorted(present - declared)} not declared in criteria.json")
    if declared - present:  errors.append(f"declared classes {sorted(declared - present)} absent from labels.npy")
    cov = (lab > 0).mean()
    if not (min_cov <= cov <= max_cov): errors.append(f"assigned coverage {cov:.3f} outside [{min_cov}, {max_cov}]")
    return meta, errors