from pathlib import Path
import json
import numpy as np
from pydantic import BaseModel, model_validator

class ComponentsMeta(BaseModel, frozen=True):
    """Metadata contract for the coding agent's components.json deliverable."""
    names: list[str]
    weights: list[float]
    rationale: list[str]

    @model_validator(mode="after")
    def _check(self):
        k = len(self.names)
        assert k == len(self.weights) == len(self.rationale), "names/weights/rationale length mismatch"
        assert all(w > 0 for w in self.weights), "weights must be > 0"
        s = sum(self.weights)
        object.__setattr__(self, "weights", [w / s for w in self.weights])  # normalize
        return self
    
def validate_components(npy_path: Path, json_path: Path, expected_hw: tuple,
                        max_k: int = 6, corr_max: float = 0.95):
    """Deterministic post-hoc validation of the agent's deliverables.
    Returns (meta | None, errors: list[str])."""
    errors = []
    if not npy_path.exists():  errors.append("components.npy not found")
    if not json_path.exists(): errors.append("components.json not found")
    if errors:
        return None, errors

    try:
        meta = ComponentsMeta(**json.loads(json_path.read_text()))
    except Exception as e:
        return None, [f"components.json invalid: {e}"]

    c = np.load(npy_path)
    if c.ndim != 3 or c.shape[1:] != tuple(expected_hw):
        errors.append(f"bad shape {c.shape}, expected (K, {expected_hw[0]}, {expected_hw[1]})")
        return meta, errors
    if len(meta.names) != c.shape[0]: errors.append(f"json lists {len(meta.names)} components, array has {c.shape[0]}")
    if c.shape[0] > max_k:            errors.append(f"K={c.shape[0]} exceeds max {max_k}")
    if not np.all(np.isfinite(c)):    errors.append("non-finite values")
    if c.min() < -1e-6 or c.max() > 1 + 1e-6: errors.append("values outside [0, 1]")
    if c.shape[0] > 1:
        corr = np.corrcoef(c.reshape(c.shape[0], -1))
        hi = np.abs(corr[np.triu_indices(c.shape[0], 1)]).max()
        if hi > corr_max: errors.append(f"redundant components (max |corr|={hi:.2f})")
    return meta, errors