from pathlib import Path
from typing import Literal
import json
import numpy as np
from pydantic import BaseModel, Field, model_validator, field_validator


class ComponentsMeta(BaseModel, frozen=True):
    """Contract for components.json — the task criteria."""
    names: list[str]
    rationale: list[str]

    @model_validator(mode="after")
    def _check(self):
        assert len(self.names) == len(self.rationale), "names and rationale must be the same length"
        assert self.names, "at least one task criterion is required"
        assert len(set(self.names)) == len(self.names), "criterion names must be unique"
        return self

class SafetyMeta(BaseModel, frozen=True):
    """Contract for safety.json — the single safety map."""
    name: str
    rationale: str


def validate_components(npy_path: Path, json_path: Path, expected_hw: tuple,
                        max_k: int = 4, corr_max: float = 0.9, min_range: float = 0.1):
    """-> (meta | None, errors, warnings). errors = unusable (retry); warnings = kept as-is.
    Error strings go verbatim to the agent on retry — keep them actionable."""
    errors, warnings = [], []
    if not npy_path.exists():  errors.append("components.npy not found in the working directory")
    if not json_path.exists(): errors.append("components.json not found in the working directory")
    if errors:
        return None, errors, warnings
    try:
        meta = ComponentsMeta(**json.loads(json_path.read_text(encoding="utf-8")))
    except Exception as e:
        return None, [f"components.json invalid: {e}"], warnings

    c = np.load(npy_path)
    if c.ndim != 3 or c.shape[1:] != tuple(expected_hw):
        return meta, [f"components.npy has shape {c.shape}, expected "
                      f"(K, {expected_hw[0]}, {expected_hw[1]})"], warnings
    if c.shape[0] != len(meta.names):
        return meta, [f"components.json lists {len(meta.names)} criteria, "
                      f"the array holds {c.shape[0]}"], warnings
    if c.shape[0] > max_k:
        errors.append(f"K={c.shape[0]} exceeds the maximum of {max_k} task criteria")

    bad = [i for i in range(c.shape[0]) if not np.all(np.isfinite(c[i]))]
    if bad:
        errors.append(f"criteria {[meta.names[i] for i in bad]} contain non-finite values")
    out = [i for i in range(c.shape[0]) if c[i].min() < -1e-6 or c[i].max() > 1 + 1e-6]
    if out:
        errors.append(f"criteria {[meta.names[i] for i in out]} have values outside [0, 1]")

    flat = [i for i in range(c.shape[0])
            if float(np.percentile(c[i], 99) - np.percentile(c[i], 1)) < min_range]
    if flat:
        msg = (f"criteria {[meta.names[i] for i in flat]} are near-constant "
               f"(p99-p01 < {min_range}) and carry no information — rebuild or drop them")
        (errors if len(flat) == c.shape[0] else warnings).append(msg)

    if c.shape[0] > 1:                                    # redundancy: advisory
        sub  = c.reshape(c.shape[0], -1)
        keep = np.flatnonzero(sub.std(axis=1) > 1e-9)     # corrcoef is nan for constant maps
        if keep.size > 1:
            corr = np.corrcoef(sub[keep])
            for a in range(keep.size):
                for b in range(a + 1, keep.size):
                    if corr[a, b] > corr_max:
                        warnings.append(
                            f"criteria '{meta.names[keep[a]]}' and '{meta.names[keep[b]]}' are "
                            f"redundant (|corr|={abs(corr[a, b]):.2f} > {corr_max})")
    return meta, errors, warnings


def validate_safety(npy_path: Path, json_path: Path, expected_hw: tuple, components=None):
    """-> (meta | None, errors, warnings). `components` enables the leak check."""
    errors, warnings = [], []
    if not npy_path.exists():  errors.append("safety.npy not found in the working directory")
    if not json_path.exists(): errors.append("safety.json not found in the working directory")
    if errors:
        return None, errors, warnings
    try:
        meta = SafetyMeta(**json.loads(json_path.read_text(encoding="utf-8")))
    except Exception as e:
        return None, [f"safety.json invalid: {e}"], warnings

    s = np.load(npy_path)
    if s.shape != tuple(expected_hw):
        return meta, [f"safety.npy has shape {s.shape}, expected {tuple(expected_hw)}"], warnings
    if not np.all(np.isfinite(s)):
        errors.append("safety.npy contains non-finite values")
    if s.min() < -1e-6 or s.max() > 1 + 1e-6:
        errors.append("safety.npy has values outside [0, 1]")

    if float(np.percentile(s, 99)) < 0.9:                 # advisory
        warnings.append(f"safety never approaches 1 (p99={np.percentile(s, 99):.2f}) — no pixel is "
                        f"rated clearly safe; it looks stretched rather than calibrated")
    if float(((s > 0.2) & (s < 0.8)).mean()) > 0.5:       # advisory
        warnings.append("safety spreads most pixels across the mid-range — that is a rescale, "
                        "not a calibrated threshold")

    if components is not None:                            # the leak: advisory, see note below
        c = np.asarray(components)
        sz = s <= 1e-6
        if sz.sum():
            for i in range(c.shape[0]):
                if ((c[i] <= 1e-6) & sz).sum() / sz.sum() > 0.9:
                    warnings.append(f"task criterion {i} is zeroed over the same region as the "
                                    f"safety map — the penalty is applied twice")
                    break
    return meta, errors, warnings