import numpy as np


def build_map(components):
    """Equal-weight overview of the task criteria. Display and no-criterion fallback only —
    picking multiplies ONE named criterion by the safety map."""
    c = np.asarray(components, float)
    c = c / (c.mean(axis=(1, 2), keepdims=True) + 1e-12)   # so a sparse criterion isn't drowned
    m = c.mean(axis=0)
    return 100 * m / (m.max() + 1e-12)