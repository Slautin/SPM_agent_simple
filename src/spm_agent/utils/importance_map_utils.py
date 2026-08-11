import numpy as np

def build_map(components, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    m = np.tensordot(w, components, axes=1)
    return 100 * m / (m.max() + 1e-12)