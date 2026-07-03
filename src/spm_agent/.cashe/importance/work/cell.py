
import numpy as np
imp = np.load('importance_map.npy')
print("Shape:", imp.shape)
print("dtype:", imp.dtype)
print("min:", imp.min(), "max:", imp.max(), "mean:", imp.mean())
print("finite:", np.all(np.isfinite(imp)))
print("Values in [0,100]:", (imp >= 0).all() and (imp <= 100).all())
