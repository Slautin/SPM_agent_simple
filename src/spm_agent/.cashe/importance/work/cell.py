
import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np

imp = np.load('importance_map.npy')
print("importance_map.npy loaded successfully")
print("Shape:", imp.shape)
print("dtype:", imp.dtype)
print("min={:.4f}, max={:.4f}, mean={:.4f}".format(imp.min(), imp.max(), imp.mean()))
print("All finite:", np.all(np.isfinite(imp)))
print("All in [0,100]:", np.all((imp >= 0) & (imp <= 100)))
