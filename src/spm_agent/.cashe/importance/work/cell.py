
import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# Final verification
components = np.load('components.npy')
print("components.npy shape:", components.shape)
print("dtype:", components.dtype)
print("All finite:", np.all(np.isfinite(components)))
print("Min:", components.min(), "Max:", components.max())

# Verify all in [0,1]
for i in range(components.shape[0]):
    print(f"  Slice {i}: min={components[i].min():.4f}, max={components[i].max():.4f}, mean={components[i].mean():.4f}")

# Final visualization with all criteria side by side
names = ['PFM Amplitude\n(w=1.5)', 'Domain Interior\n(w=2.0)', 'Wall Proximity\n(w=2.0)', 
         'Phase Clarity\n(w=1.5)', 'DART Balance\n(w=1.0)']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i in range(5):
    im = axes[i].imshow(components[i], cmap='inferno', vmin=0, vmax=1)
    axes[i].set_title(names[i], fontsize=10)
    plt.colorbar(im, ax=axes[i], fraction=0.046)
    axes[i].axis('off')

plt.suptitle('Final Scoring Components: Domain vs Wall Switching Comparison', fontsize=12)
plt.tight_layout()
plt.savefig('final_components.png', dpi=120)
plt.close()
print("Saved final_components.png")
