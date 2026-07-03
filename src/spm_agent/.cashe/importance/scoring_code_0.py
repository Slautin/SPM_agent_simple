
import numpy as np
import matplotlib.pyplot as plt

# Load all channels
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')

# Load masks
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

print("Shapes:", height.shape, amp1.shape, phase1.shape)
print("Domain mask:", domain_mask.shape, domain_mask.dtype, domain_mask.sum(), "pixels")
print("Wall mask:", wall_mask.shape, wall_mask.dtype, wall_mask.sum(), "pixels")
print("\nPhase1 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase1.min(), phase1.max(), phase1.mean()))
print("Phase2 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Amp1 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp1.min(), amp1.max(), amp1.mean()))
print("Amp2 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp2.min(), amp2.max(), amp2.mean()))


# --- next cell ---

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load all channels
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')

# Load masks
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

print("Shapes:", height.shape, amp1.shape, phase1.shape)
print("Domain mask:", domain_mask.shape, domain_mask.dtype, "coverage:", domain_mask.mean())
print("Wall mask:", wall_mask.shape, wall_mask.dtype, "coverage:", wall_mask.mean())
print("\nPhase1 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase1.min(), phase1.max(), phase1.mean()))
print("Phase2 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Amp1 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp1.min(), amp1.max(), amp1.mean()))
print("Amp2 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp2.min(), amp2.max(), amp2.mean()))
print("Freq stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(freq.min(), freq.max(), freq.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
# Remove cached matplotlib if any
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

# Load all channels
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')

domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

print("Shapes:", height.shape, amp1.shape, phase1.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase1: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase1.min(), phase1.max(), phase1.mean()))
print("Phase2: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Amp1: min={:.2e}, max={:.2e}".format(amp1.min(), amp1.max()))
print("Freq: min={:.2e}, max={:.2e}".format(freq.min(), freq.max()))


# --- next cell ---

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
from scipy import ndimage

# Reload data
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

# Visualize all channels + masks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0,0].imshow(height, cmap='gray'); axes[0,0].set_title('Height')
axes[0,1].imshow(amp1, cmap='hot'); axes[0,1].set_title('Amp1')
axes[0,2].imshow(amp2, cmap='hot'); axes[0,2].set_title('Amp2')
axes[0,3].imshow(phase1, cmap='RdBu'); axes[0,3].set_title('Phase1')
axes[1,0].imshow(phase2, cmap='RdBu'); axes[1,0].set_title('Phase2')
axes[1,1].imshow(freq, cmap='viridis'); axes[1,1].set_title('Frequency')
axes[1,2].imshow(domain_mask, cmap='gray'); axes[1,2].set_title('Domain mask')
axes[1,3].imshow(wall_mask, cmap='gray'); axes[1,3].set_title('Wall mask')
plt.tight_layout()
plt.savefig('overview.png', dpi=100)
plt.close()
print("Saved overview.png")


# --- next cell ---

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
from scipy import ndimage

# Reload
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

# Key observations:
# - Phase1 and Phase2 show clear domain contrast (PFM phase = polarization direction)
# - Amp1/Amp2 show domain walls as dark lines (amplitude dips at walls)
# - Domain mask = interior of domains (white regions)
# - Wall mask = domain wall regions (thin lines)

# For "compare switching inside domains vs near domain walls":
# We want BOTH types of locations well-represented:
# 1. Clean domain interiors: high PFM amplitude (strong signal), uniform phase
# 2. Domain walls: amplitude minima, phase transition zones

# --- CRITERION 1: PFM Amplitude signal quality ---
# High amplitude = reliable switching signal; normalize both harmonics
amp1_norm = (amp1 - amp1.min()) / (amp1.max() - amp1.min())
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())
amp_mean = (amp1_norm + amp2_norm) / 2.0

# --- CRITERION 2: Phase contrast / local phase gradient ---
# Phase gradient highlights domain walls and transition zones
# Use phase1 (cleaner domain contrast from overview)
# Compute local gradient magnitude of phase1
gy1, gx1 = np.gradient(phase1)
phase1_grad = np.sqrt(gx1**2 + gy1**2)
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase_grad = (phase1_grad + phase2_grad) / 2.0

# Normalize gradient
phase_grad_norm = (phase_grad - np.percentile(phase_grad, 1)) / (np.percentile(phase_grad, 99) - np.percentile(phase_grad, 1))
phase_grad_norm = np.clip(phase_grad_norm, 0, 1)

# --- CRITERION 3: Distance-based weighting for domain interiors ---
# For domain interiors: distance from wall (deeper inside = more representative)
# Compute distance transform from wall mask
dist_from_wall = ndimage.distance_transform_edt(~wall_mask)
dist_from_wall_norm = (dist_from_wall - dist_from_wall.min()) / (dist_from_wall.max() - dist_from_wall.min())

# --- CRITERION 4: Proximity to domain walls ---
# For wall regions: distance from domain interior (closer to wall center = better)
dist_from_domain = ndimage.distance_transform_edt(~domain_mask)
dist_from_domain_norm = (dist_from_domain - dist_from_domain.min()) / (dist_from_domain.max() - dist_from_domain.min())

print("Phase grad stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase_grad.min(), phase_grad.max(), phase_grad.mean()))
print("Dist from wall: min={:.1f}, max={:.1f}".format(dist_from_wall.min(), dist_from_wall.max()))
print("Dist from domain: min={:.1f}, max={:.1f}".format(dist_from_domain.min(), dist_from_domain.max()))
print("Amp mean: min={:.3f}, max={:.3f}".format(amp_mean.min(), amp_mean.max()))


# --- next cell ---

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
from scipy import ndimage

# Reload
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

# ---- Build component scores ----

# 1. PFM amplitude (signal quality for switching measurement)
amp1_norm = (amp1 - amp1.min()) / (amp1.max() - amp1.min())
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())
amp_score = (amp1_norm + amp2_norm) / 2.0

# 2. Phase gradient (marks transition zones / wall sharpness)
gy1, gx1 = np.gradient(phase1)
phase1_grad = np.sqrt(gx1**2 + gy1**2)
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase_grad = (phase1_grad + phase2_grad) / 2.0
p1, p99 = np.percentile(phase_grad, 1), np.percentile(phase_grad, 99)
phase_grad_norm = np.clip((phase_grad - p1) / (p99 - p1), 0, 1)

# 3. Distance from wall (for domain interior score — deeper = more representative)
dist_from_wall = ndimage.distance_transform_edt(~wall_mask)
dist_from_wall_norm = dist_from_wall / dist_from_wall.max()

# 4. Proximity to wall center (for wall score — at wall = most informative)
# Wall pixels: score by how centered they are in the wall band
# Use distance from non-wall as proxy for wall center
dist_inside_wall = ndimage.distance_transform_edt(wall_mask)
dist_inside_wall_norm = dist_inside_wall / (dist_inside_wall.max() + 1e-10)

# 5. Frequency variation (local stiffness/contact quality)
freq_local_std = ndimage.generic_filter(freq, np.std, size=5)
freq_std_norm = (freq_local_std - freq_local_std.min()) / (freq_local_std.max() - freq_local_std.min())

# ---- DOMAIN INTERIOR SCORE ----
# Want: inside domain mask, high amplitude, deep from wall, moderate phase gradient
# (not at wall, but showing clear polarization state)
domain_interior_score = (
    0.4 * amp_score +           # strong PFM signal
    0.3 * dist_from_wall_norm + # well inside domain, away from wall
    0.2 * (1 - phase_grad_norm) + # uniform phase (stable domain)
    0.1 * freq_std_norm         # some local variation (interesting spots)
)
# Apply domain mask weighting (boost domain interior pixels)
domain_interior_score = domain_interior_score * (domain_mask.astype(float) * 0.7 + 0.3)

# ---- DOMAIN WALL SCORE ----
# Want: inside wall mask, phase gradient (transition), amplitude dip (wall signature)
# centered in wall band
domain_wall_score = (
    0.35 * phase_grad_norm +      # strong phase transition
    0.35 * dist_inside_wall_norm + # centered in wall band
    0.2 * (1 - amp_score) +       # amplitude dip at wall
    0.1 * freq_std_norm           # local stiffness variation
)
# Apply wall mask weighting
domain_wall_score = domain_wall_score * (wall_mask.astype(float) * 0.7 + 0.3)

# ---- COMBINE: both regions should be well-sampled for comparison ----
# Equal weight to both types — the task is to COMPARE them
# Smooth slightly to avoid single-pixel artifacts
from scipy.ndimage import gaussian_filter
domain_interior_score_sm = gaussian_filter(domain_interior_score, sigma=1.5)
domain_wall_score_sm = gaussian_filter(domain_wall_score, sigma=1.5)

# Combined importance: max of both (so both types get high scores)
combined = np.maximum(domain_interior_score_sm, domain_wall_score_sm)

# Normalize to 0-100
combined_min = combined.min()
combined_max = combined.max()
importance_map = 100.0 * (combined - combined_min) / (combined_max - combined_min)

print("Importance map: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    importance_map.min(), importance_map.max(), importance_map.mean()))
print("All finite:", np.all(np.isfinite(importance_map)))

# Save
np.save('importance_map.npy', importance_map)
print("Saved importance_map.npy")


# --- next cell ---

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
from scipy import ndimage

# Reload for visualization
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)
importance_map = np.load('importance_map.npy')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Phase1 background
axes[0,0].imshow(phase1, cmap='RdBu', vmin=-90, vmax=270)
axes[0,0].set_title('Phase1 (PFM - domain contrast)')
axes[0,0].axis('off')

# Amplitude1
axes[0,1].imshow(amp1, cmap='hot')
axes[0,1].set_title('Amplitude1 (signal quality)')
axes[0,1].axis('off')

# Masks overlay
overlay = np.zeros((*phase1.shape, 3))
overlay[domain_mask] = [0.2, 0.6, 1.0]  # blue = domain interior
overlay[wall_mask] = [1.0, 0.4, 0.0]    # orange = wall
axes[0,2].imshow(overlay)
axes[0,2].set_title('Masks: blue=domain, orange=wall')
axes[0,2].axis('off')

# Importance map
im = axes[1,0].imshow(importance_map, cmap='hot', vmin=0, vmax=100)
axes[1,0].set_title('Importance Map')
axes[1,0].axis('off')
plt.colorbar(im, ax=axes[1,0], fraction=0.046)

# Importance overlaid on phase1
axes[1,1].imshow(phase1, cmap='gray', vmin=-90, vmax=270, alpha=0.5)
axes[1,1].imshow(importance_map, cmap='hot', alpha=0.6, vmin=0, vmax=100)
axes[1,1].set_title('Importance over Phase1')
axes[1,1].axis('off')

# Histogram of importance values
axes[1,2].hist(importance_map.ravel(), bins=50, color='steelblue', edgecolor='k')
axes[1,2].set_xlabel('Importance score')
axes[1,2].set_ylabel('Pixel count')
axes[1,2].set_title('Importance distribution')

# Print stats per region
print("Mean importance in domain interior: {:.1f}".format(importance_map[domain_mask].mean()))
print("Mean importance at domain walls: {:.1f}".format(importance_map[wall_mask].mean()))
print("Mean importance elsewhere: {:.1f}".format(importance_map[~domain_mask & ~wall_mask].mean()))

plt.tight_layout()
plt.savefig('importance_analysis.png', dpi=100)
plt.close()
print("Saved importance_analysis.png")
