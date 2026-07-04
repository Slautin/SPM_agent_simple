
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

print("Shapes:", height.shape, domain_mask.shape, wall_mask.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase2 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
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

print("Shapes:", height.shape, domain_mask.shape, wall_mask.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase2 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Amp2 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp2.min(), amp2.max(), amp2.mean()))
print("Freq stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(freq.min(), freq.max(), freq.mean()))

# Check overlap between masks
overlap = (domain_mask > 0) & (wall_mask > 0)
print("Mask overlap fraction:", overlap.mean())


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
# Remove cached matplotlib if present
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load all channels
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')

domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

print("Shapes:", height.shape, domain_mask.shape, wall_mask.shape)
print("Domain mask dtype:", domain_mask.dtype, "unique:", np.unique(domain_mask))
print("Wall mask dtype:", wall_mask.dtype, "unique:", np.unique(wall_mask))
print("Phase2 stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Amp2 stats: min={:.2e}, max={:.2e}, mean={:.2e}".format(amp2.min(), amp2.max(), amp2.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload data
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(float)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(float)

# Visualize the key channels and masks
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0,0].imshow(phase2, cmap='RdBu_r'); axes[0,0].set_title('Phase2 (PFM)')
axes[0,1].imshow(amp2, cmap='hot'); axes[0,1].set_title('Amplitude2')
axes[0,2].imshow(domain_mask, cmap='gray'); axes[0,2].set_title('Domain mask')
axes[0,3].imshow(wall_mask, cmap='gray'); axes[0,3].set_title('Wall mask')

axes[1,0].imshow(phase1, cmap='RdBu_r'); axes[1,0].set_title('Phase1')
axes[1,1].imshow(amp1, cmap='hot'); axes[1,1].set_title('Amplitude1')
axes[1,2].imshow(height, cmap='gray'); axes[1,2].set_title('Height')
axes[1,3].imshow(freq, cmap='viridis'); axes[1,3].set_title('Frequency')

plt.tight_layout()
plt.savefig('overview.png', dpi=100)
plt.close()
print("Saved overview.png")

# Check overlap
overlap = (domain_mask > 0) & (wall_mask > 0)
print("Mask overlap fraction:", overlap.mean())
print("Domain only:", ((domain_mask > 0) & (wall_mask == 0)).mean())
print("Wall only:", ((wall_mask > 0) & (domain_mask == 0)).mean())
print("Neither:", ((domain_mask == 0) & (wall_mask == 0)).mean())


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload data
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(float)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(float)

# ---- CRITERION 1: PFM Amplitude (switching signal strength) ----
# High amplitude = strong piezoelectric response = reliable switching measurement
# Normalize amp2 to 0-1
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())

# ---- CRITERION 2: Phase contrast / domain identity ----
# Phase2 encodes polarization direction. We want clear domain state (not ambiguous)
# Normalize phase2 to 0-1 range
phase2_norm = (phase2 - phase2.min()) / (phase2.max() - phase2.min())
# Phase clarity: distance from midpoint (ambiguous) → prefer extreme values
phase2_mid = 0.5
phase2_clarity = np.abs(phase2_norm - phase2_mid) * 2  # 0=ambiguous, 1=clear

# ---- CRITERION 3: Distance to domain wall (for domain interior scoring) ----
# Compute distance transform from wall mask
wall_binary = wall_mask > 0
dist_from_wall = ndimage.distance_transform_edt(~wall_binary)
# Normalize
dist_from_wall_norm = np.clip(dist_from_wall / np.percentile(dist_from_wall, 95), 0, 1)

# ---- CRITERION 4: Proximity to domain wall (for wall scoring) ----
# Gaussian-smoothed wall mask gives a "nearness to wall" score
wall_proximity = ndimage.gaussian_filter(wall_mask.astype(float), sigma=5)
wall_proximity = np.clip(wall_proximity / wall_proximity.max(), 0, 1)

print("amp2_norm range:", amp2_norm.min(), amp2_norm.max())
print("phase2_clarity range:", phase2_clarity.min(), phase2_clarity.max())
print("dist_from_wall_norm range:", dist_from_wall_norm.min(), dist_from_wall_norm.max())
print("wall_proximity range:", wall_proximity.min(), wall_proximity.max())


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload data
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(float)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(float)

# ---- Normalize amp2 ----
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())

# ---- Phase clarity (how well-defined the polarization state is) ----
phase2_norm = (phase2 - phase2.min()) / (phase2.max() - phase2.min())
phase2_clarity = np.abs(phase2_norm - 0.5) * 2  # 0=ambiguous, 1=clear

# ---- Distance from wall (interior domain score) ----
wall_binary = wall_mask > 0
dist_from_wall = ndimage.distance_transform_edt(~wall_binary)
dist_from_wall_norm = np.clip(dist_from_wall / np.percentile(dist_from_wall[dist_from_wall > 0], 95), 0, 1)

# ---- Wall proximity score ----
wall_proximity = ndimage.gaussian_filter(wall_mask.astype(float), sigma=4)
wall_proximity = np.clip(wall_proximity / wall_proximity.max(), 0, 1)

# ---- Phase gradient (local variation = interesting switching boundary) ----
gy, gx = np.gradient(phase2)
phase_grad = np.sqrt(gx**2 + gy**2)
phase_grad_norm = np.clip(phase_grad / np.percentile(phase_grad, 99), 0, 1)

# ---- DOMAIN INTERIOR SCORE ----
# Want: inside domain (domain_mask=1), high amplitude, clear phase, away from wall
# These are the "clean" domain interior points for baseline switching
domain_interior = (domain_mask > 0) & (wall_mask == 0)  # strictly inside domain, not wall
domain_score = (
    0.4 * amp2_norm +
    0.4 * phase2_clarity +
    0.2 * dist_from_wall_norm
)
domain_score = domain_score * domain_interior.astype(float)

# ---- DOMAIN WALL SCORE ----
# Want: near/at wall, high amplitude (good signal), phase gradient (actual wall)
wall_region = wall_mask > 0
wall_score = (
    0.4 * amp2_norm +
    0.3 * phase_grad_norm +
    0.3 * wall_proximity
)
wall_score = wall_score * wall_region.astype(float)

print("Domain score stats: mean={:.3f}, max={:.3f}".format(domain_score[domain_interior].mean(), domain_score.max()))
print("Wall score stats: mean={:.3f}, max={:.3f}".format(wall_score[wall_region].mean(), wall_score.max()))

# Visualize components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(amp2_norm, cmap='hot'); axes[0,0].set_title('Amp2 norm')
axes[0,1].imshow(phase2_clarity, cmap='viridis'); axes[0,1].set_title('Phase2 clarity')
axes[0,2].imshow(dist_from_wall_norm, cmap='plasma'); axes[0,2].set_title('Dist from wall (norm)')
axes[1,0].imshow(wall_proximity, cmap='plasma'); axes[1,0].set_title('Wall proximity')
axes[1,1].imshow(domain_score, cmap='hot'); axes[1,1].set_title('Domain interior score')
axes[1,2].imshow(wall_score, cmap='hot'); axes[1,2].set_title('Wall score')
plt.tight_layout()
plt.savefig('components.png', dpi=100)
plt.close()
print("Saved components.png")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload data
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(float)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(float)

# ---- Normalize amp2 ----
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())
amp1_norm = (amp1 - amp1.min()) / (amp1.max() - amp1.min())
# Combined amplitude (both PFM channels)
amp_combined = 0.5 * amp2_norm + 0.5 * amp1_norm

# ---- Phase clarity ----
phase2_norm = (phase2 - phase2.min()) / (phase2.max() - phase2.min())
phase2_clarity = np.abs(phase2_norm - 0.5) * 2

# ---- Distance from wall ----
wall_binary = wall_mask > 0
dist_from_wall = ndimage.distance_transform_edt(~wall_binary)
dist_from_wall_norm = np.clip(dist_from_wall / np.percentile(dist_from_wall[dist_from_wall > 0], 95), 0, 1)

# ---- Wall proximity (soft) ----
wall_proximity = ndimage.gaussian_filter(wall_mask.astype(float), sigma=4)
wall_proximity = np.clip(wall_proximity / wall_proximity.max(), 0, 1)

# ---- Phase gradient ----
gy, gx = np.gradient(phase2)
phase_grad = np.sqrt(gx**2 + gy**2)
phase_grad_norm = np.clip(phase_grad / np.percentile(phase_grad, 99), 0, 1)

# ---- DOMAIN INTERIOR SCORE ----
# Strictly inside domain, not wall region
# High amplitude + clear phase + away from wall = ideal for baseline switching
domain_interior = (domain_mask > 0) & (wall_mask == 0)
domain_score = (
    0.45 * amp_combined +
    0.35 * phase2_clarity +
    0.20 * dist_from_wall_norm
) * domain_interior.astype(float)

# ---- DOMAIN WALL SCORE ----
# At/near wall: high amplitude + phase gradient (real wall) + wall proximity
wall_region = wall_mask > 0
wall_score = (
    0.40 * amp_combined +
    0.35 * phase_grad_norm +
    0.25 * wall_proximity
) * wall_region.astype(float)

# ---- COMBINE: both regions are equally important for comparison ----
# Scale each to 0-100 independently, then combine
def scale_to_100(arr, mask):
    """Scale non-zero region to 0-100"""
    out = np.zeros_like(arr)
    vals = arr[mask > 0]
    if vals.max() > vals.min():
        out[mask > 0] = (vals - vals.min()) / (vals.max() - vals.min()) * 100
    return out

domain_score_scaled = scale_to_100(domain_score, domain_interior)
wall_score_scaled = scale_to_100(wall_score, wall_region)

# Combine: both regions get full 0-100 range
# Overlap pixels (both masks) get max of both
importance_map = np.maximum(domain_score_scaled, wall_score_scaled)

# Pixels in neither mask get a small background score based on amplitude
neither = (domain_mask == 0) & (wall_mask == 0)
importance_map[neither] = amp_combined[neither] * 10  # low background

print("Importance map stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    importance_map.min(), importance_map.max(), importance_map.mean()))
print("Domain interior pixels mean score:", importance_map[domain_interior].mean())
print("Wall pixels mean score:", importance_map[wall_region].mean())

# Final visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(domain_score_scaled, cmap='hot', vmin=0, vmax=100)
axes[0].set_title('Domain Interior Score (0-100)')
plt.colorbar(axes[0].images[0], ax=axes[0])

axes[1].imshow(wall_score_scaled, cmap='hot', vmin=0, vmax=100)
axes[1].set_title('Wall Score (0-100)')
plt.colorbar(axes[1].images[0], ax=axes[1])

im = axes[2].imshow(importance_map, cmap='hot', vmin=0, vmax=100)
axes[2].set_title('Combined Importance Map')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig('importance_preview.png', dpi=100)
plt.close()
print("Saved importance_preview.png")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import sys
for key in list(sys.modules.keys()):
    if 'matplotlib' in key:
        del sys.modules[key]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload data
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(float)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(float)

# ---- Normalize amp2 ----
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())
amp1_norm = (amp1 - amp1.min()) / (amp1.max() - amp1.min())
amp_combined = 0.5 * amp2_norm + 0.5 * amp1_norm

# ---- Phase clarity ----
phase2_norm = (phase2 - phase2.min()) / (phase2.max() - phase2.min())
phase2_clarity = np.abs(phase2_norm - 0.5) * 2

# ---- Distance from wall ----
wall_binary = wall_mask > 0
dist_from_wall = ndimage.distance_transform_edt(~wall_binary)
dist_from_wall_norm = np.clip(dist_from_wall / np.percentile(dist_from_wall[dist_from_wall > 0], 95), 0, 1)

# ---- Wall proximity (soft) ----
wall_proximity = ndimage.gaussian_filter(wall_mask.astype(float), sigma=4)
wall_proximity = np.clip(wall_proximity / wall_proximity.max(), 0, 1)

# ---- Phase gradient ----
gy, gx = np.gradient(phase2)
phase_grad = np.sqrt(gx**2 + gy**2)
phase_grad_norm = np.clip(phase_grad / np.percentile(phase_grad, 99), 0, 1)

# ---- DOMAIN INTERIOR SCORE ----
domain_interior = (domain_mask > 0) & (wall_mask == 0)
domain_score_raw = (
    0.45 * amp_combined +
    0.35 * phase2_clarity +
    0.20 * dist_from_wall_norm
) * domain_interior.astype(float)

# ---- DOMAIN WALL SCORE ----
wall_region = wall_mask > 0
wall_score_raw = (
    0.40 * amp_combined +
    0.35 * phase_grad_norm +
    0.25 * wall_proximity
) * wall_region.astype(float)

# ---- Scale each region independently to 0-100 ----
def scale_to_100(arr, mask):
    out = np.zeros_like(arr)
    vals = arr[mask > 0]
    if vals.max() > vals.min():
        out[mask > 0] = (vals - vals.min()) / (vals.max() - vals.min()) * 100
    return out

domain_score_scaled = scale_to_100(domain_score_raw, domain_interior)
wall_score_scaled = scale_to_100(wall_score_raw, wall_region)

# ---- Combine ----
importance_map = np.maximum(domain_score_scaled, wall_score_scaled)

# Background (neither mask) gets low score
neither = (domain_mask == 0) & (wall_mask == 0)
importance_map[neither] = amp_combined[neither] * 10

# Ensure finite and in range
importance_map = np.clip(importance_map, 0, 100)
assert np.all(np.isfinite(importance_map))
print("Final importance map: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    importance_map.min(), importance_map.max(), importance_map.mean()))

# Save
np.save('importance_map.npy', importance_map)
print("Saved importance_map.npy")

# Final figure with overlay
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Phase2 as background
axes[0].imshow(phase2, cmap='RdBu_r', alpha=1.0)
axes[0].set_title('Phase2 (PFM) - context')

im = axes[1].imshow(importance_map, cmap='inferno', vmin=0, vmax=100)
plt.colorbar(im, ax=axes[1], label='Importance (0-100)')
axes[1].set_title('Importance Map\n(domain interior + wall regions)')

# Overlay wall contour
axes[1].contour(wall_mask, levels=[0.5], colors='cyan', linewidths=0.8, alpha=0.7)

plt.tight_layout()
plt.savefig('final_importance_map.png', dpi=120)
plt.close()
print("Saved final_importance_map.png")
