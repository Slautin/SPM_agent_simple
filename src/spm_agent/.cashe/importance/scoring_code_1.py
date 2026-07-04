
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

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

print("Shapes:", height.shape, amp1.shape, phase1.shape, phase2.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))
print("\nPhase2 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Phase1 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase1.min(), phase1.max(), phase1.mean()))


# --- next cell ---

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

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

print("Shapes:", height.shape, amp1.shape, phase1.shape, phase2.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))
print("\nPhase2 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Phase1 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase1.min(), phase1.max(), phase1.mean()))
print("Amp1 stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(amp1.min(), amp1.max(), amp1.mean()))
print("Amp2 stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(amp2.min(), amp2.max(), amp2.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'

import numpy as np
from scipy import ndimage

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

print("Shapes:", height.shape, amp1.shape, phase1.shape, phase2.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))
print("\nPhase2 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase2.min(), phase2.max(), phase2.mean()))
print("Phase1 stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(phase1.min(), phase1.max(), phase1.mean()))
print("Amp1 stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(amp1.min(), amp1.max(), amp1.mean()))
print("Amp2 stats: min={:.3e}, max={:.3e}, mean={:.3e}".format(amp2.min(), amp2.max(), amp2.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
from scipy import ndimage
import sys
sys.path.insert(0, '')

# Reload data
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

# Save for reuse
np.save('height.npy', height)
np.save('amp1.npy', amp1)
np.save('amp2.npy', amp2)
np.save('phase1.npy', phase1)
np.save('phase2.npy', phase2)
np.save('freq.npy', freq)
np.save('domain_mask.npy', domain_mask)
np.save('wall_mask.npy', wall_mask)

# --- Criterion 1: Domain wall proximity ---
# Wall-mediated switching happens AT and NEAR domain walls
# Compute distance transform from wall pixels
wall_bool = wall_mask.astype(bool)
# Distance from nearest wall pixel (in pixels)
dist_from_wall = ndimage.distance_transform_edt(~wall_bool)
print("dist_from_wall: min={:.1f}, max={:.1f}, mean={:.1f}".format(
    dist_from_wall.min(), dist_from_wall.max(), dist_from_wall.mean()))

# --- Criterion 2: Phase2 gradient magnitude (local polarization contrast) ---
# High gradient = sharp domain boundary = wall region with strong contrast
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
print("phase2_grad: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase2_grad.min(), phase2_grad.max(), phase2_grad.mean()))

# --- Criterion 3: Amplitude2 suppression near walls ---
# In PFM, amplitude dips at domain walls (cancellation of opposite polarizations)
# Low amplitude near walls = genuine wall, good for switching study
amp2_norm = (amp2 - amp2.min()) / (amp2.max() - amp2.min())
print("amp2_norm: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    amp2_norm.min(), amp2_norm.max(), amp2_norm.mean()))

# --- Criterion 4: Wall curvature / complexity ---
# Curved or complex walls are more susceptible to nucleation-mediated switching
# Use local wall density in a neighborhood
wall_density = ndimage.uniform_filter(wall_bool.astype(float), size=15)
print("wall_density: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    wall_density.min(), wall_density.max(), wall_density.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
from scipy import ndimage

# Reload saved data
height = np.load('height.npy')
amp1 = np.load('amp1.npy')
amp2 = np.load('amp2.npy')
phase1 = np.load('phase1.npy')
phase2 = np.load('phase2.npy')
freq = np.load('freq.npy')
domain_mask = np.load('domain_mask.npy')
wall_mask = np.load('wall_mask.npy')

wall_bool = wall_mask.astype(bool)
dist_from_wall = ndimage.distance_transform_edt(~wall_bool)

# ============================================================
# CRITERION 1: Wall proximity score
# Highest at wall pixels, decays with distance
# Use exponential decay with sigma ~ 8 pixels (reasonable for wall-mediated switching)
sigma_prox = 8.0
wall_proximity = np.exp(-dist_from_wall / sigma_prox)
print("wall_proximity: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    wall_proximity.min(), wall_proximity.max(), wall_proximity.mean()))

# ============================================================
# CRITERION 2: Phase2 gradient magnitude (polarization contrast at walls)
# Smooth to reduce noise, then normalize
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase2_grad_smooth = ndimage.gaussian_filter(phase2_grad, sigma=2)
# Clip at 99th percentile to avoid outlier dominance
p99_grad = np.percentile(phase2_grad_smooth, 99)
phase2_grad_norm = np.clip(phase2_grad_smooth / p99_grad, 0, 1)
print("phase2_grad_norm: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase2_grad_norm.min(), phase2_grad_norm.max(), phase2_grad_norm.mean()))

# ============================================================
# CRITERION 3: Amplitude2 suppression (low amplitude = wall node)
# At domain walls, PFM amplitude dips due to cancellation
# Invert: low amplitude → high score, but only near walls
amp2_smooth = ndimage.gaussian_filter(amp2, sigma=2)
amp2_norm = (amp2_smooth - amp2_smooth.min()) / (amp2_smooth.max() - amp2_smooth.min())
amp2_suppression = 1.0 - amp2_norm  # low amplitude → high score
print("amp2_suppression: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    amp2_suppression.min(), amp2_suppression.max(), amp2_suppression.mean()))

# ============================================================
# CRITERION 4: Local wall complexity / curvature
# Regions where wall density is intermediate (0.3-0.7) are curved/complex walls
# These are prime sites for nucleation-mediated switching
wall_density = ndimage.uniform_filter(wall_bool.astype(float), size=15)
# Complexity: peaks at intermediate density (curved walls), not solid wall blocks
wall_complexity = 4.0 * wall_density * (1.0 - wall_density)  # parabola peaking at 0.5
print("wall_complexity: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    wall_complexity.min(), wall_complexity.max(), wall_complexity.mean()))

# ============================================================
# CRITERION 5: Phase1 vs Phase2 discordance (mixed/ambiguous polarization state)
# Regions where phase1 and phase2 disagree may indicate metastable wall configurations
# Normalize both phases to [0,1] range
def norm01(x):
    mn, mx = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - mn) / (mx - mn + 1e-30), 0, 1)

p1n = norm01(phase1)
p2n = norm01(phase2)
phase_discordance = np.abs(p1n - p2n)
phase_discordance_smooth = ndimage.gaussian_filter(phase_discordance, sigma=3)
phase_discordance_norm = phase_discordance_smooth / (phase_discordance_smooth.max() + 1e-30)
print("phase_discordance_norm: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase_discordance_norm.min(), phase_discordance_norm.max(), phase_discordance_norm.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
from scipy import ndimage

# Reload
height = np.load('height.npy')
amp1 = np.load('amp1.npy')
amp2 = np.load('amp2.npy')
phase1 = np.load('phase1.npy')
phase2 = np.load('phase2.npy')
freq = np.load('freq.npy')
domain_mask = np.load('domain_mask.npy')
wall_mask = np.load('wall_mask.npy')

wall_bool = wall_mask.astype(bool)
dist_from_wall = ndimage.distance_transform_edt(~wall_bool)

# --- Recompute all criteria ---
# C1: Wall proximity
sigma_prox = 8.0
wall_proximity = np.exp(-dist_from_wall / sigma_prox)

# C2: Phase2 gradient (polarization contrast)
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase2_grad_smooth = ndimage.gaussian_filter(phase2_grad, sigma=2)
p99_grad = np.percentile(phase2_grad_smooth, 99)
phase2_grad_norm = np.clip(phase2_grad_smooth / p99_grad, 0, 1)

# C3: Amplitude2 suppression (low amp = wall node)
amp2_smooth = ndimage.gaussian_filter(amp2, sigma=2)
amp2_norm = (amp2_smooth - amp2_smooth.min()) / (amp2_smooth.max() - amp2_smooth.min())
amp2_suppression = 1.0 - amp2_norm

# C4: Wall complexity (curved/complex walls)
wall_density = ndimage.uniform_filter(wall_bool.astype(float), size=15)
wall_complexity = 4.0 * wall_density * (1.0 - wall_density)

# C5: Phase discordance (metastable regions)
def norm01(x):
    mn, mx = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - mn) / (mx - mn + 1e-30), 0, 1)

p1n = norm01(phase1)
p2n = norm01(phase2)
phase_discordance = np.abs(p1n - p2n)
phase_discordance_smooth = ndimage.gaussian_filter(phase_discordance, sigma=3)
phase_discordance_norm = phase_discordance_smooth / (phase_discordance_smooth.max() + 1e-30)

# ============================================================
# COMBINE: weighted sum
# Wall-mediated switching: primary interest is AT walls and just inside domains
# Weights: wall proximity is most critical, then gradient contrast, then complexity
w1 = 0.35  # wall proximity
w2 = 0.25  # phase2 gradient (sharp contrast = real wall)
w3 = 0.15  # amplitude suppression (PFM wall signature)
w4 = 0.15  # wall complexity (curved walls = nucleation sites)
w5 = 0.10  # phase discordance (metastable)

importance_raw = (w1 * wall_proximity +
                  w2 * phase2_grad_norm +
                  w3 * amp2_suppression +
                  w4 * wall_complexity +
                  w5 * phase_discordance_norm)

print("importance_raw: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    importance_raw.min(), importance_raw.max(), importance_raw.mean()))

# Scale to 0-100
imp_min = importance_raw.min()
imp_max = importance_raw.max()
importance_map = 100.0 * (importance_raw - imp_min) / (imp_max - imp_min)

print("importance_map: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    importance_map.min(), importance_map.max(), importance_map.mean()))
print("All finite:", np.all(np.isfinite(importance_map)))

np.save('importance_map.npy', importance_map)
print("Saved importance_map.npy")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Reload
phase2 = np.load('phase2.npy')
wall_mask = np.load('wall_mask.npy')
domain_mask = np.load('domain_mask.npy')
importance_map = np.load('importance_map.npy')

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Phase2 (base channel for segmentation)
ax = axes[0, 0]
im = ax.imshow(phase2, cmap='RdBu_r', origin='upper')
ax.set_title('Phase2Retrace (Channel_004)')
plt.colorbar(im, ax=ax)

# 2. Wall mask
ax = axes[0, 1]
ax.imshow(wall_mask, cmap='hot', origin='upper')
ax.set_title('Domain Wall Mask')

# 3. Domain mask
ax = axes[0, 2]
ax.imshow(domain_mask, cmap='Blues', origin='upper')
ax.set_title('Domain Mask')

# 4. Importance map
ax = axes[1, 0]
im = ax.imshow(importance_map, cmap='hot', origin='upper', vmin=0, vmax=100)
ax.set_title('Importance Map (0-100)')
plt.colorbar(im, ax=ax)

# 5. Importance map overlaid with wall contour
ax = axes[1, 1]
im = ax.imshow(importance_map, cmap='hot', origin='upper', vmin=0, vmax=100)
ax.contour(wall_mask, levels=[0.5], colors='cyan', linewidths=1)
ax.set_title('Importance + Wall Contour (cyan)')
plt.colorbar(im, ax=ax)

# 6. Importance map overlaid on phase2
ax = axes[1, 2]
ax.imshow(phase2, cmap='gray', origin='upper', alpha=0.6)
im = ax.imshow(importance_map, cmap='hot', origin='upper', alpha=0.5, vmin=0, vmax=100)
ax.set_title('Importance over Phase2')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('importance_overview.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved importance_overview.png")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Reload all
height = np.load('height.npy')
amp1 = np.load('amp1.npy')
amp2 = np.load('amp2.npy')
phase1 = np.load('phase1.npy')
phase2 = np.load('phase2.npy')
freq = np.load('freq.npy')
domain_mask = np.load('domain_mask.npy')
wall_mask = np.load('wall_mask.npy')
importance_map = np.load('importance_map.npy')

wall_bool = wall_mask.astype(bool)
dist_from_wall = ndimage.distance_transform_edt(~wall_bool)

# Recompute individual criteria for visualization
sigma_prox = 8.0
wall_proximity = np.exp(-dist_from_wall / sigma_prox)

gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase2_grad_smooth = ndimage.gaussian_filter(phase2_grad, sigma=2)
p99_grad = np.percentile(phase2_grad_smooth, 99)
phase2_grad_norm = np.clip(phase2_grad_smooth / p99_grad, 0, 1)

amp2_smooth = ndimage.gaussian_filter(amp2, sigma=2)
amp2_norm_c = (amp2_smooth - amp2_smooth.min()) / (amp2_smooth.max() - amp2_smooth.min())
amp2_suppression = 1.0 - amp2_norm_c

wall_density = ndimage.uniform_filter(wall_bool.astype(float), size=15)
wall_complexity = 4.0 * wall_density * (1.0 - wall_density)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

titles = ['C1: Wall Proximity (w=0.35)', 'C2: Phase2 Gradient (w=0.25)',
          'C3: Amp2 Suppression (w=0.15)', 'C4: Wall Complexity (w=0.15)',
          'FINAL Importance Map', 'Importance + Wall Overlay']
data = [wall_proximity, phase2_grad_norm, amp2_suppression, wall_complexity, importance_map/100, importance_map]
cmaps = ['hot', 'hot', 'hot', 'hot', 'hot', 'hot']

for i, (ax, title, d, cmap) in enumerate(zip(axes.flat, titles, data, cmaps)):
    if i == 5:
        ax.imshow(phase2, cmap='gray', origin='upper', alpha=0.5)
        im = ax.imshow(d, cmap=cmap, origin='upper', alpha=0.6, vmin=0, vmax=100)
        ax.contour(wall_mask, levels=[0.5], colors='cyan', linewidths=0.8)
    else:
        im = ax.imshow(d, cmap=cmap, origin='upper')
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax)

plt.suptitle('Wall-Mediated Switching: Importance Criteria Breakdown', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('criteria_breakdown.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved criteria_breakdown.png")

# Print final stats
print("\nFinal importance_map stats:")
print("  min={:.2f}, max={:.2f}, mean={:.2f}, std={:.2f}".format(
    importance_map.min(), importance_map.max(), importance_map.mean(), importance_map.std()))
print("  Pixels > 80: {:.1f}%".format(100*(importance_map > 80).mean()))
print("  Pixels > 60: {:.1f}%".format(100*(importance_map > 60).mean()))
print("  Pixels > 40: {:.1f}%".format(100*(importance_map > 40).mean()))


# --- next cell ---

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
