
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

print("Shapes:", height.shape, amp1.shape, phase1.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))

# Quick overview figure
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0,0].imshow(height, cmap='gray'); axes[0,0].set_title('Height')
axes[0,1].imshow(amp1, cmap='gray'); axes[0,1].set_title('Amplitude1')
axes[0,2].imshow(amp2, cmap='gray'); axes[0,2].set_title('Amplitude2')
axes[0,3].imshow(phase1, cmap='gray'); axes[0,3].set_title('Phase1')
axes[1,0].imshow(phase2, cmap='gray'); axes[1,0].set_title('Phase2')
axes[1,1].imshow(freq, cmap='gray'); axes[1,1].set_title('Frequency')
axes[1,2].imshow(domain_mask, cmap='Reds'); axes[1,2].set_title('Domain mask')
axes[1,3].imshow(wall_mask, cmap='Reds'); axes[1,3].set_title('Wall mask')
plt.tight_layout()
plt.savefig('overview.png', dpi=100)
plt.close()
print("Saved overview.png")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
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

print("Shapes:", height.shape, amp1.shape, phase1.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes[0,0].imshow(height, cmap='gray'); axes[0,0].set_title('Height')
axes[0,1].imshow(amp1, cmap='gray'); axes[0,1].set_title('Amplitude1')
axes[0,2].imshow(amp2, cmap='gray'); axes[0,2].set_title('Amplitude2')
axes[0,3].imshow(phase1, cmap='gray'); axes[0,3].set_title('Phase1')
axes[1,0].imshow(phase2, cmap='gray'); axes[1,0].set_title('Phase2')
axes[1,1].imshow(freq, cmap='gray'); axes[1,1].set_title('Frequency')
axes[1,2].imshow(domain_mask, cmap='Reds'); axes[1,2].set_title('Domain mask')
axes[1,3].imshow(wall_mask, cmap='Reds'); axes[1,3].set_title('Wall mask')
plt.tight_layout()
plt.savefig('overview.png', dpi=100)
plt.close()
print("Saved overview.png")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Reload
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

# ---- Criterion 1: Domain wall proximity ----
# Pixels near domain walls are prime candidates for wall-mediated switching
# Extract the wall boundary (edges of wall_mask) and compute distance transform
wall_boundary = ndimage.binary_dilation(wall_mask, iterations=1) & ~wall_mask
# Distance from every pixel to the nearest wall pixel
dist_to_wall = ndimage.distance_transform_edt(~wall_mask)
print("dist_to_wall stats: min={:.1f}, max={:.1f}, mean={:.1f}".format(
    dist_to_wall.min(), dist_to_wall.max(), dist_to_wall.mean()))

# ---- Criterion 2: Phase1 gradient magnitude (local polarization contrast) ----
# High gradient = sharp domain boundary = wall region with strong contrast
gy1, gx1 = np.gradient(phase1)
phase1_grad = np.sqrt(gx1**2 + gy1**2)
print("phase1_grad stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase1_grad.min(), phase1_grad.max(), phase1_grad.mean()))

# ---- Criterion 3: Phase2 gradient magnitude ----
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
print("phase2_grad stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase2_grad.min(), phase2_grad.max(), phase2_grad.mean()))

# ---- Criterion 4: Amplitude suppression at walls ----
# Low amplitude at domain walls is a hallmark of wall regions (destructive interference)
# Normalize amp1 so that LOW amplitude = HIGH score
amp1_norm = (amp1 - amp1.min()) / (amp1.max() - amp1.min())
amp1_suppression = 1.0 - amp1_norm  # high where amplitude is low

# ---- Criterion 5: Domain-wall interface (pixels in domain mask adjacent to wall mask) ----
# These are the exact switching nucleation sites
domain_near_wall = domain_mask & ndimage.binary_dilation(wall_mask, iterations=3)
print("Domain near wall fraction:", domain_near_wall.mean())

# ---- Criterion 6: Phase1 local variance (heterogeneous polarization = switching-prone) ----
from scipy.ndimage import uniform_filter
phase1_mean = uniform_filter(phase1, size=7)
phase1_var = uniform_filter((phase1 - phase1_mean)**2, size=7)
print("phase1_var stats: min={:.3f}, max={:.3f}, mean={:.3f}".format(
    phase1_var.min(), phase1_var.max(), phase1_var.mean()))


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

# Reload
height = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/HeightRetrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

def normalize_01(arr):
    mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
    out = np.clip((arr - mn) / (mx - mn + 1e-30), 0, 1)
    return out

# ---- Score 1: Wall proximity score ----
# Pixels ON the wall or very close to it score highest
dist_to_wall = ndimage.distance_transform_edt(~wall_mask)
# Gaussian decay: sigma ~ 8 pixels (wall influence zone)
sigma_wall = 8.0
wall_prox = np.exp(-dist_to_wall**2 / (2 * sigma_wall**2))
wall_prox_norm = normalize_01(wall_prox)

# ---- Score 2: Phase1 gradient (polarization contrast at walls) ----
gy1, gx1 = np.gradient(phase1)
phase1_grad = np.sqrt(gx1**2 + gy1**2)
# Smooth slightly to reduce noise
phase1_grad_sm = ndimage.gaussian_filter(phase1_grad, sigma=1.5)
phase1_grad_norm = normalize_01(phase1_grad_sm)

# ---- Score 3: Phase2 gradient ----
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase2_grad_sm = ndimage.gaussian_filter(phase2_grad, sigma=1.5)
phase2_grad_norm = normalize_01(phase2_grad_sm)

# ---- Score 4: Amplitude1 suppression (low amp = wall region) ----
amp1_sm = ndimage.gaussian_filter(amp1, sigma=1.5)
amp1_suppression = normalize_01(1.0 - amp1_sm)

# ---- Score 5: Phase1 local variance (heterogeneous polarization) ----
phase1_mean_loc = uniform_filter(phase1, size=9)
phase1_var = uniform_filter((phase1 - phase1_mean_loc)**2, size=9)
phase1_var_norm = normalize_01(phase1_var)

# ---- Score 6: Domain-wall interface bonus ----
# Pixels that are in domain regions AND close to walls — nucleation sites
domain_near_wall_dist = ndimage.distance_transform_edt(~wall_mask)
domain_interface = domain_mask.astype(float) * np.exp(-domain_near_wall_dist**2 / (2 * 5.0**2))
domain_interface_norm = normalize_01(domain_interface)

# ---- Score 7: Frequency anomaly (local stiffness variation at walls) ----
freq_mean = uniform_filter(freq, size=9)
freq_var = uniform_filter((freq - freq_mean)**2, size=9)
freq_var_norm = normalize_01(freq_var)

# Print correlation with wall_mask to check relevance
for name, arr in [('wall_prox', wall_prox_norm), ('phase1_grad', phase1_grad_norm),
                   ('phase2_grad', phase2_grad_norm), ('amp1_supp', amp1_suppression),
                   ('phase1_var', phase1_var_norm), ('domain_iface', domain_interface_norm),
                   ('freq_var', freq_var_norm)]:
    corr = np.corrcoef(wall_mask.ravel().astype(float), arr.ravel())[0,1]
    print(f"{name}: corr with wall_mask = {corr:.3f}, mean on wall = {arr[wall_mask].mean():.3f}, mean off wall = {arr[~wall_mask].mean():.3f}")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

# Reload
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

def normalize_01(arr):
    mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
    out = np.clip((arr - mn) / (mx - mn + 1e-30), 0, 1)
    return out

# Check amp1 variation
print("amp1 std:", amp1.std(), "range:", amp1.max()-amp1.min())
print("amp1 on wall:", amp1[wall_mask].mean(), "off wall:", amp1[~wall_mask].mean())
print("amp2 on wall:", amp2[wall_mask].mean(), "off wall:", amp2[~wall_mask].mean())

# amp1 suppression - fix: use proper normalization
amp1_suppression = normalize_01(-amp1)  # negate so low amp = high score
print("amp1_supp corr:", np.corrcoef(wall_mask.ravel().astype(float), amp1_suppression.ravel())[0,1])

# Check phase1 on wall vs domain
print("\nphase1 on wall:", phase1[wall_mask].mean(), "std:", phase1[wall_mask].std())
print("phase1 on domain:", phase1[domain_mask].mean(), "std:", phase1[domain_mask].std())
print("phase2 on wall:", phase2[wall_mask].mean(), "std:", phase2[wall_mask].std())
print("phase2 on domain:", phase2[domain_mask].mean(), "std:", phase2[domain_mask].std())


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

# Reload all
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

def normalize_01(arr):
    mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - mn) / (mx - mn + 1e-30), 0, 1)

# ============================================================
# SCORING CRITERIA for wall-mediated switching
# ============================================================

# 1. WALL PROXIMITY: pixels on/near domain walls are the primary sites
#    Use a soft distance decay so the wall itself + immediate neighborhood scores high
dist_to_wall = ndimage.distance_transform_edt(~wall_mask)
sigma_wall = 6.0
S_wall_prox = np.exp(-dist_to_wall**2 / (2 * sigma_wall**2))
S_wall_prox = normalize_01(S_wall_prox)

# 2. PHASE1 GRADIENT: sharp polarization contrast = well-defined wall = switching front
gy1, gx1 = np.gradient(phase1)
phase1_grad = np.sqrt(gx1**2 + gy1**2)
phase1_grad_sm = ndimage.gaussian_filter(phase1_grad, sigma=1.5)
S_phase1_grad = normalize_01(phase1_grad_sm)

# 3. PHASE2 GRADIENT: second harmonic also captures wall sharpness
gy2, gx2 = np.gradient(phase2)
phase2_grad = np.sqrt(gx2**2 + gy2**2)
phase2_grad_sm = ndimage.gaussian_filter(phase2_grad, sigma=1.5)
S_phase2_grad = normalize_01(phase2_grad_sm)

# 4. AMPLITUDE SUPPRESSION: low PFM amplitude at walls = depolarization zone
#    Both harmonics show this; combine them
amp_combined = (amp1 + amp2) / 2.0
S_amp_supp = normalize_01(-amp_combined)  # invert: low amp → high score

# 5. PHASE1 LOCAL VARIANCE: heterogeneous polarization = switching-prone region
phase1_mean_loc = uniform_filter(phase1, size=9)
phase1_var = uniform_filter((phase1 - phase1_mean_loc)**2, size=9)
S_phase1_var = normalize_01(phase1_var)

# 6. DOMAIN-WALL INTERFACE: domain pixels immediately adjacent to walls
#    These are nucleation sites where wall-mediated switching initiates
dist_from_wall_in_domain = dist_to_wall * domain_mask.astype(float)
# Pixels in domain AND close to wall
S_domain_iface = domain_mask.astype(float) * np.exp(-dist_to_wall**2 / (2 * 4.0**2))
S_domain_iface = normalize_01(S_domain_iface)

# ============================================================
# COMBINE: weighted sum
# Weights reflect importance for wall-mediated switching:
#   - Wall proximity is the primary criterion
#   - Phase gradients confirm wall sharpness (switching front quality)
#   - Amplitude suppression confirms wall character
#   - Phase variance captures heterogeneous/metastable regions
#   - Domain-interface bonus for nucleation sites
# ============================================================
w_wall_prox   = 0.30
w_phase1_grad = 0.20
w_phase2_grad = 0.15
w_amp_supp    = 0.15
w_phase1_var  = 0.10
w_domain_iface= 0.10

importance = (w_wall_prox   * S_wall_prox +
              w_phase1_grad * S_phase1_grad +
              w_phase2_grad * S_phase2_grad +
              w_amp_supp    * S_amp_supp +
              w_phase1_var  * S_phase1_var +
              w_domain_iface* S_domain_iface)

# Scale to 0-100
importance = (importance - importance.min()) / (importance.max() - importance.min()) * 100.0

print("Importance map stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    importance.min(), importance.max(), importance.mean()))
print("Mean importance on wall:", importance[wall_mask].mean())
print("Mean importance off wall:", importance[~wall_mask].mean())
print("Mean importance on domain:", importance[domain_mask].mean())

# Save
np.save('importance_map.npy', importance)
print("Saved importance_map.npy")


# --- next cell ---

import os
os.environ['MPLBACKEND'] = 'agg'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

# Reload for visualization
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)
importance = np.load('importance_map.npy')

def normalize_01(arr):
    mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - mn) / (mx - mn + 1e-30), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Phase1 with wall overlay
ax = axes[0]
ax.imshow(phase1, cmap='gray', vmin=np.percentile(phase1,1), vmax=np.percentile(phase1,99))
wall_overlay = np.zeros((*wall_mask.shape, 4))
wall_overlay[wall_mask] = [1, 0, 0, 0.4]
ax.imshow(wall_overlay)
ax.set_title('Phase1 + Wall mask (red)', fontsize=12)
ax.axis('off')

# Panel 2: Importance map
ax = axes[1]
im = ax.imshow(importance, cmap='hot', vmin=0, vmax=100)
plt.colorbar(im, ax=ax, label='Importance (0-100)')
ax.set_title('Importance Map\n(wall-mediated switching)', fontsize=12)
ax.axis('off')

# Panel 3: Importance overlaid on Phase1
ax = axes[2]
ax.imshow(phase1, cmap='gray', vmin=np.percentile(phase1,1), vmax=np.percentile(phase1,99))
imp_rgba = plt.cm.hot(importance / 100.0)
imp_rgba[..., 3] = np.clip(importance / 100.0 * 0.75, 0, 0.75)
ax.imshow(imp_rgba)
ax.set_title('Importance overlay on Phase1', fontsize=12)
ax.axis('off')

plt.tight_layout()
plt.savefig('importance_visualization.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved importance_visualization.png")

# Also show individual score components
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

def normalize_01(arr):
    mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - mn) / (mx - mn + 1e-30), 0, 1)

dist_to_wall = ndimage.distance_transform_edt(~wall_mask)
S_wall_prox = normalize_01(np.exp(-dist_to_wall**2 / (2 * 6.0**2)))

gy1, gx1 = np.gradient(phase1)
S_phase1_grad = normalize_01(ndimage.gaussian_filter(np.sqrt(gx1**2 + gy1**2), sigma=1.5))

amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
S_amp_supp = normalize_01(-((amp1 + amp2) / 2.0))

phase1_mean_loc = uniform_filter(phase1, size=9)
S_phase1_var = normalize_01(uniform_filter((phase1 - phase1_mean_loc)**2, size=9))

S_domain_iface = normalize_01(domain_mask.astype(float) * np.exp(-dist_to_wall**2 / (2 * 4.0**2)))

scores = [S_wall_prox, S_phase1_grad, S_amp_supp, S_phase1_var, S_domain_iface, importance/100]
titles = ['Wall Proximity', 'Phase1 Gradient', 'Amp Suppression',
          'Phase1 Local Var', 'Domain-Wall Interface', 'FINAL IMPORTANCE']

for ax, sc, ti in zip(axes.ravel(), scores, titles):
    im = ax.imshow(sc, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(ti, fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('score_components.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved score_components.png")


# --- next cell ---

import numpy as np
imp = np.load('importance_map.npy')
print("Shape:", imp.shape)
print("dtype:", imp.dtype)
print("min:", imp.min(), "max:", imp.max(), "mean:", imp.mean())
print("finite:", np.all(np.isfinite(imp)))
print("Values in [0,100]:", (imp >= 0).all() and (imp <= 100).all())
