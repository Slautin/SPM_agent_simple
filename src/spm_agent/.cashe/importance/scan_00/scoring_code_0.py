
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

print("Shapes:", height.shape, amp1.shape, phase1.shape, freq.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase1 range:", phase1.min(), phase1.max(), phase1.mean())
print("Phase2 range:", phase2.min(), phase2.max(), phase2.mean())
print("Freq range:", freq.min(), freq.max(), freq.mean())
print("Amp1 range:", amp1.min(), amp1.max(), amp1.mean())
print("Amp2 range:", amp2.min(), amp2.max(), amp2.mean())


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

print("Shapes:", height.shape, amp1.shape, phase1.shape, freq.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase1 range:", phase1.min(), phase1.max(), phase1.mean())
print("Phase2 range:", phase2.min(), phase2.max(), phase2.mean())
print("Freq range:", freq.min(), freq.max(), freq.mean())
print("Amp1 range:", amp1.min(), amp1.max(), amp1.mean())
print("Amp2 range:", amp2.min(), amp2.max(), amp2.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))


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

# Load masks
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

print("Shapes:", height.shape, amp1.shape, phase1.shape, freq.shape)
print("Domain mask coverage:", domain_mask.mean())
print("Wall mask coverage:", wall_mask.mean())
print("Phase1 range:", phase1.min(), phase1.max(), phase1.mean())
print("Phase2 range:", phase2.min(), phase2.max(), phase2.mean())
print("Freq range:", freq.min(), freq.max(), freq.mean())
print("Amp1 range:", amp1.min(), amp1.max(), amp1.mean())
print("Amp2 range:", amp2.min(), amp2.max(), amp2.mean())
print("Domain mask unique:", np.unique(domain_mask))
print("Wall mask unique:", np.unique(wall_mask))


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
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy')
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy')

# Visualize all channels
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

# Reload data
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

# The task: compare switching inside domains vs near domain walls
# Key criteria for PFM switching comparison:
# 1. PFM amplitude (piezoresponse signal strength) - higher amplitude = cleaner switching signal
# 2. Domain wall proximity - distance to nearest wall (for wall vs interior comparison)
# 3. Phase contrast (domain polarity) - clear phase contrast = well-defined domain state
# 4. Contact resonance frequency stability - stable freq = reliable measurement
# 5. Amplitude uniformity within domain interior

# --- Criterion 1: PFM Amplitude (average of both drives) ---
# Higher amplitude = stronger piezoresponse = more reliable switching measurement
amp_avg = (amp1 + amp2) / 2.0

# --- Criterion 2: Distance to domain wall (proximity map) ---
# Compute distance transform from wall mask
# Pixels far from walls = domain interior; pixels near walls = wall region
# We want BOTH extremes to be interesting: near-wall AND deep interior
# So we need two separate criteria

# Distance from wall (for interior criterion)
dist_from_wall = ndimage.distance_transform_edt(~wall_mask)
print("dist_from_wall range:", dist_from_wall.min(), dist_from_wall.max())

# Distance to wall (for wall-proximity criterion)
dist_to_wall = ndimage.distance_transform_edt(wall_mask)
print("dist_to_wall range:", dist_to_wall.min(), dist_to_wall.max())

# --- Criterion 3: Phase contrast (absolute phase signal) ---
# In PFM, phase encodes polarization direction. 
# Phase2 is used for domain segmentation, so it's the primary PFM phase channel
# Use phase2 - look at how well-defined the phase is (distance from 90 deg midpoint)
# Phase values cluster around two states; intermediate = ambiguous
# Use both phase channels to get piezoelectric phase signal
# Phase1 and Phase2 should be consistent; their average gives the piezo phase
phase_avg = (phase1 + phase2) / 2.0

# --- Criterion 4: Amplitude ratio (DART quality) ---
# In DART PFM, amp1/amp2 ratio near 1 = on-resonance = best signal quality
# Deviations indicate off-resonance tracking issues
amp_ratio = amp1 / (amp2 + 1e-20)
print("amp_ratio range:", amp_ratio.min(), amp_ratio.max(), amp_ratio.mean())

# --- Criterion 5: Phase gradient (local phase variation) ---
# High gradient = near domain wall or noisy; low gradient = stable domain interior
from scipy.ndimage import gaussian_gradient_magnitude
phase2_grad = gaussian_gradient_magnitude(phase2, sigma=2)
print("phase2_grad range:", phase2_grad.min(), phase2_grad.max())

print("\nDomain mask pixels:", domain_mask.sum())
print("Wall mask pixels:", wall_mask.sum())
print("Overlap domain+wall:", (domain_mask & wall_mask).sum())


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
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter

# Reload data
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

def normalize(arr):
    """Normalize to [0,1]"""
    a_min, a_max = np.percentile(arr, 1), np.percentile(arr, 99)
    out = np.clip((arr - a_min) / (a_max - a_min + 1e-30), 0, 1)
    return out.astype(np.float32)

# ============================================================
# CRITERION 1: PFM Amplitude (signal quality for switching)
# ============================================================
# Average amplitude from both DART drives = piezoresponse strength
# Higher amplitude → stronger electromechanical coupling → more reliable switching signal
amp_avg = (amp1 + amp2) / 2.0
c1_amp = normalize(amp_avg)

# ============================================================
# CRITERION 2: Domain interior score (deep inside domain, far from wall)
# ============================================================
# For comparing switching INSIDE domains: want pixels that are clearly
# within a domain (domain_mask=1) AND far from any wall
dist_from_wall = ndimage.distance_transform_edt(~wall_mask)
# Score = distance from wall, restricted to domain interior
# Pixels in domain_mask and far from wall = ideal interior measurement sites
interior_score = dist_from_wall * domain_mask.astype(float)
c2_interior = normalize(interior_score)

# ============================================================
# CRITERION 3: Domain wall proximity score
# ============================================================
# For comparing switching NEAR domain walls: want pixels that are
# in the wall_mask OR very close to it
# Use inverse distance from wall edge (proximity to wall)
# wall_mask pixels have dist_from_wall = 0, so proximity = 1/(1+dist)
wall_proximity = 1.0 / (1.0 + dist_from_wall)
# Enhance wall region itself
wall_proximity = wall_proximity * (wall_mask.astype(float) + 0.5 * (~wall_mask).astype(float))
c3_wall = normalize(wall_proximity)

# ============================================================
# CRITERION 4: Phase clarity (well-defined polarization state)
# ============================================================
# Phase2 is the primary PFM phase channel (used for domain segmentation)
# Phase values near the two domain states (e.g., ~-90 or ~+90 from midpoint)
# indicate clear polarization; intermediate values = ambiguous/noisy
# Use distance from the midpoint of the phase range
phase2_mid = (phase2.max() + phase2.min()) / 2.0
phase2_clarity = np.abs(phase2 - phase2_mid)
c4_phase = normalize(phase2_clarity)

# ============================================================
# CRITERION 5: DART amplitude balance (measurement reliability)
# ============================================================
# In DART PFM, amp1 ≈ amp2 when tracking is on-resonance
# |amp1 - amp2| / (amp1 + amp2) = imbalance; low imbalance = reliable
amp_sum = amp1 + amp2 + 1e-30
amp_balance = 1.0 - np.abs(amp1 - amp2) / amp_sum  # 1 = perfect balance
c5_balance = normalize(amp_balance)

# ============================================================
# CRITERION 6: Contact resonance frequency (local stiffness)
# ============================================================
# Frequency tracks contact resonance = local mechanical stiffness
# Variations in freq across domain vs wall regions reveal stiffness contrast
# which affects switching coercive field
freq_norm = normalize(freq)

# Check correlations between all criteria
criteria = [c1_amp, c2_interior, c3_wall, c4_phase, c5_balance, freq_norm]
names = ['amp', 'interior', 'wall_prox', 'phase_clarity', 'amp_balance', 'freq']
print("Pairwise correlations:")
for i in range(len(criteria)):
    for j in range(i+1, len(criteria)):
        c = np.corrcoef(criteria[i].ravel(), criteria[j].ravel())[0,1]
        print(f"  {names[i]} vs {names[j]}: {c:.3f}")


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
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase1Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
freq = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/FrequencyRetrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

def normalize(arr):
    a_min, a_max = np.percentile(arr, 1), np.percentile(arr, 99)
    out = np.clip((arr - a_min) / (a_max - a_min + 1e-30), 0, 1)
    return out.astype(np.float32)

# All correlations are well below 0.9 - all criteria are non-redundant
# amp vs freq = 0.543 is the highest - let's check if we should keep both

# Final 5 criteria (all |corr| < 0.9):
# 1. PFM Amplitude (signal quality)
# 2. Domain interior score (deep inside domain)
# 3. Wall proximity score (near domain wall)
# 4. Phase clarity (well-defined polarization)
# 5. DART amplitude balance (measurement reliability)

# Note: freq (0.543 with amp) - keep it as it captures stiffness contrast independently

dist_from_wall = ndimage.distance_transform_edt(~wall_mask)

# C1: PFM Amplitude
amp_avg = (amp1 + amp2) / 2.0
c1_amp = normalize(amp_avg)

# C2: Domain interior (far from wall, inside domain)
interior_score = dist_from_wall * domain_mask.astype(float)
c2_interior = normalize(interior_score)

# C3: Wall proximity (near/at domain wall)
wall_proximity = np.exp(-dist_from_wall / 5.0)  # exponential decay, sigma=5px
# Boost wall pixels themselves
wall_proximity = np.where(wall_mask, 1.0, wall_proximity)
c3_wall = normalize(wall_proximity)

# C4: Phase clarity (distance from phase midpoint = clear domain state)
phase2_mid = (np.percentile(phase2, 1) + np.percentile(phase2, 99)) / 2.0
phase2_clarity = np.abs(phase2 - phase2_mid)
c4_phase = normalize(phase2_clarity)

# C5: DART amplitude balance (on-resonance quality)
amp_sum = amp1 + amp2 + 1e-30
amp_imbalance = np.abs(amp1 - amp2) / amp_sum
amp_balance = 1.0 - amp_imbalance
c5_balance = normalize(amp_balance)

# Final correlation check
criteria_final = [c1_amp, c2_interior, c3_wall, c4_phase, c5_balance]
names_final = ['pfm_amplitude', 'domain_interior', 'wall_proximity', 'phase_clarity', 'dart_balance']
print("Final pairwise correlations:")
for i in range(len(criteria_final)):
    for j in range(i+1, len(criteria_final)):
        c = np.corrcoef(criteria_final[i].ravel(), criteria_final[j].ravel())[0,1]
        flag = " *** HIGH ***" if abs(c) > 0.9 else ""
        print(f"  {names_final[i]} vs {names_final[j]}: {c:.3f}{flag}")

# Save components
components = np.stack(criteria_final, axis=0)
print("\nComponents shape:", components.shape)
print("Value ranges:")
for i, name in enumerate(names_final):
    print(f"  {name}: [{components[i].min():.3f}, {components[i].max():.3f}], mean={components[i].mean():.3f}")
np.save('components.npy', components)
print("Saved components.npy")


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
amp1 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude1Retrace.npy')
amp2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Amplitude2Retrace.npy')
phase2 = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/src/spm_agent/.cashe/current/Phase2Retrace.npy')
domain_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain segmentation_mask.npy').astype(bool)
wall_mask = np.load('/Users/borisslautin/Documents/GitHub/SPM_agent_simple/notebooks/seg_proto/ferroelectric domain wall segmentation_mask.npy').astype(bool)

components = np.load('components.npy')
names = ['PFM Amplitude', 'Domain Interior', 'Wall Proximity', 'Phase Clarity', 'DART Balance']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i in range(5):
    im = axes[i].imshow(components[i], cmap='hot', vmin=0, vmax=1)
    axes[i].set_title(names[i], fontsize=12)
    plt.colorbar(im, ax=axes[i])
    # Overlay wall mask as contour
    axes[i].contour(wall_mask, levels=[0.5], colors='cyan', linewidths=0.5)
    axes[i].contour(domain_mask, levels=[0.5], colors='lime', linewidths=0.5)

# Last panel: phase2 for reference
im = axes[5].imshow(phase2, cmap='RdBu')
axes[5].set_title('Phase2 (reference)', fontsize=12)
plt.colorbar(im, ax=axes[5])
axes[5].contour(wall_mask, levels=[0.5], colors='cyan', linewidths=0.5)

plt.suptitle('Scoring Criteria for Domain vs Wall Switching Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('criteria_maps.png', dpi=100)
plt.close()
print("Saved criteria_maps.png")


# --- next cell ---

import json

# Save components.json
components_json = {
    "names": [
        "pfm_amplitude",
        "domain_interior",
        "wall_proximity",
        "phase_clarity",
        "dart_balance"
    ],
    "weights": [
        1.5,
        2.0,
        2.0,
        1.5,
        1.0
    ],
    "rationale": [
        "Average DART PFM amplitude (amp1+amp2)/2 reflects local electromechanical coupling strength; higher amplitude yields a larger switching signal-to-noise ratio, making switching events more reliably detectable in both domain interior and wall regions.",
        "Distance-transform score inside the segmented domain mask (zero outside domains): pixels deep within a domain are unambiguously in a single-polarization state, providing the cleanest reference for intrinsic domain switching behavior far from wall-mediated nucleation.",
        "Exponential proximity to the domain wall mask (exp(-d/5px), clamped to 1 at wall pixels): domain walls are the locus of depolarization fields and reduced coercive fields, so measurements here directly probe wall-assisted switching mechanisms for comparison with interior switching.",
        "Absolute deviation of Phase2 from its midpoint value: pixels with phase far from the midpoint are in a well-defined polarization state (not in a transitional/noisy region), ensuring that the pre-switching polarization state is unambiguous for both domain and wall sites.",
        "DART amplitude balance 1 - |amp1-amp2|/(amp1+amp2): when both drive amplitudes are equal the tip is tracking the contact resonance symmetrically, minimizing cross-talk and phase errors that would corrupt switching-event detection in local spectroscopy."
    ]
}

with open('components.json', 'w') as f:
    json.dump(components_json, f, indent=2)

print("Saved components.json")
print(json.dumps(components_json, indent=2))


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
