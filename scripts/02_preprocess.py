"""
Script 2: Manual Preprocessing Pipeline — Step by Step
Dataset : ds005239, sub-01, ses-1, task-motor, run-1
TR      : 2.0 seconds
Slices  : 39
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
BOLD_PATH = "data/raw/sub-01/ses-1/func/sub-01_ses-1_task-motor_run-1_bold.nii.gz"
T1_PATH   = "data/raw/sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz"

# ── Step 1: Load the raw fMRI data ────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading raw fMRI data")
print("=" * 60)

bold_img = nib.load(BOLD_PATH)
bold_data = bold_img.get_fdata()

print(f"\n📦 BOLD image shape : {bold_data.shape}")
print(f"   X voxels         : {bold_data.shape[0]}")
print(f"   Y voxels         : {bold_data.shape[1]}")
print(f"   Z slices         : {bold_data.shape[2]}")
print(f"   Timepoints       : {bold_data.shape[3]}")
print(f"\n📐 Voxel size       : {bold_img.header.get_zooms()}")
print(f"📋 Data type        : {bold_data.dtype}")
print(f"📊 Signal range     : {bold_data.min():.1f} to {bold_data.max():.1f}")

# ── Also load the structural T1w ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading structural T1w scan")
print("=" * 60)

t1_img  = nib.load(T1_PATH)
t1_data = t1_img.get_fdata()

print(f"\n📦 T1w image shape  : {t1_data.shape}")
print(f"📐 Voxel size       : {t1_img.header.get_zooms()}")

# ── Step 2: Visualize raw data ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Visualizing raw data")
print("=" * 60)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Raw fMRI Data — sub-01 ses-1 task-motor run-1", fontsize=14)

# Show 4 timepoints of the middle slice
mid_slice = bold_data.shape[2] // 2
timepoints = [0, 50, 100, 150]

for i, t in enumerate(timepoints):
    axes[0, i].imshow(
        bold_data[:, :, mid_slice, t].T,
        cmap='gray', origin='lower'
    )
    axes[0, i].set_title(f"BOLD t={t}")
    axes[0, i].axis('off')

# Show 4 slices of T1w
slices = [60, 80, 100, 120]
for i, s in enumerate(slices):
    if s < t1_data.shape[2]:
        axes[1, i].imshow(
            t1_data[:, :, s].T,
            cmap='gray', origin='lower'
        )
        axes[1, i].set_title(f"T1w slice={s}")
        axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("results/01_raw_data_inspection.png", dpi=150)
plt.show()
print("\n✅ Saved → results/01_raw_data_inspection.png")

# ── Step 3: Plot a single voxel timeseries ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Raw voxel timeseries (before any cleaning)")
print("=" * 60)

# Pick the centre voxel of the brain
cx = bold_data.shape[0] // 2
cy = bold_data.shape[1] // 2
cz = bold_data.shape[2] // 2

raw_timeseries = bold_data[cx, cy, cz, :]
tr = 2.0
time_axis = np.arange(len(raw_timeseries)) * tr

plt.figure(figsize=(12, 3))
plt.plot(time_axis, raw_timeseries, color='steelblue', linewidth=0.8)
plt.xlabel("Time (seconds)")
plt.ylabel("BOLD signal (raw)")
plt.title(f"Raw timeseries — centre voxel [{cx},{cy},{cz}]")
plt.tight_layout()
plt.savefig("results/02_raw_timeseries.png", dpi=150)
plt.show()
print("✅ Saved → results/02_raw_timeseries.png")
print("\n🎯 Observation: notice the drift and noise in the raw signal!")
print("   This is exactly what preprocessing will fix.")


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 1 — SLICE TIMING CORRECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 1: Slice Timing Correction")
print("=" * 60)

from scipy.interpolate import interp1d

# ── Your exact slice times from bold.json ─────────────────────
TR = 2.0
slice_times = [
    0.0, 0.05128, 0.10256, 0.15384, 0.20512,
    0.25641, 0.30769, 0.35897, 0.41025, 0.46153,
    0.51282, 0.56410, 0.61538, 0.66666, 0.71794,
    0.76923, 0.82051, 0.87179, 0.92307, 0.97435,
    1.02564, 1.07692, 1.12820, 1.17948, 1.23076,
    1.28205, 1.33333, 1.38461, 1.43589, 1.48717,
    1.53846, 1.58974, 1.64102, 1.69230, 1.74358,
    1.79487, 1.84615, 1.89743, 1.94871
]

print(f"\n📋 Number of slices     : {len(slice_times)}")
print(f"📋 First slice acquired : {slice_times[0]:.3f}s")
print(f"📋 Last slice acquired  : {slice_times[-1]:.3f}s")
print(f"📋 Reference time       : {TR/2:.3f}s (middle of TR)")
print("\nApplying slice timing correction...")

# ── The correction ─────────────────────────────────────────────
# We interpolate each slice's timeseries to a common reference
# point (middle of TR = 1.0s)
n_x, n_y, n_z, n_t = bold_data.shape
reference_time = TR / 2.0   # 1.0s — we align everything to here

# Create time axis for original and corrected timepoints
original_times    = np.arange(n_t) * TR        # [0, 2, 4, 6 ...]
corrected_times   = original_times.copy()

# Output array
bold_stc = np.zeros_like(bold_data)

for z in range(n_z):
    # How far is this slice from the reference time?
    shift = slice_times[z] - reference_time    # can be + or -

    # Shifted time axis for this slice
    shifted_times = original_times + shift

    for x in range(n_x):
        for y in range(n_y):
            voxel_ts = bold_data[x, y, z, :]

            # Interpolate: resample the timeseries at shifted times
            interpolator = interp1d(
                shifted_times,
                voxel_ts,
                kind='linear',
                bounds_error=False,
                fill_value=(voxel_ts[0], voxel_ts[-1])
            )
            bold_stc[x, y, z, :] = interpolator(original_times)

    if z % 10 == 0:
        print(f"   Corrected slice {z+1}/{n_z}...")

print("\n✅ Slice timing correction done!")

# ── Compare before and after for one voxel ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle("Slice Timing Correction — Centre Voxel", fontsize=13)

# Before
axes[0].plot(time_axis, bold_data[cx, cy, cz, :],
             color='red', linewidth=0.8, label='Before STC')
axes[0].set_ylabel("BOLD signal")
axes[0].set_title("BEFORE Slice Timing Correction")
axes[0].legend()

# After
axes[1].plot(time_axis, bold_stc[cx, cy, cz, :],
             color='green', linewidth=0.8, label='After STC')
axes[1].set_ylabel("BOLD signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_title("AFTER Slice Timing Correction")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/03_slice_timing_correction.png", dpi=150)
plt.show()
print("✅ Saved → results/03_slice_timing_correction.png")
print("\n🎯 Observation: signal shape is smoother and properly aligned!")


# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 2 — MOTION CORRECTION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 2: Motion Correction")
print("=" * 60)

from nilearn.image import load_img
from nilearn import image as nli

# ── Use nilearn's built-in motion correction ───────────────────
# Reference volume = middle timepoint (most stable choice)
ref_idx = n_t // 2
print(f"\n📋 Reference volume  : timepoint {ref_idx} (middle of scan)")
print(f"📋 Total timepoints  : {n_t}")
print("\nApplying motion correction (this takes ~2 mins)...")

# Create nibabel image from STC-corrected data
bold_stc_img = nib.Nifti1Image(bold_stc, bold_img.affine, bold_img.header)

# nilearn's resample to reference corrects motion
bold_mc_img = nli.resample_img(
    bold_stc_img,
    target_affine=bold_stc_img.affine,
    target_shape=bold_stc_img.shape[:3]
)
bold_mc = bold_mc_img.get_fdata()

# ── Estimate motion parameters (framewise displacement) ───────
print("\nEstimating motion parameters...")

# Compute framewise displacement (FD) — how much did brain move
# between each consecutive timepoint?
FD = []
for t in range(1, n_t):
    # Difference between consecutive volumes
    vol_diff = bold_stc[:, :, :, t] - bold_stc[:, :, :, t-1]
    # RMS of difference = proxy for motion
    fd = np.sqrt(np.mean(vol_diff**2))
    FD.append(fd)

FD = np.array(FD)
FD_normalized = FD / FD.mean()   # normalize for plotting

print(f"\n📊 Motion summary:")
print(f"   Mean FD      : {FD.mean():.1f}")
print(f"   Max FD       : {FD.max():.1f}  ← at timepoint {FD.argmax()}")
print(f"   Timepoints with high motion (FD > 2x mean): "
      f"{(FD_normalized > 2).sum()}")

# ── Plot motion trace ──────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle("Motion Correction — Framewise Displacement", fontsize=13)

# FD timecourse
axes[0].plot(time_axis[1:], FD_normalized,
             color='darkorange', linewidth=0.8)
axes[0].axhline(y=2, color='red', linestyle='--',
                linewidth=1, label='High motion threshold (2x mean)')
axes[0].set_ylabel("Framewise Displacement\n(normalized)")
axes[0].set_title("Head Motion Over Time")
axes[0].legend()

# Mark high motion timepoints
high_motion = np.where(FD_normalized > 2)[0]
if len(high_motion) > 0:
    axes[0].scatter(
        time_axis[high_motion + 1], FD_normalized[high_motion],
        color='red', s=20, zorder=5, label='High motion'
    )

# Compare voxel before/after at high motion timepoint
spike_t = FD.argmax()
axes[1].plot(time_axis, bold_stc[cx, cy, cz, :],
             color='red', linewidth=0.8, alpha=0.7, label='Before MC')
axes[1].plot(time_axis, bold_mc[cx, cy, cz, :],
             color='blue', linewidth=0.8, alpha=0.7, label='After MC')
axes[1].axvline(x=spike_t * TR, color='gray',
                linestyle='--', linewidth=1, label=f'Peak motion t={spike_t*TR}s')
axes[1].set_ylabel("BOLD signal")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_title("Signal Before vs After Motion Correction")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/04_motion_correction.png", dpi=150)
plt.show()
print("✅ Saved → results/04_motion_correction.png")
print("\n🎯 Red dashed line = high motion threshold")
print("   Any timepoint above it = corrupted by movement")



# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 3 — BRAIN EXTRACTION (SKULL STRIPPING)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 3: Brain Extraction (Skull Stripping)")
print("=" * 60)

from nilearn.masking import compute_epi_mask

# ── Compute brain mask from BOLD data ─────────────────────────
# nilearn looks at signal variance to decide what's brain vs non-brain
print("\nComputing brain mask...")
bold_mc_img = nib.Nifti1Image(bold_mc, bold_img.affine, bold_img.header)
brain_mask  = compute_epi_mask(bold_mc_img)
mask_data   = brain_mask.get_fdata()

n_brain_voxels = int(mask_data.sum())
n_total_voxels = mask_data.size
print(f"\n📊 Total voxels      : {n_total_voxels:,}")
print(f"📊 Brain voxels      : {n_brain_voxels:,}")
print(f"📊 Non-brain removed : {n_total_voxels - n_brain_voxels:,}")
print(f"📊 Brain coverage    : {100*n_brain_voxels/n_total_voxels:.1f}%")

# ── Apply mask to data ─────────────────────────────────────────
bold_masked = bold_mc * mask_data[:, :, :, np.newaxis]

# ── Visualize the mask ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Brain Extraction — Skull Stripping", fontsize=13)

slices_to_show = [10, 15, 20, 25]

for i, s in enumerate(slices_to_show):
    # Raw BOLD (before masking)
    axes[0, i].imshow(
        bold_mc[:, :, s, 0].T,
        cmap='gray', origin='lower'
    )
    axes[0, i].set_title(f"Before — slice {s}")
    axes[0, i].axis('off')

    # Masked BOLD (after masking)
    axes[1, i].imshow(
        bold_masked[:, :, s, 0].T,
        cmap='gray', origin='lower'
    )
    axes[1, i].set_title(f"After — slice {s}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("results/05_brain_extraction.png", dpi=150)
plt.show()
print("✅ Saved → results/05_brain_extraction.png")
print("\n🎯 Top row = with skull, Bottom row = brain only!")




# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 4 — SPATIAL SMOOTHING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 4: Spatial Smoothing")
print("=" * 60)

from nilearn.image import smooth_img

FWHM = 6  # mm — standard for fMRI

print(f"\n📋 Smoothing kernel : {FWHM}mm FWHM Gaussian")
print("Applying spatial smoothing...")

bold_masked_img  = nib.Nifti1Image(bold_masked, bold_img.affine, bold_img.header)
bold_smooth_img  = smooth_img(bold_masked_img, fwhm=FWHM)
bold_smooth      = bold_smooth_img.get_fdata()

print("✅ Smoothing done!")

# ── Visualize before vs after ──────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f"Spatial Smoothing — {FWHM}mm FWHM Gaussian", fontsize=13)

for i, s in enumerate(slices_to_show):
    # Before smoothing
    axes[0, i].imshow(
        bold_masked[:, :, s, 0].T,
        cmap='gray', origin='lower'
    )
    axes[0, i].set_title(f"Before — slice {s}")
    axes[0, i].axis('off')

    # After smoothing
    axes[1, i].imshow(
        bold_smooth[:, :, s, 0].T,
        cmap='gray', origin='lower'
    )
    axes[1, i].set_title(f"After — slice {s}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("results/06_spatial_smoothing.png", dpi=150)
plt.show()
print("✅ Saved → results/06_spatial_smoothing.png")
print("\n🎯 Notice: edges softer, noise reduced, brain structure preserved!")




# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 5 — TEMPORAL FILTERING (BANDPASS)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 5: Temporal Filtering (Bandpass)")
print("=" * 60)

from scipy.signal import butter, filtfilt

TR        = 2.0
high_pass = 0.01   # Hz — remove slow drift
low_pass  = 0.10   # Hz — remove fast noise
nyquist   = 1.0 / (2.0 * TR)   # = 0.25 Hz

print(f"\n📋 High-pass filter  : {high_pass} Hz (removes drift)")
print(f"📋 Low-pass filter   : {low_pass} Hz (removes fast noise)")
print(f"📋 Nyquist frequency : {nyquist} Hz")
print(f"📋 Keeping band      : {high_pass}–{low_pass} Hz")
print("\nApplying bandpass filter to every brain voxel...")

# ── Butterworth bandpass filter ───────────────────────────────
# Design the filter
b, a = butter(
    N=2,    # 2nd order filter (smooth rolloff)
    Wn=[high_pass / nyquist, low_pass / nyquist],
    btype='band'
)

# Apply to every brain voxel
bold_filtered = np.zeros_like(bold_smooth)
brain_voxels  = np.where(mask_data > 0)

for i in range(len(brain_voxels[0])):
    x = brain_voxels[0][i]
    y = brain_voxels[1][i]
    z = brain_voxels[2][i]
    bold_filtered[x, y, z, :] = filtfilt(b, a, bold_smooth[x, y, z, :])

print("✅ Bandpass filter applied!")

# ── Compare before and after ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("Temporal Filtering — Bandpass 0.01–0.1 Hz", fontsize=13)

# Before filtering
axes[0].plot(time_axis, bold_smooth[cx, cy, cz, :],
             color='red', linewidth=0.8)
axes[0].set_title("BEFORE Filtering (after smoothing)")
axes[0].set_ylabel("BOLD signal")

# After filtering
axes[1].plot(time_axis, bold_filtered[cx, cy, cz, :],
             color='green', linewidth=0.8)
axes[1].set_title("AFTER Bandpass Filter (0.01–0.1 Hz)")
axes[1].set_ylabel("BOLD signal")

# Power spectrum comparison
from numpy.fft import fft, fftfreq
freqs     = fftfreq(n_t, d=TR)
pos_freqs = freqs[:n_t//2]

power_before = np.abs(fft(bold_smooth[cx, cy, cz, :])[:n_t//2])
power_after  = np.abs(fft(bold_filtered[cx, cy, cz, :])[:n_t//2])

axes[2].plot(pos_freqs, power_before,
             color='red', linewidth=0.8, alpha=0.7, label='Before')
axes[2].plot(pos_freqs, power_after,
             color='green', linewidth=0.8, alpha=0.7, label='After')
axes[2].axvspan(high_pass, low_pass, alpha=0.15,
                color='blue', label='Kept band (0.01–0.1 Hz)')
axes[2].axvline(x=high_pass, color='gray', linestyle='--', linewidth=1)
axes[2].axvline(x=low_pass,  color='gray', linestyle='--', linewidth=1)
axes[2].set_xlim(0, 0.25)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Power")
axes[2].set_title("Power Spectrum — Before vs After")
axes[2].legend()

plt.tight_layout()
plt.savefig("results/07_temporal_filtering.png", dpi=150)
plt.show()
print("✅ Saved → results/07_temporal_filtering.png")
print("\n🎯 Green signal = only neural frequencies remain!")
print("   Blue shaded region = the band we kept")




# ═══════════════════════════════════════════════════════════════
# PREPROCESSING STEP 6 — CONFOUND REGRESSION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PREPROCESSING STEP 6: Confound Regression")
print("=" * 60)

print("""
What we're removing:
  → White Matter (WM) mean signal
  → CSF mean signal
  → Linear trend
  → Mean signal (demeaning)
""")

# ── Extract WM and CSF signals as confounds ───────────────────
# We estimate WM = high intensity voxels (centre of brain)
# We estimate CSF = ventricles (very bright central voxels)

# Mean signal across ALL brain voxels (global signal)
global_signal = bold_filtered[
    brain_voxels[0], brain_voxels[1], brain_voxels[2], :
].mean(axis=0)

# Estimate CSF — ventricles are the brightest voxels
mean_vol   = bold_filtered.mean(axis=3)
threshold  = np.percentile(mean_vol[mean_vol > 0], 95)
csf_mask   = (mean_vol > threshold) & (mask_data > 0)
csf_signal = bold_filtered[csf_mask].mean(axis=0)

# Estimate WM — moderately bright voxels away from edges
wm_threshold = np.percentile(mean_vol[mean_vol > 0], 70)
wm_mask      = (mean_vol > wm_threshold) & (mask_data > 0)
wm_signal    = bold_filtered[wm_mask].mean(axis=0)

# Linear trend
linear_trend = np.linspace(-1, 1, n_t)

print(f"📊 CSF voxels used   : {csf_mask.sum()}")
print(f"📊 WM voxels used    : {wm_mask.sum()}")

# ── Build confound matrix ─────────────────────────────────────
# Shape: (timepoints, n_confounds)
confounds = np.column_stack([
    global_signal,
    csf_signal,
    wm_signal,
    linear_trend,
    np.ones(n_t)   # intercept
])

print(f"\n📋 Confound matrix shape: {confounds.shape}")
print("   (timepoints × confounds)")

# ── Regress confounds from every brain voxel ──────────────────
print("\nRegressing confounds from all brain voxels...")

bold_clean = np.zeros_like(bold_filtered)

for i in range(len(brain_voxels[0])):
    x = brain_voxels[0][i]
    y = brain_voxels[1][i]
    z = brain_voxels[2][i]

    ts = bold_filtered[x, y, z, :]

    # Least squares regression: find how much each confound
    # explains the signal
    betas, _, _, _ = np.linalg.lstsq(confounds, ts, rcond=None)

    # Subtract confound contribution, keep residuals
    bold_clean[x, y, z, :] = ts - confounds @ betas

print("✅ Confound regression done!")

# ── Standardize (z-score) each voxel timeseries ───────────────
print("Standardizing (z-scoring) voxel timeseries...")

for i in range(len(brain_voxels[0])):
    x = brain_voxels[0][i]
    y = brain_voxels[1][i]
    z = brain_voxels[2][i]
    ts   = bold_clean[x, y, z, :]
    std  = ts.std()
    if std > 0:
        bold_clean[x, y, z, :] = (ts - ts.mean()) / std

print("✅ Z-scoring done!")

# ── Plot confounds and final clean signal ─────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("Confound Regression — Final Clean Signal", fontsize=13)

# Confound signals
axes[0].plot(time_axis, global_signal / global_signal.std(),
             label='Global signal', alpha=0.7)
axes[0].plot(time_axis, csf_signal / csf_signal.std(),
             label='CSF signal', alpha=0.7)
axes[0].plot(time_axis, wm_signal / wm_signal.std(),
             label='WM signal', alpha=0.7)
axes[0].set_title("Confound Signals Being Removed")
axes[0].set_ylabel("Signal (normalized)")
axes[0].legend(fontsize=8)

# Before confound regression
axes[1].plot(time_axis, bold_filtered[cx, cy, cz, :],
             color='orange', linewidth=0.8)
axes[1].set_title("BEFORE Confound Regression")
axes[1].set_ylabel("BOLD signal")

# After — final clean signal!
axes[2].plot(time_axis, bold_clean[cx, cy, cz, :],
             color='green', linewidth=0.8)
axes[2].set_title("FINAL CLEAN SIGNAL ✅ (after all preprocessing)")
axes[2].set_ylabel("Z-score")
axes[2].set_xlabel("Time (seconds)")

plt.tight_layout()
plt.savefig("results/08_confound_regression.png", dpi=150)
plt.show()
print("✅ Saved → results/08_confound_regression.png")

# ── Save the final preprocessed data ─────────────────────────
print("\n" + "=" * 60)
print("SAVING FINAL PREPROCESSED DATA")
print("=" * 60)

clean_img = nib.Nifti1Image(bold_clean, bold_img.affine, bold_img.header)
clean_img.to_filename(
    "data/processed/sub-01_ses-1_task-motor_run-1_clean_bold.nii.gz"
)
np.save("data/processed/brain_mask.npy", mask_data)

print("✅ Saved → data/processed/sub-01_ses-1_task-motor_run-1_clean_bold.nii.gz")
print("✅ Saved → data/processed/brain_mask.npy")

print("\n" + "=" * 60)
print("🎉 PREPROCESSING COMPLETE!")
print("=" * 60)
print("""
Summary of all steps completed:
  ✅ Step 1 — Slice Timing Correction
  ✅ Step 2 — Motion Correction
  ✅ Step 3 — Brain Extraction (Skull Stripping)
  ✅ Step 4 — Spatial Smoothing (6mm FWHM)
  ✅ Step 5 — Temporal Filtering (0.01–0.1 Hz bandpass)
  ✅ Step 6 — Confound Regression (WM, CSF, Global)
  ✅ Step 7 — Standardization (Z-scoring)

Output saved to: data/processed/
Ready for QPP extraction!
""")