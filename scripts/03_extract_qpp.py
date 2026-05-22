"""
Script 3: QPP Extraction — Quasi-Periodic Pattern
Method  : Majeed et al. (2011) iterative template matching
Dataset : ds005239, sub-01, ses-1, task-motor, run-1
TR      : 2.0 seconds
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
# numpy: our math engine. Handles the huge arrays of brain data.

import nibabel as nib
# nibabel: reads/writes brain scan files (.nii.gz format)

from scipy.signal import find_peaks
# find_peaks: finds the high points (peaks) in a 1D signal.
# Like finding mountain tops on a graph.

import matplotlib.pyplot as plt
# matplotlib: for plotting graphs and brain images.

# ── Parameters ─────────────────────────────────────────────────────────────────
BOLD_PATH   = "data/processed/sub-01_ses-1_task-motor_run-1_clean_bold.nii.gz"
MASK_PATH   = "data/processed/brain_mask.npy"
QPP_OUT     = "results/qpp_template.nii.gz"
PEAKS_OUT   = "results/qpp_peak_times.npy"

TR          = 2.0   # seconds per volume
WINDOW_SEC  = 20    # QPPs last ~20 seconds
N_ITER      = 10    # how many times to refine the template
PEAK_THRESH = 0.2   # minimum correlation to count as a QPP occurrence

# ── Load preprocessed data ─────────────────────────────────────────────────────
print("=" * 60)
print("Loading preprocessed BOLD data...")
print("=" * 60)

img       = nib.load(BOLD_PATH)
# Load the clean .nii.gz file. img is a nibabel image object.

bold      = img.get_fdata()
# Extract the actual numbers into a numpy array.
# Shape: (64, 64, 39, 300) = (X, Y, Z, timepoints)

mask_data = np.load(MASK_PATH)
# Load the brain mask (1s and 0s).
# Shape: (64, 64, 39) — same as one brain volume.

print(f"BOLD shape : {bold.shape}")
print(f"TR         : {TR}s")

# ── Calculate window length ────────────────────────────────────────────────────
window_len = int(WINDOW_SEC / TR)
# How many timepoints in a 20-second window?
# 20 seconds / 2 seconds per timepoint = 10 timepoints
# int() converts to a whole number (no decimals allowed for indexing)

print(f"Window     : {window_len} timepoints ({WINDOW_SEC}s)")

# ── Flatten brain data ─────────────────────────────────────────────────────────
X, Y, Z, T = bold.shape
# Unpack the 4 dimensions into named variables.
# X=64, Y=64, Z=39, T=300

data = bold.reshape(-1, T)
# Reshape from (64, 64, 39, 300) → (159744, 300)
# -1 means "figure out this dimension automatically"
# Now each ROW is one voxel's full timeseries (300 numbers)
# This makes it easier to do math across all voxels at once

# Only keep brain voxels (where mask = 1)
brain_mask_flat = mask_data.reshape(-1)
# Flatten mask from (64,64,39) → (159744,)
# Each element is 1 (brain) or 0 (not brain)

brain_idx = np.where(brain_mask_flat > 0)[0]
# np.where() finds indices where condition is True.
# brain_idx is a list of row numbers that are brain voxels.
# e.g., [5, 12, 13, 14, ...] — the brain voxel positions

data_brain = data[brain_idx, :]
# Select ONLY brain voxel rows from data.
# Shape: (~33647, 300) — only brain voxels, all timepoints

print(f"Brain voxels used: {len(brain_idx):,}")

# ── QPP Algorithm (improved) ───────────────────────────────────────────────────
def compute_correlation(template, data_brain, window_len):
    """
    Slide template across full timeseries, compute correlation at each position.
    Returns correlation array of shape (T - window_len + 1,)
    """
    n_vox, T = data_brain.shape
    n_windows = T - window_len + 1
    corr = np.zeros(n_windows)

    # Flatten and normalise template once (faster than doing it every loop)
    t_flat = template.flatten()
    t_norm = t_flat - t_flat.mean()
    t_std  = t_norm.std()

    if t_std == 0:
        return corr   # flat template → all zeros

    for t in range(n_windows):
        segment  = data_brain[:, t : t + window_len]
        s_flat   = segment.flatten()
        s_norm   = s_flat - s_flat.mean()
        s_std    = s_norm.std()

        if s_std > 0:
            # Pearson correlation: dot product of normalised vectors
            corr[t] = np.dot(t_norm, s_norm) / (len(t_norm) * t_std * s_std)

    return corr


def extract_qpp(data_brain, window_len, n_iter, peak_thresh, n_starts=5):
    """
    Iterative QPP extraction with multiple random starts.
    Runs the algorithm n_starts times and keeps the best result
    (most QPP occurrences with highest mean correlation).
    """
    n_vox, T = data_brain.shape
    rng = np.random.default_rng(42)

    best_peaks    = np.array([])
    best_template = None
    best_corr     = np.zeros(T - window_len + 1)
    best_score    = -1

    for start_num in range(n_starts):
        print(f"\n--- Random start {start_num+1}/{n_starts} ---")

        # Pick a random starting window
        start    = rng.integers(0, T - window_len)
        template = data_brain[:, start : start + window_len].copy()
        thresh   = peak_thresh

        print(f"  Starting at timepoint {start} ({start*2}s)")

        for iteration in range(n_iter):

            # Step 1: compute correlation timecourse
            corr = compute_correlation(template, data_brain, window_len)

            # Step 2: find peaks — use half window_len as minimum distance
            peaks, _ = find_peaks(
                corr,
                height=thresh,
                distance=window_len // 2
                # Allow peaks half a window apart (less strict)
            )

            # If too few peaks, lower threshold
            if len(peaks) < 3:
                thresh *= 0.85
                if thresh < 0.05:
                    # Threshold too low → give up on this start
                    print(f"  Iter {iteration+1}: threshold too low, stopping")
                    break
                print(f"  Iter {iteration+1}: only {len(peaks)} peaks, "
                      f"lowering threshold to {thresh:.3f}")
                continue

            # Step 3: average windows at peaks → new template
            valid_peaks = [p for p in peaks if p + window_len <= T]
            windows = np.array([data_brain[:, p : p + window_len]
                                 for p in valid_peaks])
            # Shape: (n_valid_peaks, n_vox, window_len)

            new_template = windows.mean(axis=0)
            # Average across peaks → refined template

            # Check convergence: how similar is new template to old?
            similarity = np.corrcoef(
                template.flatten(), new_template.flatten()
            )[0, 1]

            template = new_template

            print(f"  Iter {iteration+1}/{n_iter}: "
                  f"{len(valid_peaks)} occurrences | "
                  f"mean corr={corr[valid_peaks].mean():.3f} | "
                  f"template stability={similarity:.3f}")

            # If template barely changed → converged, stop early
            if similarity > 0.999:
                print(f"  Converged at iteration {iteration+1}!")
                break

        # Score this run: n_peaks × mean_correlation
        if len(peaks) > 0:
            score = len(peaks) * corr[peaks].mean()
            print(f"  Score for this start: {score:.3f}")

            if score > best_score:
                best_score    = score
                best_peaks    = peaks
                best_template = template
                best_corr     = corr
                print(f"  ★ New best!")

    print(f"\n{'='*50}")
    print(f"Best result: {len(best_peaks)} QPP occurrences | score={best_score:.3f}")

    return best_template, best_peaks, best_corr


# ── Run QPP extraction ─────────────────────────────────────────────────────────
template, peak_times, corr = extract_qpp(
    data_brain,
    window_len,
    N_ITER,
    PEAK_THRESH,
    n_starts=5   # Try 5 different random starting points
)

print(f"\n✅ QPP extraction complete!")
print(f"   Found {len(peak_times)} QPP occurrences")
print(f"   Peak timepoints: {peak_times}")
print(f"   Peak times (seconds): {peak_times * TR}")

# ── Reshape template back to brain shape ──────────────────────────────────────
# template is currently (n_brain_voxels, window_len)
# We need to put it back into (64, 64, 39, window_len) for saving

qpp_volume = np.zeros((X, Y, Z, window_len))
# Empty 4D array to hold the QPP template.

qpp_volume.reshape(-1, window_len)[brain_idx, :] = template
# Put the template values back into their correct brain positions.

# ── Save results ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

qpp_img = nib.Nifti1Image(qpp_volume, affine=img.affine, header=img.header)
# Wrap QPP template back into nibabel image with same spatial info as original.

nib.save(qpp_img, QPP_OUT)
# Save to results/qpp_template.nii.gz

np.save(PEAKS_OUT, peak_times)
# Save peak timepoints as a numpy array file (.npy)

np.save("results/qpp_correlation_timecourse.npy", corr)
# Also save the full correlation timecourse — useful for plotting

print(f"✅ QPP template  → {QPP_OUT}")
print(f"✅ Peak times    → {PEAKS_OUT}")

# ── Visualize results ──────────────────────────────────────────────────────────
print("\nPlotting results...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("QPP Extraction Results", fontsize=14)

# Plot 1: Correlation timecourse
n_windows = len(corr)
time_axis_corr = np.arange(n_windows) * TR
# Time axis for the correlation timecourse

axes[0].plot(time_axis_corr, corr, color='steelblue', linewidth=0.8)
axes[0].axhline(y=PEAK_THRESH, color='red', linestyle='--',
                linewidth=1, label=f'Threshold ({PEAK_THRESH})')
# axhline draws a horizontal line at y=PEAK_THRESH

if len(peak_times) > 0:
    axes[0].scatter(peak_times * TR, corr[peak_times],
                    color='red', s=40, zorder=5, label='QPP occurrences')
    # zorder=5 puts the dots on top of the line
axes[0].set_title("Correlation Timecourse — Template vs Sliding Window")
axes[0].set_ylabel("Correlation (r)")
axes[0].set_xlabel("Time (seconds)")
axes[0].legend()
axes[0].set_ylim(-1, 1)
# Fix y-axis from -1 to 1 (correlation range)

# Plot 2: Mean brain activity across QPP frames
mean_activity = qpp_volume.mean(axis=(0, 1, 2))
# Average across all X,Y,Z voxels → one number per frame
# axis=(0,1,2) means average across first 3 dimensions
# Result shape: (window_len,) = (10,)

qpp_time = np.arange(window_len) * TR
# Time axis for the QPP template (0, 2, 4, ..., 18 seconds)

axes[1].plot(qpp_time, mean_activity,
             color='green', linewidth=2, marker='o', markersize=6)
axes[1].set_title("QPP Template — Mean Brain Activity Over 20 Seconds")
axes[1].set_ylabel("Mean Z-score")
axes[1].set_xlabel("Time within QPP (seconds)")
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
# Horizontal line at zero for reference

# Plot 3: QPP template — middle slice at peak frame
mid_slice = Z // 2
# Middle slice index: 39 // 2 = 19

peak_frame = np.argmax(np.abs(mean_activity))
# Frame with the strongest activity (positive or negative).
# np.abs() makes negative numbers positive.
# np.argmax() finds the index of the maximum value.

im = axes[2].imshow(
    qpp_volume[:, :, mid_slice, peak_frame].T,
    # .T transposes the 2D slice (flips axes for correct orientation)
    cmap='RdBu_r',
    # Red-Blue colormap: Red=positive BOLD, Blue=negative BOLD
    origin='lower',
    # Origin at bottom-left (neuroimaging convention)
    vmin=-0.5, vmax=0.5
    # Fix color scale symmetrically around zero
)
plt.colorbar(im, ax=axes[2])
# Add colorbar legend showing what colors mean

axes[2].set_title(f"QPP Template — Middle Slice (z={mid_slice}), Frame {peak_frame}")
axes[2].axis('off')
# Hide axis ticks for brain image

plt.tight_layout()
plt.savefig("results/09_qpp_extraction.png", dpi=150)
plt.show()
print("Saved → results/09_qpp_extraction.png")

print("\n" + "=" * 60)
print("QPP EXTRACTION COMPLETE!")
print("=" * 60)
print(f"""
Summary:
  QPP window     : {WINDOW_SEC}s ({window_len} timepoints)
  Iterations     : {N_ITER}
  Occurrences    : {len(peak_times)} times in {T*TR:.0f}s scan
  Frequency      : every ~{T*TR/max(len(peak_times),1):.0f}s on average
  Template saved : {QPP_OUT}
""")