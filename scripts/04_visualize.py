"""
Script 4: QPP Visualization
Produces publication-quality figures of the QPP results
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# gridspec → advanced figure layout tool. Lets us control
# exactly where each subplot goes, like a grid on a page.

from matplotlib.colors import TwoSlopeNorm
# TwoSlopeNorm → makes colormaps with a fixed centre point (zero).
# Red = positive BOLD, Blue = negative BOLD, White = zero.

# ── Paths ──────────────────────────────────────────────────────────────────────
QPP_PATH   = "results/qpp_template.nii.gz"
BOLD_PATH  = "data/processed/sub-01_ses-1_task-motor_run-1_clean_bold.nii.gz"
PEAKS_PATH = "results/qpp_peak_times.npy"
CORR_PATH  = "results/qpp_correlation_timecourse.npy"
MASK_PATH  = "data/processed/brain_mask.npy"

TR = 2.0

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
qpp_img    = nib.load(QPP_PATH)
qpp        = qpp_img.get_fdata()
# qpp shape: (64, 64, 39, 10) — 10 frames of 20s QPP template

bold_img   = nib.load(BOLD_PATH)
bold       = bold_img.get_fdata()
# bold shape: (64, 64, 39, 300) — full preprocessed scan

peak_times = np.load(PEAKS_PATH)
# peak_times: array of timepoints where QPP was detected
# e.g. [10, 40, 76, 113, ...]

corr       = np.load(CORR_PATH)
# corr: full correlation timecourse (291 values)

mask       = np.load(MASK_PATH)
# mask: (64, 64, 39) of 1s and 0s — brain voxels

print(f"QPP shape       : {qpp.shape}")
print(f"BOLD shape      : {bold.shape}")
print(f"QPP occurrences : {len(peak_times)}")
print(f"Correlation pts : {len(corr)}")

# ── Figure 1: QPP Brain Maps Across All 10 Frames ─────────────────────────────
# This shows how the spatial pattern EVOLVES over the 20-second QPP cycle
print("\nFigure 1: QPP brain maps across all frames...")

n_frames  = qpp.shape[3]
# n_frames = 10 (one per 2s TR within the 20s window)

mid_slice = qpp.shape[2] // 2
# Middle axial slice: 39 // 2 = 19

# Colour scale: find the max absolute value across all frames
vmax = np.abs(qpp[:, :, mid_slice, :]).max()
vmax = max(vmax, 0.1)
# max() ensures vmax is at least 0.1 even if signal is very small

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
# TwoSlopeNorm centres the colourmap exactly at zero.
# So white = 0, deep red = +vmax, deep blue = -vmax

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
# 2 rows × 5 columns = 10 subplots, one per QPP frame

fig.suptitle(
    f"QPP Template — All {n_frames} Frames (Middle Slice z={mid_slice})\n"
    f"Red = Active, Blue = Suppressed, White = No change",
    fontsize=13
)

for frame in range(n_frames):
    row = frame // 5
    # Integer division: frames 0-4 → row 0, frames 5-9 → row 1
    col = frame % 5
    # Modulo: 0,1,2,3,4,0,1,2,3,4

    ax = axes[row, col]

    # Get this frame's brain slice, masked to brain only
    brain_slice = qpp[:, :, mid_slice, frame].copy()
    brain_slice[mask[:, :, mid_slice] == 0] = np.nan
    # Set non-brain voxels to NaN (Not a Number).
    # matplotlib skips NaN values → shows as white/transparent.

    im = ax.imshow(
        brain_slice.T,
        # .T transposes: flips axes so brain is upright
        cmap='RdBu_r',
        # Red-Blue reversed: Red=positive, Blue=negative
        norm=norm,
        origin='lower'
    )
    ax.set_title(f"t = {frame * TR:.0f}s", fontsize=10)
    ax.axis('off')
    # Turn off axis numbers — not needed for brain images

# Add one shared colourbar on the right
fig.colorbar(im, ax=axes, shrink=0.6, label='BOLD Z-score')
# ax=axes (all subplots) → single colourbar for the whole figure
# shrink=0.6 → make colourbar 60% of figure height

plt.savefig("results/10_qpp_all_frames.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/10_qpp_all_frames.png")


# ── Figure 2: QPP Timecourse + Occurrence Map ──────────────────────────────────
# Shows WHEN the QPP occurs across the full 10-minute scan
print("\nFigure 2: QPP occurrence timeline...")

fig = plt.figure(figsize=(14, 8))

gs = gridspec.GridSpec(3, 1, hspace=0.45)
# GridSpec: 3 rows, 1 column, with 0.45 spacing between rows.
# hspace = horizontal space (vertical gap between plots — confusingly named!)

ax1 = fig.add_subplot(gs[0])
# First subplot: full BOLD timeseries
ax2 = fig.add_subplot(gs[1])
# Second subplot: correlation timecourse
ax3 = fig.add_subplot(gs[2])
# Third subplot: QPP occurrence markers

fig.suptitle("QPP Occurrences Across the Full 10-Minute Scan", fontsize=13)

# Plot 1: Mean BOLD signal across all brain voxels
brain_voxels = np.where(mask > 0)
mean_bold = bold[brain_voxels[0], brain_voxels[1], brain_voxels[2], :].mean(axis=0)
# Extract all brain voxel timeseries → average across voxels (axis=0)
# Result: one number per timepoint → (300,)

time_axis = np.arange(bold.shape[3]) * TR
# [0, 2, 4, ..., 598] seconds

ax1.plot(time_axis, mean_bold, color='steelblue', linewidth=0.8, alpha=0.8)
ax1.set_ylabel("Mean BOLD\n(z-score)", fontsize=9)
ax1.set_title("Mean Brain Signal", fontsize=10)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

# Shade QPP occurrence windows
for pt in peak_times:
    ax1.axvspan(
        pt * TR,
        (pt + 10) * TR,
        # From peak start to peak start + window_len
        alpha=0.15, color='red'
        # Semi-transparent red shading
    )

# Plot 2: Correlation timecourse
corr_time = np.arange(len(corr)) * TR
ax2.plot(corr_time, corr, color='darkorange', linewidth=0.8)
ax2.axhline(y=0.2, color='red', linestyle='--', linewidth=1,
            label='Threshold (0.2)')
ax2.scatter(
    peak_times * TR,
    corr[peak_times],
    color='red', s=50, zorder=5,
    label=f'QPP ({len(peak_times)} occurrences)'
)
ax2.set_ylabel("Correlation (r)", fontsize=9)
ax2.set_title("Template Correlation Timecourse", fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.set_ylim(-0.8, 1.0)

# Plot 3: Event timeline
ax3.scatter(
    peak_times * TR,
    np.ones(len(peak_times)),
    # y=1 for all points (just a horizontal line of dots)
    color='red', s=100, marker='|',
    # marker='|' = vertical tick marks
    linewidths=2
)
ax3.set_xlim(0, bold.shape[3] * TR)
ax3.set_ylim(0.5, 1.5)
ax3.set_yticks([])
# Remove y-axis ticks — not meaningful here
ax3.set_xlabel("Time (seconds)", fontsize=10)
ax3.set_title(f"QPP Event Timeline — {len(peak_times)} occurrences "
              f"(avg every {bold.shape[3]*TR/len(peak_times):.0f}s)", fontsize=10)
ax3.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

# Add time labels for each QPP
for pt in peak_times:
    ax3.text(
        pt * TR, 1.15,
        f"{pt*TR:.0f}s",
        fontsize=7, ha='center', color='darkred'
        # ha='center' → text centred on the tick
    )

plt.savefig("results/11_qpp_timeline.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/11_qpp_timeline.png")


# ── Figure 3: QPP Spatial Pattern — 3 Slice Views ─────────────────────────────
# Shows the brain map at the PEAK of the QPP from 3 different angles
print("\nFigure 3: QPP spatial pattern — 3 slice views...")

# Find the frame with the strongest activation
mean_per_frame = [
    np.abs(qpp[:, :, :, f][mask > 0]).mean()
    for f in range(n_frames)
]
# For each frame: get brain voxels, take abs value, compute mean
# List of 10 values → one per frame

peak_frame = np.argmax(mean_per_frame)
# Frame index with the highest mean absolute activation

print(f"Peak activation frame: {peak_frame} (t={peak_frame*TR:.0f}s into QPP)")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"QPP Spatial Pattern at Peak Frame (t={peak_frame*TR:.0f}s)\n"
    f"Red=Active, Blue=Suppressed",
    fontsize=13
)

vmax2 = np.abs(qpp[:, :, :, peak_frame][mask > 0]).max()
vmax2 = max(vmax2, 0.1)
norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)

# View 1: Axial (top-down view, z-slice)
axial_slice = qpp.shape[2] // 2
brain_axial = qpp[:, :, axial_slice, peak_frame].copy()
brain_axial[mask[:, :, axial_slice] == 0] = np.nan
im1 = axes[0].imshow(brain_axial.T, cmap='RdBu_r', norm=norm2, origin='lower')
axes[0].set_title(f"Axial (top-down)\nslice z={axial_slice}", fontsize=11)
axes[0].axis('off')
# Add orientation labels
axes[0].text(0.5, -0.05, "A → P", transform=axes[0].transAxes,
             ha='center', fontsize=8, color='gray')
# transAxes: coordinates in axis fraction (0=left, 1=right)

# View 2: Coronal (front-back view, y-slice)
coronal_slice = qpp.shape[1] // 2
brain_coronal = qpp[:, coronal_slice, :, peak_frame].copy()
brain_coronal[mask[:, coronal_slice, :] == 0] = np.nan
axes[1].imshow(brain_coronal.T, cmap='RdBu_r', norm=norm2, origin='lower')
axes[1].set_title(f"Coronal (front-back)\nslice y={coronal_slice}", fontsize=11)
axes[1].axis('off')
axes[1].text(0.5, -0.05, "L → R", transform=axes[1].transAxes,
             ha='center', fontsize=8, color='gray')

# View 3: Sagittal (side view, x-slice)
sagittal_slice = qpp.shape[0] // 2
brain_sagittal = qpp[sagittal_slice, :, :, peak_frame].copy()
brain_sagittal[mask[sagittal_slice, :, :] == 0] = np.nan
axes[2].imshow(brain_sagittal.T, cmap='RdBu_r', norm=norm2, origin='lower')
axes[2].set_title(f"Sagittal (side)\nslice x={sagittal_slice}", fontsize=11)
axes[2].axis('off')
axes[2].text(0.5, -0.05, "A → P", transform=axes[2].transAxes,
             ha='center', fontsize=8, color='gray')

# Shared colourbar
fig.colorbar(im1, ax=axes.tolist(), shrink=0.7, label='BOLD Z-score')
# axes.tolist() converts numpy array of axes to a Python list

plt.savefig("results/12_qpp_spatial_pattern.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/12_qpp_spatial_pattern.png")


# ── Figure 4: Individual QPP Occurrences ──────────────────────────────────────
# Shows the actual brain data at each of the 13 QPP moments
print("\nFigure 4: Individual QPP occurrences...")

n_peaks  = len(peak_times)
n_cols   = 5
n_rows   = int(np.ceil(n_peaks / n_cols))
# np.ceil rounds UP: 13/5 = 2.6 → 3 rows

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 3, n_rows * 3))
fig.suptitle(
    f"All {n_peaks} QPP Occurrences — Brain at Peak Moment\n"
    f"Middle slice z={mid_slice}",
    fontsize=13
)

# Flatten axes array for easy indexing
axes_flat = axes.flatten()
# Turn 2D array of axes into 1D list → easier to loop through

for i, pt in enumerate(peak_times):
    # i = 0,1,2,...,12
    # pt = actual timepoint index

    # Get the middle frame of this QPP window
    mid_t = pt + (10 // 2)
    # pt = window start, add half window = peak of this occurrence

    if mid_t >= bold.shape[3]:
        mid_t = bold.shape[3] - 1
    # Safety check: don't go past end of data

    brain_slice = bold[:, :, mid_slice, mid_t].copy()
    brain_slice[mask[:, :, mid_slice] == 0] = np.nan
    # Get actual BOLD data at this moment, mask non-brain

    vmax3 = np.nanpercentile(np.abs(brain_slice), 95)
    # 95th percentile of absolute values → robust colour scale
    # np.nanpercentile ignores NaN values
    vmax3 = max(vmax3, 0.1)
    norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)

    axes_flat[i].imshow(brain_slice.T, cmap='RdBu_r',
                        norm=norm3, origin='lower')
    axes_flat[i].set_title(f"QPP #{i+1}\nt={pt*TR:.0f}s", fontsize=9)
    axes_flat[i].axis('off')

# Hide any unused subplots
for j in range(n_peaks, len(axes_flat)):
    axes_flat[j].axis('off')
    # If 13 peaks but 15 subplots (3×5), hide last 2

plt.tight_layout()
plt.savefig("results/13_qpp_occurrences.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/13_qpp_occurrences.png")


# ── Figure 5: Before vs After Preprocessing Summary ───────────────────────────
# Side-by-side comparison to show the professor what preprocessing did
print("\nFigure 5: Preprocessing summary comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Effect of Preprocessing — Centre Voxel [32,32,19]", fontsize=13)

# Load raw data for comparison
raw_img  = nib.load(
    "data/raw/sub-01/ses-1/func/sub-01_ses-1_task-motor_run-1_bold.nii.gz"
)
raw_data = raw_img.get_fdata()
raw_ts   = raw_data[32, 32, 19, :]
# Raw timeseries of centre voxel

clean_ts = bold[32, 32, 19, :]
# Clean timeseries of same voxel after preprocessing

time_axis = np.arange(300) * TR

axes[0].plot(time_axis, raw_ts, color='red', linewidth=0.7)
axes[0].set_title("RAW Signal (before preprocessing)", fontsize=11)
axes[0].set_ylabel("Scanner units (arbitrary)")
axes[0].set_xlabel("Time (seconds)")
axes[0].text(0.02, 0.95,
             f"Range: {raw_ts.min():.0f} – {raw_ts.max():.0f}",
             transform=axes[0].transAxes,
             fontsize=9, color='darkred',
             verticalalignment='top')
# transAxes: 0.02 = 2% from left, 0.95 = 95% from bottom

axes[1].plot(time_axis, clean_ts, color='green', linewidth=0.7)
axes[1].set_title("CLEAN Signal (after all 6 preprocessing steps)", fontsize=11)
axes[1].set_ylabel("Z-score")
axes[1].set_xlabel("Time (seconds)")
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1].text(0.02, 0.95,
             f"Range: {clean_ts.min():.2f} – {clean_ts.max():.2f}",
             transform=axes[1].transAxes,
             fontsize=9, color='darkgreen',
             verticalalignment='top')

plt.tight_layout()
plt.savefig("results/14_preprocessing_comparison.png", dpi=150,
            bbox_inches='tight')
plt.show()
print("Saved → results/14_preprocessing_comparison.png")


# ── Final Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL VISUALIZATIONS COMPLETE!")
print("=" * 60)
print("""
Files saved to results/:
  10_qpp_all_frames.png       → QPP brain map across all 10 frames
  11_qpp_timeline.png         → QPP occurrences across full scan
  12_qpp_spatial_pattern.png  → 3-view brain map at QPP peak
  13_qpp_occurrences.png      → all 13 individual QPP moments
  14_preprocessing_comparison → raw vs clean signal comparison
""")