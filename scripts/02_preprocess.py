"""
Script 2: Post-fMRIPrep signal cleaning for ds005239
Scanner : Philips Achieva 3T
Task    : motor, run-1
TR      : 2.0 seconds
Slices  : 39
"""
import os
import nibabel as nib
import pandas as pd
from nilearn import image

# ── Paths ──────────────────────────────────────────────────────────────────────
# These will point to fMRIPrep output (after you run fMRIPrep)
# For now we define them so the script is ready
BOLD_PATH = (
    "data/fmriprep/sub-01/ses-1/func/"
    "sub-01_ses-1_task-motor_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
)
CONFOUNDS_PATH = (
    "data/fmriprep/sub-01/ses-1/func/"
    "sub-01_ses-1_task-motor_run-1_desc-confounds_timeseries.tsv"
)
OUTPUT_PATH = "data/processed/sub-01_ses-1_task-motor_run-1_clean_bold.nii.gz"

TR = 2.0  # confirmed from bold.json

# ── Check files exist before proceeding ───────────────────────────────────────
if not os.path.exists(BOLD_PATH):
    print("❌ BOLD file not found. Have you run fMRIPrep yet?")
    print(f"   Expected at: {BOLD_PATH}")
    exit()

if not os.path.exists(CONFOUNDS_PATH):
    print("❌ Confounds file not found. Have you run fMRIPrep yet?")
    exit()

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading BOLD image...")
img = nib.load(BOLD_PATH)
print(f"   Shape: {img.shape}")   # should be (X, Y, Z, timepoints)

print("Loading confounds...")
confounds_df = pd.read_csv(CONFOUNDS_PATH, sep='\t')
print(f"   Available confounds: {list(confounds_df.columns[:10])} ...")

# ── Select nuisance regressors ─────────────────────────────────────────────────
confound_cols = [
    'trans_x', 'trans_y', 'trans_z',   # head translation
    'rot_x',   'rot_y',   'rot_z',     # head rotation
    'white_matter',                     # WM signal
    'csf'                               # CSF signal
]

# Only keep columns that actually exist in this dataset
confound_cols = [c for c in confound_cols if c in confounds_df.columns]
print(f"   Using confounds: {confound_cols}")

confounds = confounds_df[confound_cols].fillna(0).values

# ── Clean signal ───────────────────────────────────────────────────────────────
print("\nCleaning signal...")
print("   → Regressing out motion + WM + CSF")
print("   → Bandpass filtering: 0.01–0.1 Hz")
print("   → Standardizing voxel timeseries")

clean_img = image.clean_img(
    img,
    confounds=confounds,
    t_r=TR,
    low_pass=0.1,    # remove fast noise
    high_pass=0.01,  # remove slow drift
    standardize=True
)

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
print(f"\nSaving cleaned image → {OUTPUT_PATH}")
clean_img.to_filename(OUTPUT_PATH)
print("Done! ✅")