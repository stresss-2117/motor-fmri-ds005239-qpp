"""
Script 1: Download motor fMRI dataset ds005239 from OpenNeuro
Structure: sub-01/ses-1/func/ with task-motor runs 1-10
We download just run-1 to start (saves time + storage)
"""
import openneuro

print("Starting download of ds005239...")

openneuro.download(
    dataset="ds005239",
    tag="1.0.1",
    target_dir="data/raw",
    include=[
        # Metadata files
        "dataset_description.json",
        "participants.tsv",
        "README",

        # Structural scan (needed for preprocessing)
        "sub-01/ses-1/anat/sub-01_ses-1_T1w.nii.gz",
        "sub-01/ses-1/anat/sub-01_ses-1_T1w.json",

        # Functional scan — just run-1 to start
        "sub-01/ses-1/func/sub-01_ses-1_task-motor_run-1_bold.nii.gz",
        "sub-01/ses-1/func/sub-01_ses-1_task-motor_run-1_bold.json",
        "sub-01/ses-1/func/sub-01_ses-1_task-motor_run-1_events.tsv",

        # Fieldmap (for distortion correction in fMRIPrep)
        "sub-01/ses-1/fmap/sub-01_ses-1_dir-AP_run-1_epi.nii.gz",
        "sub-01/ses-1/fmap/sub-01_ses-1_dir-AP_run-1_epi.json",
    ]
)

print("Download complete! Check data/raw/")