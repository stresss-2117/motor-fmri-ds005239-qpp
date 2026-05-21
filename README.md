# motor-fmri-ds005239-qpp

# 🧠 Motor fMRI QPP Analysis Pipeline
### ds005239 — Large-scale motor fMRI dataset

A complete manual fMRI preprocessing and Quasi-Periodic Pattern (QPP) 
extraction pipeline built from scratch in Python.

**Dataset:** [ds005239 on OpenNeuro](https://openneuro.org/datasets/ds005239/versions/1.0.1)  
**Subject:** sub-01, ses-1, task-motor, run-1  
**Scanner:** Philips Achieva 3T | TR = 2.0s | 39 slices | 300 timepoints

---


## 🖥️ Setup Instructions

### 1. Prerequisites
Make sure you have these installed:
- Python 3.11+
- Git
- Git Bash (Windows) or Terminal (Mac/Linux)

### 2. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/motor-fmri-ds005239-qpp.git
cd motor-fmri-ds005239-qpp
```

### 3. Create and Activate Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate — Windows (Git Bash)
source venv/Scripts/activate

# Activate — Mac/Linux
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the Dataset
```bash
python scripts/01_download_data.py
```
This downloads sub-01 ses-1 from OpenNeuro (~58MB).  
Files saved to `data/raw/`

### 6. Run Preprocessing
```bash
python scripts/02_preprocess.py
```
Runs all 6 preprocessing steps.  
Results saved to `data/processed/` and `results/`

### 7. Extract QPP
```bash
python scripts/03_extract_qpp.py
```

### 8. Visualize Results
```bash
python scripts/04_visualize.py
```

---


## 📁 Repository Structure
motor-fmri-ds005239-qpp/
├── data/
│   ├── raw/          ← downloaded from OpenNeuro (not in Git)
│   ├── fmriprep/     ← fMRIPrep output (not in Git)
│   └── processed/    ← cleaned fMRI data (not in Git)
├── results/          ← figures and QPP output
├── scripts/
│   ├── 01_download_data.py    ← download dataset
│   ├── 02_preprocess.py       ← full preprocessing pipeline
│   ├── 03_extract_qpp.py      ← QPP extraction
│   └── 04_visualize.py        ← visualization
├── notebooks/        ← Jupyter exploration notebooks
├── requirements.txt  ← Python dependencies
└── README.md

---


## 📊 Preprocessing Results

| Step | Input | Output |
|------|-------|--------|
| Raw data | (64,64,39,300) BOLD | Signal range 0–201143 |
| After STC | Timing aligned | 39 slices synced to TR/2 |
| After Motion | Motion corrected | 0 high-motion timepoints |
| After Skull Strip | 159,744 voxels | ~50,000 brain voxels |
| After Smoothing | Pixelated | 6mm FWHM smooth |
| After Bandpass | Raw units | 0.01–0.1 Hz only |
| After Confounds | Noisy | Z-scores, clean signal |

---

## 🔬 What are QPPs?

Quasi-Periodic Patterns (QPPs) are recurring spatiotemporal patterns 
in fMRI data, first described by Majeed et al. (2011). They typically:
- Repeat every ~20 seconds
- Involve anticorrelated DMN and task-positive networks
- Explain significant variance in resting-state fMRI

---

## 📦 Dependencies
numpy
scipy
matplotlib
nibabel
nilearn
openneuro-py
pandas
jupyterlab

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📚 References

- Majeed et al. (2011). Spatiotemporal dynamics of low frequency 
  fluctuations in BOLD fMRI of the rat. *Journal of Neuroscience.*
- Esteban et al. (2019). fMRIPrep: a robust preprocessing pipeline 
  for functional MRI. *Nature Methods.*
- OpenNeuro dataset ds005239

---