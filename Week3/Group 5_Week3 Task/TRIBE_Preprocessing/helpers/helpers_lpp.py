"""
============================================================================
helpers_lpp.py  -  Simple preprocessing functions for the LPP dataset
============================================================================
Le Petit Prince (OpenNeuro ds003643), volume-only (Colin27).

The whole pipeline is just 6 small steps, done for each run:
    1. load the brain file
    2. parcellate the cortex   -> (timepoints, 1000)   [Schaefer-1000 atlas]
    3. parcellate the subcortex -> (timepoints, 32)     [Tian S2 atlas]
    4. clean   (remove drift + z-score)
    5. resample to 1 sample per second
    6. save as .npy

Cortex and subcortex use the SAME function (`parcellate`) - only the atlas
differs. Every comment is written so a non-coder can follow along.
----------------------------------------------------------------------------
"""

import os, re, glob, json, time, subprocess, urllib.request
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
from nilearn import signal, datasets
from nilearn.maskers import NiftiLabelsMasker

# --- The ONLY dataset-specific setting: which file holds the brain data ---
BOLD_PATTERN = "*space-MNIColin27_desc-preproc_bold.nii.gz"


# ===========================================================================
# ATLASES  (downloaded the first time, then reused)
# ===========================================================================

def get_cortical_atlas():
    """Schaefer-1000: labels every cortical voxel with one of 1000 regions.
    Downloaded into the project 'atlases' folder so the project stays self-contained."""
    atlas_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atlases"))
    a = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2, data_dir=atlas_dir)
    return nib.load(a.maps) if isinstance(a.maps, str) else a.maps


def get_subcortical_atlas(atlas_dir):
    """Tian S2: labels every subcortical voxel with one of 32 regions.
    Downloaded once into the atlases folder."""
    os.makedirs(atlas_dir, exist_ok=True)
    path = os.path.join(atlas_dir, "Tian_Subcortex_S2_3T.nii")
    if not os.path.exists(path):
        url = ("https://github.com/yetianmed/subcortex/raw/master/"
               "Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S2_3T.nii")
        print("  downloading Tian subcortical atlas (first time only) ...")
        urllib.request.urlretrieve(url, path)
    return nib.load(path)


# ===========================================================================
# THE 6 CORE STEPS  (identical for every dataset)
# ===========================================================================

def get_tr(bold_path):
    """Step 1 helper: read the TR (seconds between scans) from the file header.
    Some files store it in milliseconds (e.g. 1500) - if so, convert to seconds."""
    tr = float(nib.load(bold_path).header.get_zooms()[3])
    return tr / 1000.0 if tr > 20 else tr


def parcellate(bold_path, atlas_img):
    """Steps 2 & 3: average the brain voxels inside each atlas region.
    Returns (timepoints, number_of_regions). Used for BOTH cortex and subcortex -
    just pass the cortical atlas (1000) or the subcortical atlas (32)."""
    masker = NiftiLabelsMasker(atlas_img, resampling_target="data", verbose=0)
    return masker.fit_transform(bold_path).astype("float32")


def clean(data, tr):
    """Step 4: remove slow drift (detrend) and rescale each region to mean 0,
    std 1 (z-score). NaNs from flat regions become 0."""
    out = signal.clean(data, detrend=True, standardize="zscore_sample", t_r=tr)
    return np.nan_to_num(out).astype("float32")


def resample_1hz(data, tr):
    """Step 5: put the time series on a common 1-sample-per-second timeline."""
    n = data.shape[0]
    old_times = np.arange(n) * tr
    new_times = np.arange(0, old_times[-1] + 1e-9, 1.0)
    return interp1d(old_times, data, axis=0, fill_value="extrapolate")(new_times).astype("float32")


# ===========================================================================
# FIND THE RUNS  (dataset-specific: file names)
# ===========================================================================

def find_runs(func_dir, subject):
    """List a subject's runs. Each run is one brain file. Returns name, path, TR."""
    runs = []
    pattern = os.path.join(func_dir, f"{subject}_{BOLD_PATTERN}")
    for path in sorted(glob.glob(pattern)):
        name = re.search(re.escape(subject) + r"_(.+?)_space-", os.path.basename(path)).group(1)
        runs.append({"name": name, "bold": path, "tr": get_tr(path)})
    return runs


# ===========================================================================
# SAVE
# ===========================================================================

def save_npy(out_dir, filename, array):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, filename), array.astype("float32"))
    print(f"  saved {filename}  shape={array.shape}")


def save_metadata(out_dir, filename, info):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w") as f:
        json.dump(info, f, indent=2)


# ===========================================================================
# DOWNLOAD / CLEANUP  (datalad plumbing - kept apart from the science above)
# ===========================================================================

def _func_dir(dataset_root, subject):
    return os.path.join(dataset_root, "derivatives", subject, "func")


def enable_fallback_remote(dataset_root, remote="fcp-indi"):
    """Turn on a public backup download source (harmless if it doesn't exist)."""
    try:
        subprocess.run(["git", "annex", "enableremote", remote],
                       cwd=dataset_root, capture_output=True, text=True)
    except Exception:
        pass


def download_subject(dataset_root, subject):
    """Download just this subject's brain file, retrying transient network errors."""
    func = _func_dir(dataset_root, subject)
    files = glob.glob(os.path.join(func, BOLD_PATTERN))
    if not files:
        raise FileNotFoundError(f"No brain file for {subject} in {func}")
    print(f"  downloading {len(files)} file(s) for {subject} ...")
    for attempt in range(3):
        r = subprocess.run(["datalad", "get", *files], cwd=dataset_root,
                           capture_output=True, text=True)
        if r.returncode == 0:
            return func
        if "not available" in (r.stderr + r.stdout).lower():
            break
        time.sleep(5)
    raise RuntimeError("datalad get failed:\n" + (r.stderr + r.stdout)[-1500:])


def cleanup_subject(dataset_root, subject):
    """Free disk space after outputs are saved: datalad drop + delete leftovers."""
    func = _func_dir(dataset_root, subject)
    subprocess.run(["datalad", "drop", f"derivatives/{subject}/func"],
                   cwd=dataset_root, check=False)
    removed = 0
    for f in glob.glob(os.path.join(func, BOLD_PATTERN)):
        if os.path.exists(f) and os.path.getsize(f) > 100_000:
            os.remove(f)
            removed += 1
    print(f"  cleaned up {subject}: removed {removed} file(s)")


def outputs_complete(out_dir, n_runs):
    """Safety check before deleting raw data: 2 .npy per run (cortex + subcortex)."""
    if not os.path.isdir(out_dir):
        return False
    return len([f for f in os.listdir(out_dir) if f.endswith(".npy")]) >= 2 * n_runs
