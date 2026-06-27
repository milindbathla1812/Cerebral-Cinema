# TRIBE v2 Preprocessing — Dataset Clone Guide (Windows)

This is the exact process we used to set up the environment and clone the
**Narratives** and **LPP** datasets for preprocessing. Follow it top to bottom.

---

## 0. Prerequisites (install once)

1. **Git for Windows** — https://git-scm.com  (check with `git --version`)
2. **Python 3.11** — https://www.python.org  (check with `py -3.11 --version`)
3. **DataLad + git-annex** (these download the datasets):

   **Open PowerShell as Administrator for the git-annex step** (right-click the
   PowerShell icon → **Run as administrator**). git-annex installs into a system
   location, so without admin rights Windows blocks it with a permission error.

   ```powershell
   pip install datalad datalad-installer
   datalad-installer git-annex -m datalad/git-annex:release
   ```
   Close and reopen the terminal, then confirm both work:
   ```powershell
   datalad --version
   git annex version
   ```
   > If `datalad-installer git-annex` is blocked by Windows permissions, install
   > git-annex manually from https://git-annex.branchable.com/install/Windows/ and
   > reopen the terminal.

---

## 1. Project + virtual environment (once)

```powershell
cd C:\TRIBE_Preprocessing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name tribe-venv --display-name "TRIBE venv"
```
(If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.)

---

## 2. Clone NARRATIVES (ds002345 — Hasson Lab full fMRIPrep derivatives)

> We use the Hasson Lab DataLad copy, **not** the OpenNeuro auto-derivatives
> (those are "minimal-mode" and lack the preprocessed BOLD/surfaces we need).

```powershell
cd C:\TRIBE_Preprocessing\data
datalad clone https://datasets.datalad.org/labs/hasson/narratives narratives
cd narratives

# Install the fmriprep sub-dataset's file list (pointers only, no big files yet)
datalad get -n derivatives/fmriprep

# Enable the public fallback source (for files missing from the main server)
cd derivatives\fmriprep
git annex enableremote fcp-indi
```

Cloning only fetches tiny pointer files. The actual ~1.5 GB per subject is
downloaded **by the notebook**, one subject at a time, and deleted after the
outputs are saved.

Subject folders: `derivatives\fmriprep\sub-001 … sub-345`.

---

## 3. Clone LPP (Le Petit Prince, ds003643)

> LPP has **no** separate fMRIPrep derivatives repo. Its preprocessed data
> (volume-only, MNIColin27) lives inside the main dataset under `derivatives\`.

```powershell
cd C:\TRIBE_Preprocessing\data
datalad clone https://github.com/OpenNeuroDatasets/ds003643.git lpp
cd lpp
```

That's it for LPP — the derivatives are part of the main dataset, so no
`datalad get -n` of a sub-dataset is needed. The notebook downloads each
subject's BOLD, processes it, and cleans up. The notebook also calls
`enable_fallback_remote` automatically.

Subject folders: `derivatives\sub-EN057 … sub-EN115` (English), plus `sub-CN…` and `sub-FR…`.

---

## 4. Run the preprocessing

Open the matching notebook in VS Code, pick the **TRIBE venv** kernel, edit the
`SUBJECTS` list, and Run All:

| Dataset | Notebook | Helper |
|---------|----------|--------|
| Narratives | `notebooks\01_narratives_batch.ipynb` | `helpers\helpers_narratives.py` |
| LPP        | `notebooks\02_lpp_batch.ipynb`        | `helpers\helpers_lpp.py` |

Each batch run **downloads → processes → saves `.npy` → verifies → deletes the
raw data**, so disk usage stays low. Outputs land in `outputs\<dataset>\<subject>\`.

---

## 5. Re-fetching subject data after it has been dropped

After preprocessing, each subject's raw download is removed to save space. To get a
subject's data back (to re-run it, or for the loss analysis):

**Normal case — the tiny pointer files are still there**, so just download again
(or simply re-run the batch notebook for that subject — it downloads automatically):
```powershell
cd C:\TRIBE_Preprocessing\data\<dataset>
datalad get <path-to-subject-or-file>
```

**If the pointer files were also deleted** (e.g. you cleared the data folder, and a
batch then fails with `No brain file for sub-XXX` / empty `func` folders), restore
the skeleton first with `git checkout .`, then download:
```powershell
# Narratives
cd C:\TRIBE_Preprocessing\data\narratives\derivatives\fmriprep
git checkout .

# LPP
cd C:\TRIBE_Preprocessing\data\lpp
git checkout .
```
`git checkout .` recreates only the small pointer files (fast, no data download); the
real brain data then re-downloads per subject when you run a notebook. **Never delete
the `.git` folders** — that is exactly what makes this restoration possible.

---

## Notes / gotchas we hit

- **Disk space:** on Windows, `datalad drop` alone does **not** free space (it
  leaves file copies). The notebooks delete the working files directly — don't
  rely on `datalad drop` by itself.
- **`[not available]` on download:** a file missing from the main server. The
  `fcp-indi` fallback remote (enabled above) usually fixes it automatically.
- **Transient network errors:** the download step auto-retries 3 times.
- **VS Code Source Control clutter:** the datasets are git repos, so VS Code may
  show hundreds of "changes." Hide them with Setting `git.autoRepositoryDetection`
  = `false`. Do **not** delete the `.git` folders — that breaks datalad.
- **"No brain file" error / empty `func` folders:** if a batch fails with
  `No brain file for sub-XXX` and the dataset's `func` folders are empty, the tiny
  pointer files (the dataset skeleton) were deleted — usually when clearing data.
  Git still tracks them, so restore them with **`git checkout .`** run inside the
  dataset folder. This only recreates the small pointers (fast, no data download);
  the real data then re-downloads per subject when you run the notebook.
    - Narratives: `cd C:\TRIBE_Preprocessing\data\narratives\derivatives\fmriprep`  then  `git checkout .`
    - LPP:        `cd C:\TRIBE_Preprocessing\data\lpp`   then  `git checkout .`
