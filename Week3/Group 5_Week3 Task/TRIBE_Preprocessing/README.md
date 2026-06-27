# fMRI Preprocessing

A parcel-level fMRI preprocessing pipeline that turns fMRIPrep-derived BOLD data
into compact, model-ready tensors — **1000 cortical parcels (Schaefer-2018)** +
**32 subcortical parcels (Tian S2)** per time point at 1 Hz — applied identically
to the Narratives and LPP datasets.

---

## 1. Prerequisites

- **Python 3.11** — https://www.python.org
- **Git** — https://git-scm.com

(For downloading the datasets you also need DataLad + git-annex — see the dataset
clone guide. They are **not** required just to set up the environment.)

---

## 2. Set up the virtual environment

From the project root (`TRIBE_Preprocessing/`):

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name tribe-venv --display-name "TRIBE venv"
```

> If PowerShell blocks activation with a script-execution error, run this once and
> then activate again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name tribe-venv --display-name "TRIBE venv"
```

You'll know the environment is active when your prompt starts with `(.venv)`.

---

## 3. Run the notebooks

1. Open the project folder in VS Code (or Jupyter).
2. Open a notebook and select the **TRIBE venv** kernel (top-right in VS Code).
3. Run All.

| Notebook | What it does |
|----------|--------------|
| `notebooks/01_narratives_batch.ipynb` | Preprocess the Narratives dataset |
| `notebooks/02_lpp_batch.ipynb`        | Preprocess the LPP dataset |
| `notebooks/03_plots.ipynb`            | Visualise a preprocessed output |
| `data analysis/*_analysis.ipynb`      | Quantify parcellation detail loss |

The atlases (Schaefer-1000, Tian S2) download automatically on first run.

---

## 4. Notes

- The `.venv/`, `data/`, `outputs/`, and `atlases/` folders are **not** version-controlled
  (see `.gitignore`). They are recreated by setting up the environment and running the
  notebooks; the datasets are cloned separately (see the dataset clone guide).
- Full pipeline details are in `TRIBE_Pipeline_Documentation.docx`.
