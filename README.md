# Epilepsy ML Pipeline

Open-source implementation accompanying the paper **The Use of Machine Learning in Predicting
Neurological Disorders for Epilepsy** (Vaish, IJFMR 2024). See [papers/](papers/README.md) for the
publication, methods summary, and citation.

The pipeline covers data loading, EEG filtering, feature extraction and engineering, classical
(scikit-learn) and deep (PyTorch) models, training, evaluation with medical metrics, and plotting.

## Requirements

- Python **3.15**
- Dependencies are declared in [pyproject.toml](pyproject.toml). PyTorch publishes Python 3.15
  wheels on `download.pytorch.org` rather than PyPI; [requirements.txt](requirements.txt) adds that
  index for you.

## Setup

```bash
# Windows
py -3.15 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.15 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode with both optional groups:

| Group       | Packages                                   | Used by                      |
|-------------|--------------------------------------------|------------------------------|
| (core)      | numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, PyWavelets, torch | `src/` |
| `notebooks` | notebook, ipykernel, ipywidgets, plotly    | the two notebooks            |
| `extras`    | openml, shap                               | OpenML download, SHAP cell   |

To install only the core library: `pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu`.

### Python 3.15 notes

- **PyYAML** (needed by Jupyter) has no 3.15 wheel yet. If its source build fails, build the
  pure-Python version: set `PYYAML_FORCE_CYTHON=0` and `PYYAML_FORCE_LIBYAML=0` before
  `pip install`. The smoke-test scripts already do this.
- **gradio** cannot be installed yet (its dependency `brotli` has no 3.15 wheel), so the Gradio demo
  cell in the extended notebook prints an install hint and skips.
- **Windows Smart App Control** may block freshly released 3.15 wheels (numpy, torch, ...) with
  "An Application Control policy has blocked this file". This is an OS policy, not a project
  problem; the wheels become loadable once Microsoft's reputation service recognises them.

## Quick start

Run the smoke test, which checks the environment and runs a small end-to-end pipeline
(dataset → features → preprocessing → random forest + MLP → metrics):

```bash
python scripts/smoke_test.py
```

Or let a script create `.venv`, install everything, and run it:

```bash
scripts\run_smoke_test.bat     # Windows
./scripts/run_smoke_test.sh    # macOS / Linux
```

Then open either notebook from the repository root:

- [epilepsy_ml_pipeline_demo.ipynb](epilepsy_ml_pipeline_demo.ipynb): self-contained walkthrough on public datasets
- [epilepsy_ml_pipeline_extended.ipynb](epilepsy_ml_pipeline_extended.ipynb): end-to-end pipeline built on `src/`

## Data

`load_epileptic_seizure()` loads the UCI Epileptic Seizure Recognition dataset from
`data/epileptic_seizure.csv` if present, otherwise tries OpenML (requires the `extras` group).
The `data/` folder is git-ignored. Without the dataset, the notebooks fall back to the bundled
scikit-learn datasets (`breast_cancer`, `wine`, `iris`, `digits`) via `load_public_dataset()`.

## Project layout

```
.
├── src/
│   ├── data/loader.py              # dataset loading (epileptic seizure, public UCI datasets)
│   ├── preprocessing/
│   │   ├── eeg_filters.py          # bandpass/high/low/notch filters, artifact removal, CAR
│   │   └── pipeline.py             # sklearn preprocessing pipelines, EEGPreprocessor
│   ├── features/
│   │   ├── extractor.py            # time/frequency/wavelet/Hjorth signal features
│   │   └── engineer.py             # tabular feature engineering (ratios, statistics, interactions)
│   ├── models/
│   │   ├── classical.py            # sklearn model factory, voting/stacking ensembles, k-means
│   │   └── deep.py                 # PyTorch MLP, CNN, attention, residual models
│   ├── training/train.py           # data splits, sklearn/PyTorch training, CV, grid search
│   ├── evaluation/metrics.py       # classification + medical metrics (sensitivity, specificity, PPV, NPV)
│   ├── visualization/plots.py      # EDA and result plots
│   └── utils/seed.py               # reproducible seeding
├── scripts/                        # smoke test and setup helpers
├── papers/                         # publication PDF, metadata, and citation
├── epilepsy_ml_pipeline_demo.ipynb
└── epilepsy_ml_pipeline_extended.ipynb
```

