"""
Data loading utilities for public biomedical datasets.

Datasets:
- Epileptic Seizure Recognition (UCI) - local CSV or OpenML
- Breast Cancer Wisconsin (UCI ML Repository)
- Wine Classification (UCI ML Repository)
- Iris (UCI ML Repository) - for testing
- Optical Recognition of Handwritten Digits (UCI ML Repository) - for testing
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPILEPSY_CSV = REPO_ROOT / "data" / "epileptic_seizure.csv"

DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "iris": load_iris,
    "digits": load_digits,
}

DATASET_INFO = {
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": 569,
        "features": 30,
        "classes": 2,
        "class_names": ["malignant", "benign"],
        "source": "UCI ML Repository",
        "url": "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)",
        "description": "Diagnostic features from breast cancer biopsies",
        "paper": "Wolberg, Street, Mangasarian (1995)",
    },
    "wine": {
        "name": "Wine Classification",
        "samples": 178,
        "features": 13,
        "classes": 3,
        "class_names": ["class_0", "class_1", "class_2"],
        "source": "UCI ML Repository",
        "url": "https://archive.ics.uci.edu/ml/datasets/Wine",
        "description": "Chemical composition of wines from three cultivars",
        "paper": "Aeberhard, Coomans, De Vel (1992)",
    },
    "iris": {
        "name": "Iris Flower Dataset",
        "samples": 150,
        "features": 4,
        "classes": 3,
        "class_names": ["setosa", "versicolor", "virginica"],
        "source": "UCI ML Repository",
        "url": "https://archive.ics.uci.edu/ml/datasets/Iris",
        "description": "Measurements of iris flowers from three species",
        "paper": "Fisher (1936)",
    },
    "digits": {
        "name": "Optical Recognition of Handwritten Digits",
        "samples": 1797,
        "features": 64,
        "classes": 10,
        "class_names": [str(i) for i in range(10)],
        "source": "UCI ML Repository",
        "url": "https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits",
        "description": "Handwritten digit recognition (8x8 pixel images)",
        "paper": "Alpaydin, Kaynak (1995)",
    },
}


def load_public_dataset(name="breast_cancer", random_state=42):
    """
    Load a public biomedical dataset and return a pandas DataFrame plus metadata.

    Args:
        name: Dataset name ('breast_cancer', 'wine', 'iris', 'digits')
        random_state: Unused; kept for API compatibility

    Returns:
        Tuple of (DataFrame with features and `target` column, metadata dict)

    Citations:
        Breast Cancer: Wolberg, Street, Mangasarian (1995)
        Wine: Aeberhard, Coomans, De Vel (1992)
        Iris: Fisher (1936)
        Digits: Alpaydin, Kaynak (1995)
    """
    if name not in DATASETS:
        raise ValueError(f"Dataset '{name}' not found. Available: {list(DATASETS)}")

    raw = DATASETS[name]()
    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="target")
    df = pd.concat([X, y], axis=1)

    target_names = [str(t) for t in raw.target_names]
    metadata = {
        "name": name,
        "target_names": target_names,
        # Alias used by the notebooks
        "class_names": target_names,
        "feature_names": list(raw.feature_names),
        "n_samples": len(df),
        "n_features": X.shape[1],
        "n_classes": y.nunique(),
        "class_distribution": y.value_counts().to_dict(),
        "source": "UCI ML Repository",
        "description": raw.DESCR,
    }
    return df, metadata


def load_real_dataset(name="breast_cancer", random_state=42, **kwargs):
    """Alias of `load_public_dataset` used by the demo notebook."""
    return load_public_dataset(name=name, random_state=random_state)


def get_dataset_info(name: str | None = None) -> dict:
    """
    Get information about available datasets.

    Args:
        name: Specific dataset name or None for all datasets

    Returns:
        Dictionary with dataset information including source and citations
    """
    if name is None:
        return DATASET_INFO
    return DATASET_INFO.get(name, {"error": f"Unknown dataset: {name}"})


def _normalize_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the label column to `target`, falling back to a numeric last column."""
    for col in ("target", "class", "y", "label"):
        if col in df.columns:
            return df.rename(columns={col: "target"})

    last_col = df.columns[-1]
    if pd.api.types.is_numeric_dtype(df[last_col]):
        return df.rename(columns={last_col: "target"})
    return df


def _fetch_epileptic_from_openml() -> tuple[pd.DataFrame, dict]:
    """Best-effort fetch of the first OpenML dataset whose name contains 'epileptic'."""
    import openml

    dlist = openml.datasets.list_datasets(output_format="dataframe")
    matches = dlist[dlist["name"].str.contains("epileptic", case=False, na=False)]
    if matches.empty:
        raise RuntimeError("No OpenML dataset name contains 'epileptic'")

    did = int(matches.iloc[0]["did"])
    ds = openml.datasets.get_dataset(did)
    X, y, *_ = ds.get_data(target=ds.default_target_attribute)

    if y is None:
        if not (isinstance(X, pd.DataFrame) and X.shape[1] > 0):
            raise RuntimeError("OpenML dataset had no target column")
        X, y = X.iloc[:, :-1], X.iloc[:, -1]

    df = pd.concat([pd.DataFrame(X), pd.Series(y, name="target")], axis=1)
    meta = {
        "name": ds.name or "epileptic_seizure",
        "n_samples": df.shape[0],
        "n_features": df.shape[1] - 1,
        "source": "openml",
        "openml_id": did,
    }
    return df, meta


def load_epileptic_seizure(local_path: str | None = None, openml_lookup: bool = True):
    """
    Load the Epileptic Seizure Recognition dataset (UCI).

    Prefers a local CSV (`local_path`, or `data/epileptic_seizure.csv` at the
    repository root). Otherwise, if `openml_lookup` is True, attempts a
    best-effort fetch from OpenML.

    The label column is renamed to `target`.

    Returns:
        (DataFrame, metadata dict)

    Raises:
        FileNotFoundError: if neither the local CSV nor OpenML is available.
    """
    csv_path = Path(local_path) if local_path else DEFAULT_EPILEPSY_CSV

    if csv_path.exists():
        df = _normalize_target_column(pd.read_csv(csv_path))
        has_target = "target" in df.columns
        meta = {
            "name": "epileptic_seizure",
            "source": str(csv_path),
            "n_samples": len(df),
            "n_features": df.shape[1] - int(has_target),
            "target_col": "target" if has_target else None,
        }
        return df, meta

    if not openml_lookup:
        raise FileNotFoundError(
            "Could not find local CSV for Epileptic dataset and OpenML lookup is disabled. "
            f"Expected file at: {DEFAULT_EPILEPSY_CSV}"
        )

    try:
        return _fetch_epileptic_from_openml()
    except Exception as e:
        raise FileNotFoundError(
            "Could not find local CSV and OpenML fetch failed. "
            "Please download the UCI Epileptic Seizure Recognition CSV and "
            f"save it to data/epileptic_seizure.csv. Original error: {e}"
        ) from e
