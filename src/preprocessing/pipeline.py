"""
Preprocessing pipelines for biomedical and EEG data.
Includes filtering, normalization, feature selection, and dimensionality reduction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

from src.preprocessing.eeg_filters import apply_eeg_filter


def _base_steps(scale_method):
    scaler = RobustScaler() if scale_method == "robust" else StandardScaler()
    return [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]


def build_preprocessing_pipeline(k_best=12, scale_method="standard", add_poly=True):
    """
    Create a reusable preprocessing pipeline for numeric tabular data.

    Args:
        k_best: Number of features to select
        scale_method: 'standard' or 'robust' scaling
        add_poly: Whether to add degree-2 polynomial features

    Returns:
        sklearn Pipeline object
    """
    steps = _base_steps(scale_method)
    if add_poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("selector", SelectKBest(score_func=f_classif, k=k_best)))
    return Pipeline(steps)


def build_pca_pipeline(n_components=20, scale_method="standard"):
    """
    Create a preprocessing pipeline with PCA for dimensionality reduction.

    Args:
        n_components: Number of PCA components
        scale_method: 'standard' or 'robust' scaling

    Returns:
        sklearn Pipeline object
    """
    steps = _base_steps(scale_method)
    steps.append(("pca", PCA(n_components=n_components, random_state=42)))
    return Pipeline(steps)


def build_eeg_preprocessing_pipeline(
    fs=256,
    apply_filter=True,
    filter_type="bandpass",
    lowcut=1.0,
    highcut=50.0,
    order=4,
):
    """
    Create an EEG-specific preprocessing function that filters each row as a signal.

    Args:
        fs: Sampling frequency (Hz)
        apply_filter: Whether to apply frequency filtering
        filter_type: 'bandpass', 'highpass', 'lowpass', or 'notch'
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Function `preprocess(X, y=None)` mapping a DataFrame to a filtered DataFrame
    """
    def preprocess(X, y=None):
        if not apply_filter:
            return X.copy()
        return X.apply(
            lambda row: apply_eeg_filter(
                row.values, fs=fs, filter_type=filter_type,
                lowcut=lowcut, highcut=highcut, order=order,
            ),
            axis=1,
            result_type="expand",
        )

    return preprocess


def adaptive_k_best(X, y=None, percentile=0.8, min_k=5):
    """
    Adaptively select number of features based on data shape.

    Args:
        X: Feature matrix
        y: Unused; kept for API compatibility
        percentile: Fraction of features to keep
        min_k: Minimum number of features

    Returns:
        Number of features to select
    """
    n_features = X.shape[1]
    return min(max(min_k, int(n_features * percentile)), n_features)


class EEGPreprocessor:
    """Single-signal EEG preprocessing: bandpass filter, artifact removal, normalization."""

    def __init__(self, fs=256, lowcut=1.0, highcut=50.0, order=4):
        """
        Initialize EEG preprocessor.

        Args:
            fs: Sampling frequency
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def apply_bandpass_filter(self, signal_data):
        """Apply bandpass filter to remove artifacts and noise."""
        return apply_eeg_filter(
            signal_data, fs=self.fs, filter_type="bandpass",
            lowcut=self.lowcut, highcut=self.highcut, order=self.order,
        )

    def remove_artifacts(self, signal_data, threshold=3.0):
        """
        Replace samples beyond `threshold` standard deviations with the nearest valid sample.

        Args:
            signal_data: Input signal
            threshold: Standard deviation threshold for artifact detection

        Returns:
            Signal with artifacts attenuated
        """
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        artifact_mask = np.abs(signal_data - mean) > (threshold * std)
        signal_clean = signal_data.copy()

        valid_indices = np.flatnonzero(~artifact_mask)
        if artifact_mask.any() and len(valid_indices) > 1:
            for idx in np.flatnonzero(artifact_mask):
                closest = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                signal_clean[idx] = signal_data[closest]

        return signal_clean

    def normalize(self, signal_data):
        """Normalize signal to zero mean and unit variance."""
        return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)

    def preprocess(self, signal_data):
        """Apply bandpass filtering, artifact removal, and normalization."""
        signal_data = self.apply_bandpass_filter(signal_data)
        signal_data = self.remove_artifacts(signal_data)
        return self.normalize(signal_data)
