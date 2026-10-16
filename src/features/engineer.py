"""
Feature engineering utilities for transforming raw features.
Applies domain-specific feature transformations and enrichment.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.features.extractor import SignalFeatureExtractor


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple feature-engineered ratio features.

    Args:
        df: Input dataframe with numeric features

    Returns:
        Dataframe with additional ratio features
    """
    engineered = df.copy()

    if engineered.shape[1] >= 2:
        # Ratio of mean to standard deviation
        engineered["mean_div_std"] = (
            engineered.mean(axis=1) / (engineered.std(axis=1) + 1e-8)
        )

        # Ratio of max to min
        engineered["max_div_min"] = (
            engineered.max(axis=1) / (engineered.min(axis=1).replace(0, 1e-8))
        )

        # Coefficient of variation
        engineered["cv"] = (
            engineered.std(axis=1) / (engineered.mean(axis=1).abs() + 1e-8)
        )

        # Range normalized by median
        engineered["range_div_median"] = (
            (engineered.max(axis=1) - engineered.min(axis=1)) /
            (engineered.median(axis=1).abs() + 1e-8)
        )

    # Signal characteristics
    engineered["sum"] = engineered.sum(axis=1)
    engineered["product"] = engineered.prod(axis=1)

    return engineered


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced statistical features.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with statistical features
    """
    engineered = df.copy()

    # Skewness and kurtosis
    engineered["skewness"] = df.apply(lambda row: skew(row.values), axis=1)
    engineered["kurtosis"] = df.apply(lambda row: kurtosis(row.values), axis=1)

    # Percentile-based features
    engineered["q25"] = df.quantile(0.25, axis=1)
    engineered["q50"] = df.quantile(0.50, axis=1)
    engineered["q75"] = df.quantile(0.75, axis=1)
    engineered["iqr"] = engineered["q75"] - engineered["q25"]

    # Entropy-based feature
    def shannon_entropy(row):
        # Normalize to probability distribution
        normalized = np.abs(row) / (np.sum(np.abs(row)) + 1e-10)
        return -np.sum(normalized * np.log(normalized + 1e-10))

    engineered["entropy"] = df.apply(shannon_entropy, axis=1)

    return engineered


def add_interaction_features(df: pd.DataFrame, n_interactions=5) -> pd.DataFrame:
    """
    Add polynomial interaction features between top features.

    Args:
        df: Input dataframe
        n_interactions: Number of interaction terms

    Returns:
        Dataframe with interaction features
    """
    engineered = df.copy()

    # Get top features by variance
    variances = df.var()
    top_features = variances.nlargest(min(n_interactions, len(variances))).index.tolist()

    # Create interaction terms
    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):
            feat1 = top_features[i]
            feat2 = top_features[j]
            engineered[f"{feat1}_x_{feat2}"] = df[feat1] * df[feat2]

    return engineered


def normalize_features(df: pd.DataFrame, method="standard") -> pd.DataFrame:
    """
    Normalize features.

    Args:
        df: Input dataframe
        method: 'standard' (z-score) or 'minmax'

    Returns:
        Normalized dataframe
    """
    normalized = df.copy()

    if method == "standard":
        normalized = (df - df.mean()) / (df.std() + 1e-8)
    elif method == "minmax":
        normalized = (df - df.min()) / (df.max() - df.min() + 1e-8)

    # Handle any NaNs or infs
    normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)

    return normalized


def enrich_features(df: pd.DataFrame, include_statistical=True,
                   include_interactions=False) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.

    Args:
        df: Input dataframe
        include_statistical: Whether to add statistical features
        include_interactions: Whether to add interaction features

    Returns:
        Feature-engineered dataframe
    """
    df_enriched = add_ratio_features(df)

    if include_statistical:
        df_enriched = add_statistical_features(df_enriched)

    if include_interactions:
        df_enriched = add_interaction_features(df_enriched, n_interactions=3)

    return df_enriched


def extract_eeg_features_tabular(eeg_data, fs=256):
    """
    Convert 3D EEG data (samples, channels, timepoints) to 2D tabular format.

    Args:
        eeg_data: EEG array (n_samples, n_channels, signal_length)
        fs: Sampling frequency

    Returns:
        Tuple of (DataFrame with one row per sample, list of feature column names)
    """
    _, n_channels, _ = eeg_data.shape
    base_names = SignalFeatureExtractor.get_feature_names(include_wavelets=False)
    feature_names = [f"ch{ch}_{name}" for ch in range(n_channels) for name in base_names]

    rows = [
        np.concatenate([
            SignalFeatureExtractor.extract_all_features(sample[ch], fs=fs, include_wavelets=False)
            for ch in range(n_channels)
        ])
        for sample in eeg_data
    ]

    return pd.DataFrame(rows, columns=feature_names), feature_names
