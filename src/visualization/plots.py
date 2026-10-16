"""
Visualization utilities for ML results and data analysis.
Provides plots for EDA, model evaluation, and publication-ready figures.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_correlation_matrix(df, figsize=(12, 10), cmap="coolwarm"):
    """
    Plot correlation matrix heatmap.

    Args:
        df: Input dataframe with numeric features
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    corr_matrix = df.corr()
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap=cmap,
        center=0,
        square=True,
        ax=ax,
        cbar_kws={'label': 'Correlation'},
    )

    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_pca_embedding(X, y, n_components=2, labels=None, figsize=(8, 6)):
    """
    Plot PCA projection of features.

    Args:
        X: Feature matrix
        y: Labels or colors
        n_components: Number of PCA components
        labels: Class labels for legend
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    pca = PCA(n_components=n_components, random_state=42)
    projection = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(np.unique(y))
    palette = sns.color_palette("tab10", n_colors=n_classes)

    sns.scatterplot(
        x=projection[:, 0],
        y=projection[:, 1],
        hue=y,
        palette=palette,
        legend="full",
        ax=ax,
        s=100,
        alpha=0.7,
    )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=12)
    ax.set_title("PCA Projection of Features", fontsize=14, fontweight='bold')

    if labels:
        ax.legend(labels, title="Class", loc="best")

    plt.tight_layout()

    return fig


def plot_tsne_embedding(X, y, n_components=2, figsize=(8, 6), perplexity=30):
    """
    Plot t-SNE embedding of features.

    Args:
        X: Feature matrix
        y: Labels
        n_components: Number of dimensions
        figsize: Figure size
        perplexity: t-SNE perplexity parameter

    Returns:
        Matplotlib figure
    """
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    projection = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    n_classes = len(np.unique(y))
    palette = sns.color_palette("tab10", n_colors=n_classes)

    sns.scatterplot(
        x=projection[:, 0],
        y=projection[:, 1],
        hue=y,
        palette=palette,
        legend="full",
        ax=ax,
        s=100,
        alpha=0.7,
    )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE Embedding of Features", fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def plot_feature_importance(feature_names, importances, top_n=20, figsize=(10, 8)):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Importance values
        top_n: Number of top features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get top features
    importances = np.asarray(importances)
    top_indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight='bold')

    plt.tight_layout()

    return fig


def plot_training_history(history, figsize=(12, 4)):
    """
    Plot training history for deep learning models.

    Args:
        history: Dictionary with training metrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2 if "val_loss" in history else 1, figsize=figsize)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Loss plot
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss", marker='o')
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", marker='s')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy plot (if available)
    if len(axes) > 1:
        ax = axes[1]
        if "train_acc" in history:
            ax.plot(history["train_acc"], label="Train Acc", marker='o')
        if "val_acc" in history:
            ax.plot(history["val_acc"], label="Val Acc", marker='s')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Training Accuracy", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


def plot_class_distribution(y, labels=None, figsize=(8, 6)):
    """
    Plot class distribution.

    Args:
        y: Labels
        labels: Class names
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique, counts = np.unique(y, return_counts=True)

    if labels:
        x_labels = [labels[i] if i < len(labels) else str(i) for i in unique]
    else:
        x_labels = [str(i) for i in unique]

    colors = sns.color_palette("husl", len(unique))
    ax.bar(x_labels, counts, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution", fontsize=14, fontweight='bold')

    # Add value labels on bars
    for i, (label, count) in enumerate(zip(x_labels, counts)):
        ax.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    return fig


def plot_feature_distributions(df, n_features=6, figsize=(12, 8)):
    """
    Plot distributions of top features.

    Args:
        df: Input dataframe
        n_features: Number of features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_features = min(n_features, len(numeric_cols))

    fig, axes = plt.subplots(max(1, math.ceil(n_features / 3)), 3, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols[:n_features]):
        axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Distribution of {col}", fontweight='bold')
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")

    # Hide extra subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    return fig


def plot_eeg_signal(eeg_data, channel=0, start=0, end=None, figsize=(12, 4), fs=256):
    """
    Plot raw EEG signal.

    Args:
        eeg_data: EEG signal array (1D or 2D)
        channel: Channel index (for 2D data)
        start: Start time index
        end: End time index
        figsize: Figure size
        fs: Sampling frequency

    Returns:
        Matplotlib figure
    """
    if eeg_data.ndim == 2:
        signal = eeg_data[channel, :]
    else:
        signal = eeg_data

    if end is None:
        end = len(signal)

    signal = signal[start:end]
    time = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, signal, linewidth=1, color='steelblue')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(f"EEG Signal - Channel {channel}", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
