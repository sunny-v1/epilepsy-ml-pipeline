"""
Comprehensive evaluation metrics for medical ML applications.
Includes classification metrics, ROC curves, and medical-specific measures.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)


def classification_summary(y_true, y_pred, y_proba=None, target_names=None):
    """
    Generate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        target_names: Names of target classes

    Returns:
        Dictionary with various metrics
    """
    summary = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=target_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # ROC-AUC if probabilities available and binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            summary["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except ValueError:
            pass

    return summary


def binary_classification_summary(y_true, y_pred, y_proba=None, target_names=None):
    """
    Generate metrics specifically for binary classification.
    Includes sensitivity, specificity, and PPV/NPV for medical applications.

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        target_names: Names of target classes

    Returns:
        Dictionary with medical-relevant metrics
    """
    summary = classification_summary(y_true, y_pred, y_proba, target_names)

    tn, fp, fn, tp = summary["confusion_matrix"].ravel()

    summary.update({
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,  # True positive rate
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,  # True negative rate
        "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive predictive value
        "npv": tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    })

    return summary


def clustering_summary(X, labels):
    """
    Generate clustering evaluation metrics.

    Args:
        X: Feature matrix
        labels: Cluster assignments

    Returns:
        Dictionary with clustering metrics
    """
    return {
        "silhouette_score": silhouette_score(X, labels),
        "cluster_counts": dict(Counter(labels)),
    }


def plot_confusion_matrix(cm, target_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        target_names: Names of target classes
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=target_names if target_names is not None else "auto",
        yticklabels=target_names if target_names is not None else "auto",
        ax=ax,
        cbar_kws={'label': 'Count'},
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    return fig


def plot_roc_curve(y_true, y_proba, figsize=(8, 6)):
    """
    Plot ROC curve.

    Args:
        y_true: True labels (binary)
        y_proba: Predicted probabilities
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if len(np.unique(y_true)) != 2:
        print("ROC curve only available for binary classification")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    return fig


def plot_precision_recall_curve(y_true, y_proba, figsize=(8, 6)):
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels (binary)
        y_proba: Predicted probabilities
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if len(np.unique(y_true)) != 2:
        print("PR curve only available for binary classification")
        return None

    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    return fig


def evaluate_model_performance(y_true, y_pred, y_proba=None, name="Model"):
    """
    Print comprehensive evaluation report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        name: Model name
    """
    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}\n")

    if len(np.unique(y_true)) == 2:
        summary = binary_classification_summary(y_true, y_pred, y_proba)
    else:
        summary = classification_summary(y_true, y_pred, y_proba)

    print(f"Accuracy:  {summary['accuracy']:.4f}")
    print(f"F1-Score:  {summary['f1_score']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall:    {summary['recall']:.4f}")
    print(f"MCC:       {summary['mcc']:.4f}")

    if "sensitivity" in summary:
        print("\nMedical Metrics:")
        print(f"Sensitivity (TPR): {summary['sensitivity']:.4f}")
        print(f"Specificity (TNR): {summary['specificity']:.4f}")
        print(f"PPV:               {summary['ppv']:.4f}")
        print(f"NPV:               {summary['npv']:.4f}")

    if "roc_auc" in summary:
        print(f"ROC-AUC:   {summary['roc_auc']:.4f}")

    print(f"\n{summary['report']}")
