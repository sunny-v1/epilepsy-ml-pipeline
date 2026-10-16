"""
Training utilities for both classical ML and deep learning models.
Includes hyperparameter tuning, cross-validation, and PyTorch training loops.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split

from src.models.deep import get_device


def _to_tensor(data, dtype, device):
    """Convert a DataFrame/Series, ndarray, or tensor to a tensor on `device`."""
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(data), dtype=dtype, device=device)


def train_sklearn(model, X_train, y_train, **kwargs):
    """
    Train a scikit-learn model.

    Args:
        model: Sklearn model instance
        X_train: Training features
        y_train: Training labels
        **kwargs: Unused; kept for API compatibility

    Returns:
        Trained model
    """
    return model.fit(X_train, y_train)


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, n_jobs=-1):
    """
    Perform grid search for hyperparameter tuning.

    Args:
        model: Sklearn model
        param_grid: Parameter grid for search
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        n_jobs: Number of parallel jobs

    Returns:
        Tuple of (best_model, best_params, cv_results)
    """
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def cross_val_evaluate(model, X_train, y_train, cv=5):
    """
    Perform cross-validation evaluation.

    Args:
        model: Sklearn model
        X_train: Training features
        y_train: Training labels
        cv: Number of folds

    Returns:
        Dictionary with cross-validation results
    """
    scoring = {
        'accuracy': 'accuracy',
        'f1_weighted': 'f1_weighted',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
    }
    return cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)


def train_torch(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    lr=1e-3,
    batch_size=32,
    device=None,
    early_stopping=True,
    patience=5,
    verbose=True,
):
    """
    Train a PyTorch classifier with optional early stopping.

    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device to use (defaults to CUDA if available)
        early_stopping: Whether to use early stopping on validation loss
        patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, history_dict)
    """
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = _to_tensor(X_train, torch.float32, device)
    y_train_t = _to_tensor(y_train, torch.long, device)
    X_val_t = _to_tensor(X_val, torch.float32, device)
    y_val_t = _to_tensor(y_val, torch.long, device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_train = len(train_dataset)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "train_acc": []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            epoch_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        train_loss = epoch_loss / n_train
        train_acc = epoch_correct / n_train
        val_loss, val_acc = evaluate_torch(model, X_val_t, y_val_t, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

    return model, history


def evaluate_torch(model, X, y, device=None):
    """
    Evaluate a PyTorch classifier.

    Args:
        model: PyTorch model
        X: Features (tensor, ndarray, or DataFrame)
        y: Labels (tensor, ndarray, or Series)
        device: Device to use

    Returns:
        Tuple of (loss, accuracy)
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    X = _to_tensor(X, torch.float32, device)
    y = _to_tensor(y, torch.long, device)

    with torch.no_grad():
        logits = model(X)
        loss = F.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(dim=1) == y).float().mean().item()

    return loss, accuracy


def predict_torch(model, X, device=None):
    """
    Make predictions with a PyTorch classifier.

    Args:
        model: PyTorch model
        X: Features (tensor, ndarray, or DataFrame)
        device: Device to use

    Returns:
        Tuple of (predicted labels, class probabilities) as numpy arrays
    """
    device = device or get_device()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(_to_tensor(X, torch.float32, device))
        proba = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

    return pred.cpu().numpy(), proba.cpu().numpy()


def split_data(df, target_col="target", test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into stratified train, validation, and test sets.

    Args:
        df: Input dataframe
        target_col: Name of target column
        test_size: Test set fraction (of the full dataset)
        val_size: Validation set fraction (of the full dataset)
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
