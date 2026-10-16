"""Reproducibility helpers."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
