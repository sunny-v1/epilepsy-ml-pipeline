"""
Deep learning models for biomedical classification.
Implements MLP, CNN, and attention-based architectures.
"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data."""

    def __init__(self, input_dim: int, n_classes: int, hidden_dims=None):
        """
        Initialize MLP.

        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    """Deep Multi-Layer Perceptron with batch normalization."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class ConvNet1D(nn.Module):
    """1D Convolutional Neural Network for time-series data."""

    def __init__(self, input_channels: int, n_classes: int, seq_length: int = 256):
        """
        Initialize 1D CNN.

        Args:
            input_channels: Number of input channels
            n_classes: Number of output classes
            seq_length: Length of input sequence
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # Calculate flattened size
        conv_output_size = 128 * (seq_length // 8)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class AttentionBlock(nn.Module):
    """Self-attention block for neural networks."""

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        return out


class AttentionMLP(nn.Module):
    """MLP with attention mechanism."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()

        self.embedding = nn.Linear(input_dim, 128)
        self.attention = AttentionBlock(128)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.embedding(x)  # (batch_size, 128)
        x = x.unsqueeze(1)  # (batch_size, 1, 128)
        x = self.attention(x)  # (batch_size, 1, 128)
        x = x.squeeze(1)  # (batch_size, 128)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connections."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResidualMLP(nn.Module):
    """MLP with residual connections."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, 128)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


def get_device():
    """Get available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
