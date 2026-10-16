"""
EEG signal filtering utilities for preprocessing.
Implements various frequency domain filtering techniques.
"""

import numpy as np
from scipy import signal


def _clip_normalized(freq_hz, fs):
    """Convert a cutoff in Hz to a normalized frequency safely inside (0, 1)."""
    return float(np.clip(freq_hz / (fs / 2.0), 0.001, 0.999))


def apply_eeg_filter(
    signal_data,
    fs=256,
    filter_type="bandpass",
    lowcut=1.0,
    highcut=50.0,
    order=4,
):
    """
    Apply frequency domain filter to EEG signal.

    Args:
        signal_data: Input signal array
        fs: Sampling frequency
        filter_type: 'bandpass', 'highpass', 'lowpass', 'notch'
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal (the original signal if filtering fails)
    """
    if filter_type == "notch":
        return apply_notch_filter(signal_data, fs=fs)

    try:
        if filter_type == "bandpass":
            low = _clip_normalized(lowcut, fs)
            high = _clip_normalized(highcut, fs)
            if low >= high:
                low, high = 0.01, 0.99
            b, a = signal.butter(order, [low, high], btype='band')

        elif filter_type == "highpass":
            b, a = signal.butter(order, _clip_normalized(lowcut, fs), btype='high')

        elif filter_type == "lowpass":
            b, a = signal.butter(order, _clip_normalized(highcut, fs), btype='low')

        else:
            return signal_data

        # Forward-backward filtering for zero phase distortion
        return signal.filtfilt(b, a, signal_data)

    except ValueError as e:
        print(f"Warning: Filter application failed ({e}), returning original signal")
        return signal_data


def apply_notch_filter(signal_data, fs=256, freq=50.0, Q=30.0):
    """
    Apply notch filter to remove power line interference.

    Args:
        signal_data: Input signal
        fs: Sampling frequency
        freq: Frequency to notch in Hz (50 or 60)
        Q: Quality factor

    Returns:
        Filtered signal (the original signal if filtering fails)
    """
    try:
        b, a = signal.iirnotch(freq, Q, fs=fs)
        return signal.filtfilt(b, a, signal_data)
    except ValueError:
        return signal_data


def apply_artifact_removal(signal_data, method="simple_threshold", threshold=3.0):
    """
    Remove artifacts from EEG signal.

    Args:
        signal_data: Input signal
        method: 'simple_threshold' or 'iqr'
        threshold: Threshold value

    Returns:
        Signal with artifacts replaced by the median of nearby valid values
    """
    signal_clean = signal_data.copy()

    if method == "simple_threshold":
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        artifact_indices = np.abs(signal_data - mean) > (threshold * std)

    elif method == "iqr":
        q1, q3 = np.percentile(signal_data, [25, 75])
        iqr = q3 - q1
        artifact_indices = (signal_data < q1 - threshold * iqr) | (signal_data > q3 + threshold * iqr)

    else:
        return signal_clean

    window_size = 5
    for idx in np.flatnonzero(artifact_indices):
        start = max(0, idx - window_size)
        end = min(len(signal_data), idx + window_size + 1)
        valid_vals = signal_clean[start:end][~artifact_indices[start:end]]
        if len(valid_vals) > 0:
            signal_clean[idx] = np.median(valid_vals)

    return signal_clean


def common_average_reference(eeg_data):
    """
    Apply Common Average Reference (CAR) to multi-channel EEG.

    Args:
        eeg_data: EEG data (n_channels, n_timepoints)

    Returns:
        Referenced EEG data with same shape
    """
    if eeg_data.ndim != 2:
        return eeg_data
    return eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
