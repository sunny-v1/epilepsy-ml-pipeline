"""
Advanced feature extraction for EEG signals and biomedical data.
Implements time-domain, frequency-domain, wavelet, and statistical features.
"""

import numpy as np
import pywt
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import kurtosis, skew


class SignalFeatureExtractor:
    """Extract comprehensive feature set from time-series signals."""

    @staticmethod
    def time_domain_features(x: np.ndarray) -> dict:
        """Extract time-domain statistical features."""
        if len(x) == 0:
            return {}

        return {
            'mean': np.mean(x),
            'std': np.std(x),
            'var': np.var(x),
            'min': np.min(x),
            'max': np.max(x),
            'median': np.median(x),
            'skewness': skew(x),
            'kurtosis': kurtosis(x),
            'peak_to_peak': np.ptp(x),
            'rms': np.sqrt(np.mean(x ** 2)),
            'energy': np.sum(x ** 2),
            'entropy': -np.sum(np.abs(x) * np.log(np.abs(x) + 1e-10)),
        }

    @staticmethod
    def hjorth_parameters(x: np.ndarray) -> dict:
        """Calculate Hjorth parameters (activity, mobility, complexity)."""
        dx = np.diff(x)
        ddx = np.diff(dx)

        activity = np.var(x)
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / activity) if activity > 0 else 0
        complexity = (
            np.sqrt(np.var(ddx) / var_dx) / mobility
            if var_dx > 0 and mobility > 0 else 0
        )

        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity,
        }

    @staticmethod
    def frequency_domain_features(x: np.ndarray, fs: float = 1.0) -> dict:
        """Extract frequency-domain features using FFT and Welch's method."""
        features = {}

        # Power spectral density using Welch's method
        freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))

        if len(psd) > 0:
            psd_sum = np.sum(psd)
            features['psd_mean'] = np.mean(psd)
            features['psd_std'] = np.std(psd)
            features['psd_max'] = np.max(psd)
            features['psd_energy'] = psd_sum

            if psd_sum > 0:
                centroid = np.sum(freqs * psd) / psd_sum
                features['spectral_centroid'] = centroid
                features['spectral_bandwidth'] = (
                    np.sqrt(np.sum(((freqs - centroid) ** 2) * psd)) / psd_sum
                )

        # FFT magnitude spectrum
        fft_vals = np.abs(fft(x))
        features['fft_mean'] = np.mean(fft_vals)
        features['fft_std'] = np.std(fft_vals)
        features['fft_max'] = np.max(fft_vals)

        return features

    @staticmethod
    def wavelet_features(x: np.ndarray, wavelet: str = 'db4', level: int = 3) -> dict:
        """Extract wavelet decomposition features."""
        features = {}

        try:
            coeffs = pywt.wavedec(x, wavelet, level=min(level, int(np.log2(len(x)))))
        except ValueError:
            return features

        for i, coeff in enumerate(coeffs):
            prefix = f'wavelet_d{i}' if i > 0 else 'wavelet_a'
            features[f'{prefix}_mean'] = np.mean(coeff)
            features[f'{prefix}_std'] = np.std(coeff)
            features[f'{prefix}_energy'] = np.sum(coeff ** 2)
            features[f'{prefix}_entropy'] = -np.sum(np.abs(coeff) * np.log(np.abs(coeff) + 1e-10))

        return features

    @staticmethod
    def zero_crossings(x: np.ndarray) -> dict:
        """Calculate zero crossing rate and other crossing features."""
        zero_crossings = np.sum(np.abs(np.diff(np.sign(x)))) / 2

        return {
            'zero_crossing_rate': zero_crossings / len(x) if len(x) > 0 else 0,
            'zero_crossings': zero_crossings,
        }

    @classmethod
    def _feature_dict(cls, x: np.ndarray, fs: float, include_wavelets: bool) -> dict:
        features = {}
        features.update(cls.time_domain_features(x))
        features.update(cls.hjorth_parameters(x))
        features.update(cls.zero_crossings(x))
        features.update(cls.frequency_domain_features(x, fs))
        # Wavelets are optional due to computational cost
        if include_wavelets:
            features.update(cls.wavelet_features(x))
        return features

    @classmethod
    def extract_all_features(
        cls,
        x: np.ndarray,
        fs: float = 1.0,
        include_wavelets: bool = True,
    ) -> np.ndarray:
        """
        Extract all available features from signal.

        Args:
            x: Input signal (1D array)
            fs: Sampling frequency
            include_wavelets: Whether to include wavelet features

        Returns:
            1D feature vector, ordered by sorted feature name
        """
        features = cls._feature_dict(x, fs, include_wavelets)
        feature_vector = np.array([features[k] for k in sorted(features)])
        return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def get_feature_names(cls, include_wavelets: bool = True) -> list:
        """Get names of all extracted features, in the order of `extract_all_features`."""
        dummy_signal = np.random.default_rng(0).standard_normal(256)
        return sorted(cls._feature_dict(dummy_signal, 1.0, include_wavelets))
