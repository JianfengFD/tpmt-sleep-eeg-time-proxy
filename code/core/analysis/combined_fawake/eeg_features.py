#!/usr/bin/env python3
"""Dataset-harmonized EEG feature extraction for 30-second epochs."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
from scipy import signal, stats


TARGET_RATE_HZ = 100.0
EPOCH_SECONDS = 30.0
MAX_ADJACENT_REPEAT_TAU_S = 0.10
CANONICAL_POSITIONS = ("F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2")
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
    "gamma": (30.0, 45.0),
}


def canonical_channel_names(raw: mne.io.BaseRaw, dataset: str) -> list[str]:
    suffix = "-REF" if dataset == "Zhang" else ""
    expected = [f"{name}{suffix}" for name in CANONICAL_POSITIONS]
    missing = [name for name in expected if name not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing canonical channels: {missing}")
    return expected


def valid_signal_end_s(dataset: str, metadata_duration_s: float, raw_duration_s: float) -> float:
    """Return the end of the stageable signal.

    Zhang files contain about five seconds after the last scored epoch/report
    marker. The prior verified extraction used 68 s as the end of the labelled
    portion of each nominal 73 s record. Kumral files end immediately before the
    awakening, so no offset is applied.
    """
    end = min(float(metadata_duration_s), float(raw_duration_s))
    if dataset == "Zhang":
        end -= 5.0
    if end < EPOCH_SECONDS:
        raise ValueError(f"Stageable duration is too short: {end:.3f}s")
    return end


def reverse_aligned_epochs(valid_end_s: float) -> tuple[list[tuple[float, float]], float]:
    """Create complete 30-s epochs backwards so the final epoch matches label."""
    count = int(math.floor(valid_end_s / EPOCH_SECONDS))
    prefix_s = valid_end_s - count * EPOCH_SECONDS
    epochs = [
        (prefix_s + index * EPOCH_SECONDS, prefix_s + (index + 1) * EPOCH_SECONDS)
        for index in range(count)
    ]
    return epochs, prefix_s


def _resample_and_filter(data_uv: np.ndarray, source_rate_hz: float) -> np.ndarray:
    data = np.asarray(data_uv, dtype=np.float32)
    data -= np.mean(data, axis=0, keepdims=True)  # 8-position common average
    ratio = Fraction(TARGET_RATE_HZ / source_rate_hz).limit_denominator(1000)
    if not math.isclose(source_rate_hz, TARGET_RATE_HZ):
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator, axis=1)
    sos = signal.butter(4, [0.5, 45.0], btype="bandpass", fs=TARGET_RATE_HZ, output="sos")
    data = signal.sosfiltfilt(sos, data, axis=1).astype(np.float32, copy=False)
    return data


def load_stageable_record(
    path: Path,
    dataset: str,
    metadata_duration_s: float,
) -> tuple[np.ndarray, float, list[tuple[float, float]], float, list[str]]:
    """Read and harmonize the complete stageable part of one EDF."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    source_rate = float(raw.info["sfreq"])
    raw_duration = float(raw.n_times / source_rate)
    valid_end = valid_signal_end_s(dataset, metadata_duration_s, raw_duration)
    epochs, prefix_s = reverse_aligned_epochs(valid_end)
    channels = canonical_channel_names(raw, dataset)
    stop = min(int(raw.n_times), int(round(valid_end * source_rate)))
    data_uv = raw.get_data(picks=channels, start=0, stop=stop) * 1e6
    prepared = _resample_and_filter(data_uv, source_rate)
    expected = int(round(valid_end * TARGET_RATE_HZ))
    prepared = prepared[:, :expected]
    return prepared, valid_end, epochs, prefix_s, channels


def _robust_epoch(epoch: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.median(epoch, axis=1, keepdims=True)
    mad = np.median(np.abs(epoch - center), axis=1, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    z = (epoch - center) / scale
    artifact_fraction = float(np.mean(np.abs(z) > 12.0))
    return np.clip(z, -12.0, 12.0), artifact_fraction


def _structure_residuals(epoch_z: np.ndarray, half_window_s: float) -> np.ndarray:
    """Return the local log-structure-function residuals underlying S.

    Keeping the residual computation separate lets a gamma grid be evaluated
    without repeatedly fitting the same local power law.  This is exactly the
    residual definition used by the original ``_structure_score`` routine.
    """
    center = epoch_z.shape[1] // 2
    half = int(round(half_window_s * TARGET_RATE_HZ))
    window = epoch_z[:, center - half : center + half]
    max_lag_s = min(2.0, half_window_s * 0.8, (window.shape[1] - 2) / TARGET_RATE_HZ)
    lags = np.unique(
        np.round(np.geomspace(1.0 / TARGET_RATE_HZ, max_lag_s, 12) * TARGET_RATE_HZ).astype(int)
    )
    lags = lags[(lags >= 1) & (lags < window.shape[1] - 1)]
    residuals: list[np.ndarray] = []
    log_tau = np.log(lags / TARGET_RATE_HZ)
    for channel in window:
        structure = []
        for lag in lags:
            increments = channel[lag:] - channel[:-lag]
            structure.append(max(float(np.median(increments * increments)), 1e-12))
        values = np.log(np.asarray(structure))
        slope, intercept = np.polyfit(log_tau, values, 1)
        residuals.append(np.abs(values - (slope * log_tau + intercept)))
    return np.concatenate(residuals)


def structure_scores(
    epoch_z: np.ndarray,
    half_window_s: float,
    gammas: Iterable[float],
) -> dict[float, float]:
    """Evaluate S(t; Delta, gamma) for several thresholds from one fit."""
    residuals = _structure_residuals(epoch_z, half_window_s)
    return {float(gamma): float(np.mean(residuals <= gamma)) for gamma in gammas}


def adjacent_repeat_similarities(
    epoch_z: np.ndarray,
    tau_s: float,
    *,
    context_half_window_s: float = 2.0,
    sample_rate_hz: float = TARGET_RATE_HZ,
    stride_samples: int = 1,
) -> np.ndarray:
    """Return signed adjacent-block repeat similarities around an epoch center.

    For every sliding start ``u`` in the centered context, this compares the
    consecutive blocks ``x[u:u+tau]`` and ``x[u+tau:u+2*tau]``.  Pearson
    correlation is computed separately for every EEG channel after removing
    each block's own mean.  The returned value at ``u`` is the median valid
    channel correlation.  Thus a result near one requires the same local
    waveform, with the same polarity, to recur immediately after ``tau``.

    Keeping the correlation signed is intentional: two half-cycles with
    opposite polarity are not counted as a repeat.  Constant channel/block
    pairs have undefined Pearson correlation and are omitted.  Starts for
    which every channel is undefined are also omitted.

    ``tau_s`` is restricted to at most 0.1 s because this statistic represents
    the proposed short-scale adjacent-repeat branch, rather than the longer
    scale-law statistic :func:`structure_scores`.
    """
    data = np.asarray(epoch_z)
    if data.ndim != 2:
        raise ValueError("epoch_z must have shape (channels, samples)")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not math.isfinite(tau_s) or tau_s <= 0 or tau_s > MAX_ADJACENT_REPEAT_TAU_S + 1e-12:
        raise ValueError(f"tau_s must satisfy 0 < tau_s <= {MAX_ADJACENT_REPEAT_TAU_S:g}")
    if not math.isfinite(context_half_window_s) or context_half_window_s <= 0:
        raise ValueError("context_half_window_s must be positive and finite")
    if isinstance(stride_samples, (bool, np.bool_)) or int(stride_samples) != stride_samples:
        raise ValueError("stride_samples must be a positive integer")
    stride = int(stride_samples)
    if stride <= 0:
        raise ValueError("stride_samples must be a positive integer")

    block_samples = int(round(tau_s * sample_rate_hz))
    if block_samples < 3:
        raise ValueError("tau_s must contain at least three samples for Pearson correlation")
    half_context_samples = int(round(context_half_window_s * sample_rate_hz))
    if half_context_samples < block_samples:
        raise ValueError("The centered context must contain at least two adjacent tau blocks")
    if data.shape[1] < 2 * half_context_samples:
        raise ValueError("epoch_z is shorter than the requested centered context")

    center = data.shape[1] // 2
    context = np.asarray(
        data[:, center - half_context_samples : center + half_context_samples],
        dtype=np.float64,
    )
    starts = np.arange(0, context.shape[1] - 2 * block_samples + 1, stride, dtype=int)
    if starts.size == 0:
        return np.empty(0, dtype=float)
    offsets = np.arange(block_samples, dtype=int)
    left = context[:, starts[:, None] + offsets[None, :]]
    right = context[:, starts[:, None] + block_samples + offsets[None, :]]
    left -= np.mean(left, axis=2, keepdims=True)
    right -= np.mean(right, axis=2, keepdims=True)
    numerator = np.sum(left * right, axis=2)
    denominator = np.sqrt(np.sum(left * left, axis=2) * np.sum(right * right, axis=2))
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
    correlations = np.full(numerator.shape, np.nan, dtype=float)
    correlations[valid] = numerator[valid] / denominator[valid]
    correlations = np.clip(correlations, -1.0, 1.0)

    # Avoid numpy's all-NaN-slice warning while retaining the mathematically
    # meaningful rule that undefined channel pairs contribute no vote.
    valid_start = np.any(np.isfinite(correlations), axis=0)
    if not np.any(valid_start):
        return np.empty(0, dtype=float)
    return np.nanmedian(correlations[:, valid_start], axis=0)


def adjacent_repeat_scores(
    epoch_z: np.ndarray,
    tau_s: float,
    etas: Iterable[float],
    *,
    context_half_window_s: float = 2.0,
    sample_rate_hz: float = TARGET_RATE_HZ,
    stride_samples: int = 1,
) -> dict[float, float]:
    """Evaluate the adjacent-repeat fraction for several thresholds.

    If ``r_med(u; tau)`` is the median channel correlation returned by
    :func:`adjacent_repeat_similarities`, the score is

    ``A(t; context, tau, eta) = mean_u[r_med(u; tau) >= eta]``.

    Several ``eta`` values reuse the same correlations, which makes a grid
    search deterministic and substantially cheaper than repeated extraction.
    """
    thresholds = [float(eta) for eta in etas]
    if any(not math.isfinite(eta) or eta < -1.0 or eta > 1.0 for eta in thresholds):
        raise ValueError("Every eta must be finite and lie in [-1, 1]")
    similarities = adjacent_repeat_similarities(
        epoch_z,
        tau_s,
        context_half_window_s=context_half_window_s,
        sample_rate_hz=sample_rate_hz,
        stride_samples=stride_samples,
    )
    if similarities.size == 0:
        return {eta: math.nan for eta in thresholds}
    return {eta: float(np.mean(similarities >= eta)) for eta in thresholds}


def _structure_score(epoch_z: np.ndarray, half_window_s: float, gamma: float = 0.18) -> float:
    return structure_scores(epoch_z, half_window_s, [gamma])[float(gamma)]


def extract_epoch_features(epoch_uv: np.ndarray) -> dict[str, float]:
    """Extract scale-free spectral, temporal, and structure features."""
    if epoch_uv.shape[0] != len(CANONICAL_POSITIONS):
        raise ValueError("Expected eight common-position EEG channels")
    if epoch_uv.shape[1] < int(20 * TARGET_RATE_HZ):
        raise ValueError("Epoch must contain at least 20 seconds")
    epoch_z, artifact_fraction = _robust_epoch(epoch_uv)
    nperseg = min(epoch_z.shape[1], int(4 * TARGET_RATE_HZ))
    freqs, psd = signal.welch(
        epoch_z,
        fs=TARGET_RATE_HZ,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=1,
        detrend="linear",
    )
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    integrate = getattr(np, "trapezoid", np.trapz)
    total = np.maximum(integrate(psd[:, total_mask], freqs[total_mask], axis=1), 1e-12)
    features: dict[str, float] = {"artifact_fraction_z12": artifact_fraction}
    relative: dict[str, np.ndarray] = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        relative[name] = integrate(psd[:, mask], freqs[mask], axis=1) / total
        features[f"rel_{name}_median"] = float(np.median(relative[name]))
        features[f"rel_{name}_iqr"] = float(np.quantile(relative[name], 0.75) - np.quantile(relative[name], 0.25))

    selected_psd = np.maximum(psd[:, total_mask], 1e-15)
    selected_freqs = freqs[total_mask]
    probability = selected_psd / selected_psd.sum(axis=1, keepdims=True)
    entropy = -np.sum(probability * np.log(probability), axis=1) / np.log(probability.shape[1])
    slopes = []
    edges = []
    for row in selected_psd:
        slopes.append(float(np.polyfit(np.log(selected_freqs), np.log(row), 1)[0]))
        cumulative = np.cumsum(row)
        edges.append(float(selected_freqs[np.searchsorted(cumulative, 0.95 * cumulative[-1])]))
    features["spectral_entropy_median"] = float(np.median(entropy))
    features["spectral_entropy_iqr"] = float(np.quantile(entropy, 0.75) - np.quantile(entropy, 0.25))
    features["spectral_slope_median"] = float(np.median(slopes))
    features["spectral_edge95_median_hz"] = float(np.median(edges))

    dx = np.diff(epoch_z, axis=1)
    ddx = np.diff(dx, axis=1)
    variance = np.maximum(np.var(epoch_z, axis=1), 1e-12)
    dx_variance = np.maximum(np.var(dx, axis=1), 1e-12)
    mobility = np.sqrt(dx_variance / variance)
    complexity = np.sqrt(np.maximum(np.var(ddx, axis=1), 1e-12) / dx_variance) / np.maximum(mobility, 1e-12)
    features["hjorth_mobility_median"] = float(np.median(mobility))
    features["hjorth_complexity_median"] = float(np.median(complexity))
    features["line_length_median"] = float(np.median(np.mean(np.abs(dx), axis=1)))
    features["zero_crossing_rate_median"] = float(
        np.median(np.mean(np.signbit(epoch_z[:, 1:]) != np.signbit(epoch_z[:, :-1]), axis=1))
    )
    features["kurtosis_median"] = float(np.median(stats.kurtosis(epoch_z, axis=1, fisher=True, bias=False)))
    features["log_alpha_delta_ratio"] = float(
        np.log((np.median(relative["alpha"]) + 1e-6) / (np.median(relative["delta"]) + 1e-6))
    )
    features["log_beta_delta_ratio"] = float(
        np.log((np.median(relative["beta"]) + 1e-6) / (np.median(relative["delta"]) + 1e-6))
    )
    for half_window_s in (0.5, 2.0, 7.5):
        features[f"structure_S_d{str(half_window_s).replace('.', 'p')}_g0p18"] = _structure_score(
            epoch_z, half_window_s, 0.18
        )
    return features


def extract_record_epochs(
    path: Path,
    dataset: str,
    metadata_duration_s: float,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    data, valid_end, epochs, prefix_s, channels = load_stageable_record(path, dataset, metadata_duration_s)
    rows: list[dict[str, float]] = []
    for index, (start_s, end_s) in enumerate(epochs):
        start = int(round(start_s * TARGET_RATE_HZ))
        stop = int(round(end_s * TARGET_RATE_HZ))
        features = extract_epoch_features(data[:, start:stop])
        rows.append(
            {
                "epoch_index": index,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "duration_s": float(end_s - start_s),
                **features,
            }
        )
    metadata = {
        "valid_end_s": float(valid_end),
        "unmodelled_prefix_s": float(prefix_s),
        "n_epochs": len(epochs),
        "channels": channels,
        "target_rate_hz": TARGET_RATE_HZ,
    }
    return rows, metadata


def feature_columns(rows: Iterable[dict[str, float]]) -> list[str]:
    excluded = {"epoch_index", "start_s", "end_s", "duration_s"}
    first = next(iter(rows))
    return [key for key in first if key not in excluded]
