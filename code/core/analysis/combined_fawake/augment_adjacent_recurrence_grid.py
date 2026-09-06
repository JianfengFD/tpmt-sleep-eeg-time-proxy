#!/usr/bin/env python3
"""Add a physical short-lag adjacent-recurrence grid to EEG epochs.

The statistic is designed to operationalize the ``densely repeat`` idea while
not confusing ordinary smoothness with recurrence.  For the robustly scaled
EEG of channel ``c`` in the central context ``[t-H, t+H]``, define the Pearson
lag autocorrelation

    C_c(t; H, tau) = corr(x_c(u), x_c(u+tau)).

The two vectors in the correlation are demeaned separately.  The signed
return prominence is

    P_c(t; H, tau) = [C_c(t; H, tau) - C_c(t; H, tau/2)] / 2,

and the eight-channel robust aggregate is

    P(t; H, tau) = median_c P_c(t; H, tau),
    F_R(t; H, tau) = [1 + P(t; H, tau)] / 2.

Thus P and F_R lie in [-1, 1] and [0, 1], respectively.  A genuinely repeated
waveform with period tau has high C(tau) relative to C(tau/2), whereas a slow,
merely smooth signal has C(tau) approximately equal to C(tau/2) and therefore
P approximately zero.  Keeping P signed is important: the bounded fusion

    F = F_S + alpha F_S (1-F_S) P

can increase or decrease the self-similarity score without moving the exact
endpoints F_S=0 or F_S=1.

All lags are implemented in integer samples after the shared 100-Hz
preprocessing.  The output records both requested and effective lag values so
the discretization is explicit.  Dream durations and stage labels are never
read by this extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.combined_fawake.eeg_features import (
    TARGET_RATE_HZ,
    _robust_epoch,
    load_stageable_record,
)


CONTEXT_HALF_WINDOWS_S = (0.5, 1.0, 2.0, 5.0)
RETURN_LAGS_S = (0.04, 0.05, 0.06, 0.08, 0.10)
CACHE_SCHEMA_VERSION = "adjacent-return-prominence-v1"


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def prominence_column(half_window_s: float, tau_s: float) -> str:
    return f"recurrence_P_h{token(half_window_s)}_tau{token(tau_s)}"


def recurrence_column(half_window_s: float, tau_s: float) -> str:
    return f"recurrence_Fr_h{token(half_window_s)}_tau{token(tau_s)}"


def _channel_lag_correlation(window: np.ndarray, lag_samples: int) -> np.ndarray:
    """Pearson lag correlation for each channel, using paired valid samples."""
    if lag_samples < 1 or lag_samples >= window.shape[1] - 1:
        raise ValueError("lag must leave at least two paired samples")
    left = window[:, :-lag_samples]
    right = window[:, lag_samples:]
    left = left - np.mean(left, axis=1, keepdims=True)
    right = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(
        np.sum(left * left, axis=1) * np.sum(right * right, axis=1)
    )
    return np.clip(numerator / np.maximum(denominator, 1.0e-12), -1.0, 1.0)


def adjacent_return_prominence(
    epoch_z: np.ndarray,
    half_window_s: float,
    tau_s: float,
) -> tuple[float, float, dict[str, float | int]]:
    """Return ``(P, F_R, lag metadata)`` for one robust-scaled EEG epoch."""
    center = epoch_z.shape[1] // 2
    half_window_samples = int(round(half_window_s * TARGET_RATE_HZ))
    if 2 * half_window_samples > epoch_z.shape[1]:
        raise ValueError("context does not fit inside the epoch")
    window = epoch_z[
        :, center - half_window_samples : center + half_window_samples
    ]
    tau_samples = max(1, int(round(tau_s * TARGET_RATE_HZ)))
    half_tau_samples = max(1, int(round(tau_samples / 2.0)))
    correlation_tau = _channel_lag_correlation(window, tau_samples)
    correlation_half_tau = _channel_lag_correlation(window, half_tau_samples)
    channel_prominence = np.clip(
        (correlation_tau - correlation_half_tau) / 2.0,
        -1.0,
        1.0,
    )
    prominence = float(np.median(channel_prominence))
    recurrence = float((1.0 + prominence) / 2.0)
    metadata: dict[str, float | int] = {
        "tau_samples": tau_samples,
        "half_tau_samples": half_tau_samples,
        "effective_tau_s": tau_samples / TARGET_RATE_HZ,
        "effective_half_tau_s": half_tau_samples / TARGET_RATE_HZ,
    }
    return prominence, recurrence, metadata


def extract_recurrence_grid(epoch_uv: np.ndarray) -> dict[str, float]:
    """Extract the fixed label-independent recurrence grid from one epoch."""
    epoch_z, _ = _robust_epoch(epoch_uv)
    features: dict[str, float] = {}
    for half_window_s in CONTEXT_HALF_WINDOWS_S:
        for tau_s in RETURN_LAGS_S:
            prominence, recurrence, _ = adjacent_return_prominence(
                epoch_z, half_window_s, tau_s
            )
            features[prominence_column(half_window_s, tau_s)] = prominence
            features[recurrence_column(half_window_s, tau_s)] = recurrence
    return features


def discover_tasks(feature_cache: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in feature_cache.glob("*.json"):
        if path.name.startswith("._"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_task = payload["task"]
        source = Path(str(source_task["file_path"]))
        source_stat = source.stat()
        uid = str(source_task["record_uid"])
        tasks[uid] = {
            "record_uid": uid,
            "dataset": str(source_task["dataset"]),
            "file_path": str(source),
            "metadata_duration_s": float(source_task["metadata_duration_s"]),
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
        }
    return tasks


def algorithm_signature() -> str:
    eeg_features_path = Path(__file__).with_name("eeg_features.py")
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "contexts_s": CONTEXT_HALF_WINDOWS_S,
        "return_lags_s": RETURN_LAGS_S,
        "target_rate_hz": TARGET_RATE_HZ,
        "eeg_features_sha256": hashlib.sha256(eeg_features_path.read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "definition": "median-channel signed C(tau)-C(tau/2) return prominence",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_cache_signature(task: dict[str, Any], algorithm_hash: str) -> str:
    payload = {
        "algorithm": algorithm_hash,
        "record_uid": str(task["record_uid"]),
        "dataset": str(task["dataset"]),
        "file_path": str(Path(task["file_path"]).resolve()),
        "metadata_duration_s": float(task["metadata_duration_s"]),
        "source_size_bytes": int(task["source_size_bytes"]),
        "source_mtime_ns": int(task["source_mtime_ns"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path, uid: str) -> Path:
    digest = hashlib.sha1(uid.encode()).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


def calculate_record(task: dict[str, Any]) -> dict[str, Any]:
    data, _, epochs, _, _ = load_stageable_record(
        Path(task["file_path"]),
        str(task["dataset"]),
        float(task["metadata_duration_s"]),
    )
    rows: list[dict[str, float | int]] = []
    for epoch_index, (start_s, end_s) in enumerate(epochs):
        start = int(round(start_s * TARGET_RATE_HZ))
        stop = int(round(end_s * TARGET_RATE_HZ))
        rows.append(
            {
                "epoch_index": epoch_index,
                **extract_recurrence_grid(data[:, start:stop]),
            }
        )
    return {
        "record_uid": str(task["record_uid"]),
        "cache_signature": str(task["cache_signature"]),
        "epochs": rows,
    }


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-features", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--record-cache", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.epoch_features, low_memory=False)
    tasks = discover_tasks(args.feature_cache)
    required_uids = list(dict.fromkeys(frame["record_uid"].astype(str)))
    missing = sorted(set(required_uids) - set(tasks))
    if missing:
        raise RuntimeError(f"Missing source metadata for {missing[:5]}")

    cache_dir = args.record_cache or args.output.parent / ".adjacent_recurrence_record_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    algorithm_hash = algorithm_signature()
    results: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for uid in required_uids:
        task = dict(tasks[uid])
        task["cache_signature"] = expected_cache_signature(task, algorithm_hash)
        path = cache_path(cache_dir, uid)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("cache_signature") == task["cache_signature"]
                and len(payload.get("epochs", []))
                == int(frame["record_uid"].eq(uid).sum())
            ):
                results[uid] = payload
                continue
        pending.append(task)

    print(
        f"records={len(required_uids)} cached={len(results)} "
        f"pending={len(pending)} workers={max(1, args.workers)}",
        flush=True,
    )
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(calculate_record, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                payload = future.result()
                uid = str(payload["record_uid"])
                results[uid] = payload
                write_cache(cache_path(cache_dir, uid), payload)
                if completed == 1 or completed % 10 == 0 or completed == len(pending):
                    print(f"completed {completed}/{len(pending)}: {task['record_uid']}", flush=True)

    lookup: dict[tuple[str, int], dict[str, float | int]] = {}
    for uid, payload in results.items():
        for row in payload["epochs"]:
            lookup[(uid, int(row["epoch_index"]))] = row
    keys = list(zip(frame["record_uid"].astype(str), frame["epoch_index"].astype(int), strict=True))
    if any(key not in lookup for key in keys):
        raise RuntimeError("At least one epoch is missing from the recurrence cache")

    columns = sorted(
        {
            name
            for half_window_s in CONTEXT_HALF_WINDOWS_S
            for tau_s in RETURN_LAGS_S
            for name in (
                prominence_column(half_window_s, tau_s),
                recurrence_column(half_window_s, tau_s),
            )
        }
    )
    for column in columns:
        frame[column] = np.asarray([lookup[key][column] for key in keys], dtype=float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    lag_metadata = {}
    for tau_s in RETURN_LAGS_S:
        # Only metadata is used; avoid evaluating a constant-signal score.
        tau_samples = max(1, int(round(tau_s * TARGET_RATE_HZ)))
        half_tau_samples = max(1, int(round(tau_samples / 2.0)))
        lag_metadata[str(tau_s)] = {
            "tau_samples": tau_samples,
            "half_tau_samples": half_tau_samples,
            "effective_tau_s": tau_samples / TARGET_RATE_HZ,
            "effective_half_tau_s": half_tau_samples / TARGET_RATE_HZ,
        }
    manifest = {
        "definition": "P=median_channel((C(tau)-C(tau/2))/2); Fr=(1+P)/2",
        "input": str(args.epoch_features),
        "output": str(args.output),
        "records": len(required_uids),
        "epochs": len(frame),
        "context_half_windows_s": CONTEXT_HALF_WINDOWS_S,
        "requested_return_lags_s": RETURN_LAGS_S,
        "effective_lags": lag_metadata,
        "columns_added": columns,
        "cache_schema": CACHE_SCHEMA_VERSION,
        "algorithm_signature": algorithm_hash,
        "cache_validation": (
            "record UID, resolved source path, source size/mtime, fixed grid, "
            "cache schema, and SHA-256 of extractor plus shared eeg_features.py"
        ),
        "label_usage": "none",
        "preprocessing": "shared load_stageable_record and per-epoch robust normalization",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output} with {len(columns)} recurrence columns", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
