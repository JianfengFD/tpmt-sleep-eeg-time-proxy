#!/usr/bin/env python3
"""Apply the locked portable q75/beta=.25 R* model to one EEG record.

Input may be either a raw Zhang/Kumral EDF or one record in an epoch-feature
CSV. EDF mode repeats the shared preprocessing and calculates only the
self-similarity and return-prominence columns required by the portable JSON.
The output ends at the physical ``f_repeat_micro_v3`` trajectory and its
integral; it never reads dream text/duration, predicts stages, or applies g.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.combined_fawake.adjacent_repeat_v3_model import (
    RSTAR_COLUMNS,
    RSTAR_CONTEXT_HALF_WINDOW_S,
    RSTAR_TAU_GRID_S,
    apply_adjacent_repeat_v3,
    validate_adjacent_repeat_v3_model,
)
from analysis.combined_fawake.augment_adjacent_recurrence_grid import (
    adjacent_return_prominence,
)
from analysis.combined_fawake.eeg_features import (
    EPOCH_SECONDS,
    TARGET_RATE_HZ,
    _robust_epoch,
    load_stageable_record,
)
from analysis.combined_fawake.predict_dense_repeat_fawake_initial import (
    OUTPUT_METADATA_COLUMNS,
    _edf_header_duration_s,
    _validate_portable_physical_model,
    extract_selected_epoch_features,
    selected_physical_feature_columns,
)


OUTPUT_COMPONENT_COLUMNS = (
    "self_similarity_Fs_raw",
    "f_self_similarity",
    "return_prominence_P_old_raw",
    "return_prominence_R_old",
    "f_old_return_prominence",
    "return_prominence_P_star",
    "return_prominence_R_star",
    "f_repeat_micro_v3",
)


def validate_portable_payload(payload: dict[str, Any]) -> None:
    """Validate both the locked R* wrapper and its portable v2 base."""
    if payload.get("format") != "adjacent_repeat_logit_shift_portable_v3":
        raise ValueError("Expected adjacent_repeat_logit_shift_portable_v3 JSON")
    validate_adjacent_repeat_v3_model(payload)
    _validate_portable_physical_model(payload["base_model"])


def required_feature_columns(payload: dict[str, Any]) -> list[str]:
    """Return the minimal, ordered feature set needed by the portable model."""
    validate_portable_payload(payload)
    columns = [*selected_physical_feature_columns(payload["base_model"]), *RSTAR_COLUMNS]
    return list(dict.fromkeys(columns))


def extract_required_epoch_features(
    epoch_uv: np.ndarray, payload: dict[str, Any]
) -> dict[str, float]:
    """Calculate the locked S columns, old P, and two P* source columns."""
    features = extract_selected_epoch_features(epoch_uv, payload["base_model"])
    epoch_z, _ = _robust_epoch(np.asarray(epoch_uv))
    for column, tau_s in zip(RSTAR_COLUMNS, RSTAR_TAU_GRID_S, strict=True):
        prominence, _recurrence, _metadata = adjacent_return_prominence(
            epoch_z, RSTAR_CONTEXT_HALF_WINDOW_S, tau_s
        )
        features[column] = float(prominence)
    required = required_feature_columns(payload)
    return {column: float(features[column]) for column in required}


def extract_required_features_from_edf(
    path: Path,
    dataset: str,
    metadata_duration_s: float,
    payload: dict[str, Any],
    *,
    record_uid: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run shared preprocessing and extract one row per reverse-aligned epoch."""
    data, valid_end_s, epochs, prefix_s, channels = load_stageable_record(
        path, dataset, metadata_duration_s
    )
    uid = record_uid or f"prediction:{dataset}:{path.name}"
    rows: list[dict[str, Any]] = []
    for epoch_index, (start_s, end_s) in enumerate(epochs):
        start = int(round(start_s * TARGET_RATE_HZ))
        stop = int(round(end_s * TARGET_RATE_HZ))
        rows.append(
            {
                "record_uid": uid,
                "dataset": dataset,
                "filename": path.name,
                "split": "prediction",
                "manual_final_stage": "unknown",
                "epoch_index": epoch_index,
                "is_final_labelled_epoch": False,
                "valid_signal_end_s": valid_end_s,
                "unmodelled_prefix_s": prefix_s,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                **extract_required_epoch_features(data[:, start:stop], payload),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The EDF contains no complete reverse-aligned 30-s epoch")
    return frame, {
        "source_kind": "EDF",
        "edf": str(path.resolve()),
        "dataset": dataset,
        "metadata_duration_s": float(metadata_duration_s),
        "valid_signal_end_s": float(valid_end_s),
        "unmodelled_prefix_s": float(prefix_s),
        "channels": channels,
        "target_rate_hz": TARGET_RATE_HZ,
        "epoch_alignment": "complete 30-s epochs aligned backward from valid signal end",
        "features_recomputed_from_edf": required_feature_columns(payload),
    }


def load_single_record_epoch_csv(
    path: Path,
    payload: dict[str, Any],
    *,
    record_uid: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one record and validate all portable physical inputs."""
    frame = pd.read_csv(path, low_memory=False)
    if record_uid is not None:
        if "record_uid" not in frame:
            raise ValueError("--record-uid requires a record_uid column")
        frame = frame.loc[frame["record_uid"].astype(str).eq(str(record_uid))].copy()
        if frame.empty:
            raise ValueError(f"No rows matched record_uid={record_uid!r}")
    elif "record_uid" in frame and frame["record_uid"].astype(str).nunique() != 1:
        raise ValueError("The epoch CSV has multiple records; select one with --record-uid")
    if frame.empty:
        raise ValueError("The epoch CSV is empty")

    required = [*required_feature_columns(payload), "duration_s"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"The epoch CSV is missing required columns: {missing}")
    numeric = frame.loc[:, required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Required features and duration_s must be finite")
    if np.any(numeric["duration_s"].to_numpy(float) <= 0.0):
        raise ValueError("duration_s must be positive")
    if "epoch_index" in frame and frame["epoch_index"].duplicated().any():
        raise ValueError("epoch_index must be unique within the selected record")

    uid = str(frame["record_uid"].iloc[0]) if "record_uid" in frame else None
    return frame.reset_index(drop=True), {
        "source_kind": "epoch_feature_CSV",
        "epoch_features": str(path.resolve()),
        "record_uid": uid,
        "features_read_from_csv": required_feature_columns(payload),
    }


def predict_preferred_rstar(
    frame: pd.DataFrame, payload: dict[str, Any]
) -> pd.DataFrame:
    """Apply only physical feature columns and expose every mapping component."""
    required = required_feature_columns(payload)
    missing = [column for column in [*required, "duration_s"] if column not in frame]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    physical = frame.loc[:, required].copy()
    applied = apply_adjacent_repeat_v3(physical, payload)

    metadata = [column for column in OUTPUT_METADATA_COLUMNS if column in frame]
    result = frame.loc[:, list(dict.fromkeys([*metadata, *required]))].copy()
    for column in OUTPUT_COMPONENT_COLUMNS:
        result[column] = applied[column].to_numpy(float)
    duration = frame["duration_s"].to_numpy(float)
    result["pure_Fs_apparent_increment_s"] = (
        result["f_self_similarity"].to_numpy(float) * duration
    )
    result["preferred_f_apparent_increment_s"] = (
        result["f_repeat_micro_v3"].to_numpy(float) * duration
    )
    for column in ("f_self_similarity", "return_prominence_R_star", "f_repeat_micro_v3"):
        values = result[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError(f"Portable model produced invalid {column}")
    return result


def build_summary(
    predicted: pd.DataFrame,
    payload: dict[str, Any],
    source_metadata: dict[str, Any],
    model_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    duration = predicted["duration_s"].to_numpy(float)
    pure = predicted["f_self_similarity"].to_numpy(float)
    preferred = predicted["f_repeat_micro_v3"].to_numpy(float)
    physical_s = float(np.sum(duration))

    def summary(column: str) -> dict[str, float]:
        values = predicted[column].to_numpy(float)
        return {
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }

    return {
        **source_metadata,
        "model_json": str(model_json.resolve()),
        "model_format": payload["format"],
        "formula": "f=expit(logit(F_S)+0.25*(2R_star-1))",
        "n_complete_epochs": int(len(predicted)),
        "epoch_seconds": EPOCH_SECONDS,
        "modelled_physical_time_s": physical_s,
        "pure_Fs_apparent_time_s": float(np.sum(pure * duration)),
        "preferred_f_apparent_time_s": float(np.sum(preferred * duration)),
        "mean_preferred_f": (
            float(np.sum(preferred * duration) / physical_s) if physical_s > 0.0 else None
        ),
        "component_summaries": {
            column: summary(column)
            for column in (
                "f_self_similarity",
                "return_prominence_P_star",
                "return_prominence_R_star",
                "f_repeat_micro_v3",
            )
        },
        "output_epochs_csv": str(output_csv.resolve()),
        "scope": {
            "global_g_used": False,
            "dream_report_or_duration_used": False,
            "spectral_or_sleep_stage_classifier_used": False,
            "sleep_stage_required_for_prediction": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edf", type=Path)
    source.add_argument("--epoch-features", type=Path)
    parser.add_argument("--dataset", choices=("Zhang", "Kumral"))
    parser.add_argument("--metadata-duration-s", type=float)
    parser.add_argument(
        "--record-uid",
        help="Output UID in EDF mode, or selector if a CSV contains several records",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.model_json.read_text(encoding="utf-8"))
    validate_portable_payload(payload)
    if args.edf is not None:
        if args.dataset is None:
            parser.error("--dataset is required with --edf")
        metadata_duration_s = (
            float(args.metadata_duration_s)
            if args.metadata_duration_s is not None
            else _edf_header_duration_s(args.edf)
        )
        if not math.isfinite(metadata_duration_s) or metadata_duration_s <= 0.0:
            parser.error("--metadata-duration-s must be positive and finite")
        frame, source_metadata = extract_required_features_from_edf(
            args.edf,
            args.dataset,
            metadata_duration_s,
            payload,
            record_uid=args.record_uid,
        )
    else:
        if args.dataset is not None or args.metadata_duration_s is not None:
            parser.error("--dataset/--metadata-duration-s apply only to --edf")
        frame, source_metadata = load_single_record_epoch_csv(
            args.epoch_features, payload, record_uid=args.record_uid
        )

    predicted = predict_preferred_rstar(frame, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(args.output, index=False)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = build_summary(
        predicted, payload, source_metadata, args.model_json, args.output
    )
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
