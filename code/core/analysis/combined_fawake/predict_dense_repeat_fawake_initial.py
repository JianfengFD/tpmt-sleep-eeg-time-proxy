#!/usr/bin/env python3
"""Apply the portable physical dense-repeat model to one EEG record.

The command accepts either a Zhang/Kumral EDF or an epoch-feature CSV.  EDF
mode recomputes only the physical features selected by the portable model.
The robust default fixes one long/short scale pair and evaluates that pair at
the nine predeclared gamma values (18 S columns, without fitted weights), plus
one signed return-prominence P value.  Legacy single-S and one-gamma/two-S
portable models remain supported.

The mapping implemented here stops at the initial f_awake:

    raw Fs -> smooth Fs calibration
    raw P  -> smooth R calibration
    f_awake_initial = Fs + alpha Fs (1-Fs) (2R-1).

No elastic shell g, dream report/duration, spectral sleep-stage classifier,
or predicted sleep stage is read or evaluated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from analysis.combined_fawake.augment_adjacent_recurrence_grid import (
    adjacent_return_prominence,
)
from analysis.combined_fawake.dense_repeat_fawake_model import (
    FIXED_FS_CANDIDATES,
    FIXED_RETURN_PROMINENCE_COLUMNS,
    apply_dense_repeat_portable_model,
)
from analysis.combined_fawake.eeg_features import (
    EPOCH_SECONDS,
    TARGET_RATE_HZ,
    _robust_epoch,
    load_stageable_record,
    structure_scores,
)


STRUCTURE_COLUMN_PATTERN = re.compile(
    r"^structure_S_d(?P<delta>[0-9]+(?:p[0-9]+)?)_g(?P<gamma>[0-9]+(?:p[0-9]+)?)$"
)
PROMINENCE_COLUMN_PATTERN = re.compile(
    r"^recurrence_P_h(?P<context>[0-9]+(?:p[0-9]+)?)_tau(?P<tau>[0-9]+(?:p[0-9]+)?)$"
)
OUTPUT_METADATA_COLUMNS = (
    "record_uid",
    "dataset",
    "subject_id",
    "subject_group",
    "case_id",
    "filename",
    "split",
    "manual_final_stage",
    "experience",
    "epoch_index",
    "is_final_labelled_epoch",
    "valid_signal_end_s",
    "unmodelled_prefix_s",
    "start_s",
    "end_s",
    "duration_s",
)


def _number_from_column_token(token: str) -> float:
    return float(token.replace("p", "."))


def _validate_portable_physical_model(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str], str]:
    """Validate the locked model and return its selected physical columns."""
    if payload.get("format") != "dense_repeat_fawake_portable_v2":
        raise ValueError("Expected dense_repeat_fawake_portable_v2 model JSON")

    candidate_key = str(payload.get("selected_Fs_candidate", ""))
    if candidate_key not in FIXED_FS_CANDIDATES:
        raise ValueError("The selected Fs candidate is outside the declared fixed grid")
    expected_spec = FIXED_FS_CANDIDATES[candidate_key]
    supplied_spec = payload.get("selected_Fs_candidate_spec", expected_spec)
    for field, expected_value in expected_spec.items():
        if supplied_spec.get(field) != expected_value:
            raise ValueError(f"The model changed fixed Fs field {field!r}")

    if expected_spec["kind"] == "single_S":
        s_columns = [str(expected_spec["column"])]
    elif expected_spec["kind"] == "persistence_ratio":
        s_columns = [
            str(expected_spec["long_column"]),
            str(expected_spec["short_column"]),
        ]
    elif expected_spec["kind"] == "gamma_median_persistence_ratio":
        s_columns = list(
            dict.fromkeys(
                str(pair[column])
                for pair in expected_spec["gamma_pairs"]
                for column in ("long_column", "short_column")
            )
        )
    else:  # pragma: no cover - FIXED_FS_CANDIDATES controls this branch.
        raise ValueError(f"Unsupported Fs candidate kind: {expected_spec['kind']}")

    if payload.get("selected_repeat_mode") != "return_prominence":
        raise ValueError(
            "This initial-f_awake EDF predictor supports the physical "
            "return_prominence mode only"
        )
    repeat_column = str(payload.get("selected_repeat_column", ""))
    if repeat_column not in FIXED_RETURN_PROMINENCE_COLUMNS:
        raise ValueError("The selected return-prominence column is outside the fixed grid")

    # The model implementation performs a second independent validation when
    # applying the mapping.  Returning the canonical rather than supplied spec
    # prevents altered JSON formula text from affecting extraction.
    return copy.deepcopy(expected_spec), s_columns, repeat_column


def selected_physical_feature_columns(payload: dict[str, Any]) -> list[str]:
    """Return the exact S/P columns needed to evaluate ``payload``."""
    _, s_columns, repeat_column = _validate_portable_physical_model(payload)
    return [*s_columns, repeat_column]


def extract_selected_epoch_features(
    epoch_uv: np.ndarray,
    payload: dict[str, Any],
) -> dict[str, float]:
    """Recompute exactly the selected S column(s) and return prominence P."""
    _, s_columns, repeat_column = _validate_portable_physical_model(payload)
    epoch_z, _ = _robust_epoch(np.asarray(epoch_uv))

    by_delta: dict[float, list[tuple[str, float]]] = {}
    for column in s_columns:
        match = STRUCTURE_COLUMN_PATTERN.fullmatch(column)
        if match is None:
            raise ValueError(f"Cannot parse selected S column {column!r}")
        delta_s = _number_from_column_token(match.group("delta"))
        gamma = _number_from_column_token(match.group("gamma"))
        by_delta.setdefault(delta_s, []).append((column, gamma))

    calculated_s: dict[str, float] = {}
    for delta_s, requests in by_delta.items():
        scores = structure_scores(epoch_z, delta_s, [gamma for _, gamma in requests])
        for column, gamma in requests:
            calculated_s[column] = float(scores[gamma])

    repeat_match = PROMINENCE_COLUMN_PATTERN.fullmatch(repeat_column)
    if repeat_match is None:
        raise ValueError(f"Cannot parse selected recurrence column {repeat_column!r}")
    context_s = _number_from_column_token(repeat_match.group("context"))
    tau_s = _number_from_column_token(repeat_match.group("tau"))
    prominence, _, _ = adjacent_return_prominence(epoch_z, context_s, tau_s)
    features = {column: calculated_s[column] for column in s_columns}
    features[repeat_column] = float(prominence)
    return features


def _edf_header_duration_s(path: Path) -> float:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    return float(raw.n_times / float(raw.info["sfreq"]))


def extract_selected_features_from_edf(
    path: Path,
    dataset: str,
    metadata_duration_s: float,
    payload: dict[str, Any],
    *,
    record_uid: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preprocess one EDF and recompute the locked features for each 30-s epoch."""
    data, valid_end_s, epochs, prefix_s, channels = load_stageable_record(
        path, dataset, metadata_duration_s
    )
    uid = record_uid or f"prediction:{dataset}:{path.name}"
    rows: list[dict[str, Any]] = []
    for epoch_index, (start_s, end_s) in enumerate(epochs):
        start = int(round(start_s * TARGET_RATE_HZ))
        stop = int(round(end_s * TARGET_RATE_HZ))
        features = extract_selected_epoch_features(data[:, start:stop], payload)
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
                **features,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The EDF contains no complete reverse-aligned 30-s epoch")
    metadata = {
        "source_kind": "EDF",
        "edf": str(path.resolve()),
        "dataset": dataset,
        "metadata_duration_s": float(metadata_duration_s),
        "valid_signal_end_s": float(valid_end_s),
        "unmodelled_prefix_s": float(prefix_s),
        "channels": channels,
        "target_rate_hz": TARGET_RATE_HZ,
        "epoch_alignment": "complete 30-s epochs aligned backward from valid signal end",
        "selected_features_recomputed_from_edf": selected_physical_feature_columns(payload),
    }
    return frame, metadata


def load_single_record_epoch_csv(
    path: Path,
    payload: dict[str, Any],
    *,
    record_uid: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one record from an existing epoch table without reading labels."""
    frame = pd.read_csv(path, low_memory=False)
    if record_uid is not None:
        if "record_uid" not in frame:
            raise ValueError("--record-uid requires a record_uid column")
        frame = frame.loc[frame["record_uid"].astype(str).eq(str(record_uid))].copy()
        if frame.empty:
            raise ValueError(f"No rows matched record_uid={record_uid!r}")
    elif "record_uid" in frame and frame["record_uid"].astype(str).nunique() != 1:
        raise ValueError(
            "The epoch CSV contains multiple records; select one with --record-uid"
        )
    if frame.empty:
        raise ValueError("The epoch CSV is empty")

    required_features = selected_physical_feature_columns(payload)
    missing = [column for column in [*required_features, "duration_s"] if column not in frame]
    if missing:
        raise ValueError(f"The epoch CSV is missing required columns: {missing}")
    selected_values = frame[required_features + ["duration_s"]].to_numpy(float)
    if not np.isfinite(selected_values).all():
        raise ValueError("The selected physical features/duration contain non-finite values")
    if np.any(frame["duration_s"].to_numpy(float) <= 0.0):
        raise ValueError("duration_s must be positive")
    if "epoch_index" in frame and frame["epoch_index"].duplicated().any():
        raise ValueError("epoch_index must be unique within the selected record")

    uid = None
    if "record_uid" in frame:
        uid = str(frame["record_uid"].iloc[0])
    metadata = {
        "source_kind": "epoch_feature_CSV",
        "epoch_features": str(path.resolve()),
        "record_uid": uid,
        "selected_features_read_from_csv": required_features,
    }
    return frame.reset_index(drop=True), metadata


def predict_initial_fawake(
    frame: pd.DataFrame,
    payload: dict[str, Any],
) -> pd.DataFrame:
    """Apply only the selected physical columns and expose all mapping stages."""
    required_features = selected_physical_feature_columns(payload)
    missing = [column for column in [*required_features, "duration_s"] if column not in frame]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    # Deliberately pass only the selected physical columns to the model.  Thus
    # dream fields, stage labels, spectral features, and any later g parameters
    # in an input CSV cannot influence this prediction path.
    physical = frame.loc[:, required_features].copy()
    applied = apply_dense_repeat_portable_model(physical, payload)

    metadata_columns = [column for column in OUTPUT_METADATA_COLUMNS if column in frame]
    result = frame.loc[:, metadata_columns + required_features].copy()
    result["Fs_raw"] = applied["self_similarity_Fs_raw"].to_numpy(float)
    result["Fs"] = applied["f_self_similarity"].to_numpy(float)
    result["return_prominence_P_raw"] = applied["repeat_feature_raw"].to_numpy(float)
    result["R"] = applied["adjacent_repeat_R"].to_numpy(float)
    result["f_awake_initial"] = applied["f_awake_initial"].to_numpy(float)
    result["apparent_time_increment_s"] = (
        result["f_awake_initial"].to_numpy(float)
        * frame["duration_s"].to_numpy(float)
    )
    for column in ("Fs", "R", "f_awake_initial"):
        values = result[column].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise RuntimeError(f"Portable model produced invalid {column} values")
    return result


def build_summary(
    predicted: pd.DataFrame,
    payload: dict[str, Any],
    source_metadata: dict[str, Any],
    model_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    """Build a JSON-safe audit and initial apparent-time integral."""
    duration = predicted["duration_s"].to_numpy(float)
    initial = predicted["f_awake_initial"].to_numpy(float)
    physical_time_s = float(np.sum(duration))
    apparent_time_s = float(np.sum(duration * initial))

    def component_summary(column: str) -> dict[str, float]:
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
        "selected_Fs_candidate": payload["selected_Fs_candidate"],
        "selected_Fs_candidate_spec": payload["selected_Fs_candidate_spec"],
        "selected_return_prominence_column": payload["selected_repeat_column"],
        "alpha": float(payload["alpha"]),
        "n_complete_epochs": int(len(predicted)),
        "epoch_seconds": EPOCH_SECONDS,
        "modelled_physical_time_s": physical_time_s,
        "initial_apparent_time_s": apparent_time_s,
        "mean_f_awake_initial": (
            apparent_time_s / physical_time_s if physical_time_s > 0.0 else None
        ),
        "component_summaries": {
            column: component_summary(column)
            for column in ("Fs_raw", "Fs", "return_prominence_P_raw", "R", "f_awake_initial")
        },
        "output_epochs_csv": str(output_csv.resolve()),
        "scope": {
            "elastic_shell_g_used": False,
            "dream_report_or_duration_used": False,
            "spectral_or_sleep_stage_classifier_used": False,
            "definition": "f_awake_initial=Fs+alpha*Fs*(1-Fs)*(2R-1)",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the self-similarity-first portable dense-repeat model to "
            "one EDF or one record from an epoch-feature CSV."
        )
    )
    parser.add_argument("--model-json", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edf", type=Path)
    source.add_argument("--epoch-features", type=Path)
    parser.add_argument("--dataset", choices=("Zhang", "Kumral"))
    parser.add_argument(
        "--metadata-duration-s",
        type=float,
        help=(
            "Record duration from dataset metadata. If omitted in EDF mode, "
            "the EDF header duration is used."
        ),
    )
    parser.add_argument(
        "--record-uid",
        help="Optional output UID, or selector when the epoch CSV has several records",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary",
        type=Path,
        help="JSON summary path (default: OUTPUT with .summary.json suffix)",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.model_json.read_text(encoding="utf-8"))
    _validate_portable_physical_model(payload)
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
        frame, source_metadata = extract_selected_features_from_edf(
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
            args.epoch_features,
            payload,
            record_uid=args.record_uid,
        )

    predicted = predict_initial_fawake(frame, payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(args.output, index=False)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        predicted, payload, source_metadata, args.model_json, args.output
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
