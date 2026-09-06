#!/usr/bin/env python3
"""Fit and audit the self-similarity-first, short-repeat-corrected f_awake.

This runner intentionally stops before the dream-duration shell ``g``.  It
uses only manually staged final epochs from the declared training subjects to
select the initial mapping.  Its training estimate is a genuinely nested,
participant-grouped out-of-fold prediction: the complete physical model,
including S/repetition parameter selection and calibration, is refitted inside
every outer fold.  The declared test subjects are reported only as a *repeated
held-aside diagnostic*, because they have already been inspected in earlier
iterations of this project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent.parent
CACHE_ROOT = SCRIPT_ROOT / ".cache_dense_repeat_fit"
for _directory in (CACHE_ROOT, CACHE_ROOT / "matplotlib"):
    _directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from analysis.combined_fawake.dense_repeat_fawake_model import (  # noqa: E402
    FIXED_ADJACENT_REPEAT_COLUMNS,
    FIXED_RETURN_PROMINENCE_COLUMNS,
    FIXED_SELF_SIMILARITY_COLUMNS,
    apply_dense_repeat_model,
    export_dense_repeat_model,
    fit_dense_repeat_model,
)


RANDOM_STATE = 20260904
STAGE_ORDER = ("W", "N1", "N2", "N3", "REM")
STAGE_COLORS = {
    "W": "#d73027",
    "N1": "#fdae61",
    "N2": "#74add1",
    "N3": "#313695",
    "REM": "#8c6bb1",
}
COMPONENTS = (
    ("f_self_similarity", r"pure $F_S$"),
    ("f_awake_initial", r"corrected $f_{\mathrm{awake}}$"),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.casefold()
    allowed = {"true", "false", "1", "0"}
    _require(set(lowered.unique()) <= allowed, "is_final_labelled_epoch is not Boolean")
    return lowered.isin({"true", "1"})


def load_and_assert_feature_table(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "record_uid",
        "dataset",
        "subject_group",
        "split",
        "manual_final_stage",
        "epoch_index",
        "is_final_labelled_epoch",
        "start_s",
        "end_s",
        "duration_s",
    }
    missing = sorted(required - set(frame.columns))
    _require(not missing, f"feature table is missing required columns: {missing}")
    _require(len(frame) > 0, "feature table is empty")
    _require(set(frame["split"].astype(str).unique()) == {"train", "test"},
             "split must contain exactly train and test")
    _require(not frame[["record_uid", "subject_group", "split"]].isna().any().any(),
             "record/group/split identifiers must be non-null")
    _require(not frame.duplicated(["record_uid", "epoch_index"]).any(),
             "record_uid + epoch_index must be unique")
    _require(np.isfinite(frame["duration_s"].to_numpy(float)).all(),
             "duration_s contains non-finite values")
    _require((frame["duration_s"].to_numpy(float) > 0.0).all(),
             "duration_s must be positive")
    frame["is_final_labelled_epoch"] = _as_bool(frame["is_final_labelled_epoch"])

    group_split_counts = frame.groupby("subject_group")["split"].nunique()
    record_split_counts = frame.groupby("record_uid")["split"].nunique()
    _require(int(group_split_counts.max()) == 1,
             "subject_group leakage: at least one subject appears in train and test")
    _require(int(record_split_counts.max()) == 1,
             "record leakage: at least one record appears in train and test")
    train_groups = set(frame.loc[frame["split"].eq("train"), "subject_group"].astype(str))
    test_groups = set(frame.loc[frame["split"].eq("test"), "subject_group"].astype(str))
    _require(train_groups.isdisjoint(test_groups), "train/test subject groups overlap")

    final = frame.loc[frame["is_final_labelled_epoch"]].copy()
    _require(len(final) > 0, "no manually labelled final epochs were found")
    _require(set(final["manual_final_stage"].astype(str)) <= set(STAGE_ORDER),
             "final labelled rows contain an unknown sleep stage")
    train_final = final.loc[final["split"].eq("train")]
    test_final = final.loc[final["split"].eq("test")]
    for stage in ("W", "N1", "N3"):
        _require((train_final["manual_final_stage"].astype(str) == stage).any(),
                 f"training final epochs contain no {stage}")
        _require((test_final["manual_final_stage"].astype(str) == stage).any(),
                 f"diagnostic test final epochs contain no {stage}")
    _require(set(train_final.loc[train_final["manual_final_stage"].eq("W"), "dataset"])
             == {"Zhang"}, "explicit W training labels must come from Zhang")

    available_s = [column for column in FIXED_SELF_SIMILARITY_COLUMNS if column in frame]
    available_r = [column for column in FIXED_ADJACENT_REPEAT_COLUMNS if column in frame]
    available_p = [column for column in FIXED_RETURN_PROMINENCE_COLUMNS if column in frame]
    _require(bool(available_s), "no declared S grid columns are present")
    _require(bool(available_p), "no declared return-prominence grid columns are present")
    for column in available_s + available_r:
        values = frame[column].to_numpy(float)
        _require(np.isfinite(values).all(), f"{column} contains non-finite values")
        _require(bool(np.all((values >= -1.0e-12) & (values <= 1.0 + 1.0e-12))),
                 f"{column} is not confined to [0,1]")
    for column in available_p:
        values = frame[column].to_numpy(float)
        _require(np.isfinite(values).all(), f"{column} contains non-finite values")
        _require(bool(np.all((values >= -1.0 - 1.0e-12) & (values <= 1.0 + 1.0e-12))),
                 f"{column} is not confined to [-1,1]")

    audit = {
        "input": str(path.resolve()),
        "rows": int(len(frame)),
        "records": int(frame["record_uid"].nunique()),
        "subjects": int(frame["subject_group"].nunique()),
        "train_rows": int(frame["split"].eq("train").sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "train_final_labelled_epochs": int(len(train_final)),
        "test_final_labelled_epochs": int(len(test_final)),
        "train_subjects": int(len(train_groups)),
        "test_subjects": int(len(test_groups)),
        "subject_disjoint": True,
        "record_disjoint": True,
        "available_S_columns": len(available_s),
        "available_adjacent_repeat_columns": len(available_r),
        "available_return_prominence_columns": len(available_p),
        "train_final_stage_counts": {
            str(key): int(value)
            for key, value in train_final["manual_final_stage"].value_counts().items()
        },
        "test_final_stage_counts": {
            str(key): int(value)
            for key, value in test_final["manual_final_stage"].value_counts().items()
        },
    }
    return frame, audit


def fitting_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only stage/group metadata and the declared physical feature grids."""
    metadata = ["record_uid", "dataset", "subject_group", "manual_final_stage"]
    features = [
        column
        for column in (
            *FIXED_SELF_SIMILARITY_COLUMNS,
            *FIXED_RETURN_PROMINENCE_COLUMNS,
            *FIXED_ADJACENT_REPEAT_COLUMNS,
        )
        if column in frame
    ]
    return frame.loc[:, metadata + features].copy()


def _outer_splits(frame: pd.DataFrame, requested_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    stages = frame["manual_final_stage"].astype(str).to_numpy()
    groups = frame["subject_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    stage_group_counts = [len(np.unique(groups[stages == stage])) for stage in ("W", "N1", "N3")]
    n_splits = min(int(requested_splits), len(unique_groups), min(stage_group_counts))
    _require(n_splits >= 2, "not enough independent subject groups for outer grouped CV")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(frame, stages, groups))
    seen: list[int] = []
    for fit_index, validation_index in splits:
        fit_groups = set(groups[fit_index])
        validation_groups = set(groups[validation_index])
        _require(fit_groups.isdisjoint(validation_groups), "outer-fold subject leakage")
        seen.extend(validation_index.tolist())
    _require(sorted(seen) == list(range(len(frame))), "outer OOF coverage is not exactly once")
    return splits


def _model_repeat_column(model: dict[str, Any]) -> str:
    for key in (
        "selected_adjacent_repeat_column",
        "selected_repeat_column",
        "selected_recurrence_column",
    ):
        if key in model:
            return str(model[key])
    raise KeyError("fitted model does not expose its selected repeat column")


def _selection_row(
    model: dict[str, Any],
    *,
    fold: str | int,
    fit_rows: int,
    validation_rows: int,
    fit_groups: Iterable[str],
    validation_groups: Iterable[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "outer_fold": fold,
        "fit_epochs": int(fit_rows),
        "validation_epochs": int(validation_rows),
        "fit_subjects": int(len(set(fit_groups))),
        "validation_subjects": int(len(set(validation_groups))),
        "validation_groups": "|".join(sorted(set(validation_groups))),
        "selected_Fs_candidate": str(model.get("selected_Fs_candidate", "")),
        "selected_S_column": str(model["selected_S_column"]),
        "selected_Fs_kind": str(model.get("selected_Fs_candidate_spec", {}).get("kind", "")),
        "selected_repeat_column": _model_repeat_column(model),
        "selected_repeat_mode": str(model.get("selected_repeat_mode", "")),
        "alpha": float(model.get("alpha", np.nan)),
        "n3_anchor": float(model["n3_anchor"]),
        "wake_anchor": float(model["wake_anchor"]),
    }
    fs_spec = model.get("selected_Fs_candidate_spec", {})
    for key in ("gamma", "long_delta_s", "short_delta_s", "ratio_epsilon"):
        if key in fs_spec:
            row[f"Fs_{key}"] = float(fs_spec[key])
    repeat_calibration = model.get("repeat_calibration", {})
    row["repeat_calibration_kind"] = str(repeat_calibration.get("kind", ""))
    for key in ("n3_anchor", "wake_anchor", "endpoint_epsilon"):
        if key in repeat_calibration:
            row[f"repeat_{key}"] = float(repeat_calibration[key])
    for prefix, metrics_key in (
        ("inner_selected_Fs", "selected_Fs_metrics_oof"),
        ("inner_selected_corrected", "selected_dense_repeat_metrics_oof"),
    ):
        metrics = model.get(metrics_key, {})
        for key in ("auc_Zhang_W_vs_N1", "auc_W_vs_N2", "auc_W_vs_N3", "auc_W_vs_REM"):
            if metrics.get(key) is not None:
                row[f"{prefix}_{key}"] = float(metrics[key])
    selected_repeat_metrics = model.get("selected_dense_repeat_metrics_oof", {})
    for key in (
        "observed_mean_absolute_correction_oof",
        "observed_maximum_absolute_correction_oof",
        "regularized_selection_objective",
    ):
        if selected_repeat_metrics.get(key) is not None:
            row[f"inner_{key}"] = float(selected_repeat_metrics[key])
    # Preserve any scalar calibration values introduced by a newer model API.
    for key, value in model.items():
        lowered = key.casefold()
        if key in row or not isinstance(value, (int, float, np.integer, np.floating)):
            continue
        if any(token in lowered for token in ("anchor", "center", "scale", "slope", "temperature")):
            row[f"model_{key}"] = float(value)
    return row


def nested_grouped_oof(
    train_final: pd.DataFrame,
    *,
    outer_splits: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    compact = fitting_view(train_final.reset_index(drop=True))
    splits = _outer_splits(compact, outer_splits)
    groups = compact["subject_group"].astype(str).to_numpy()
    prediction_parts: list[pd.DataFrame] = []
    stability_rows: list[dict[str, Any]] = []
    for fold_index, (fit_index, validation_index) in enumerate(splits, start=1):
        fit = compact.iloc[fit_index].reset_index(drop=True)
        validation = compact.iloc[validation_index].copy()
        model = fit_dense_repeat_model(
            fit,
            n_splits=inner_splits,
            random_state=RANDOM_STATE + 1009 * fold_index,
        )
        predicted = apply_dense_repeat_model(validation, model)
        predicted["outer_fold"] = fold_index
        predicted["prediction_protocol"] = "nested_subject_group_oof"
        prediction_parts.append(predicted)
        stability_rows.append(
            _selection_row(
                model,
                fold=fold_index,
                fit_rows=len(fit_index),
                validation_rows=len(validation_index),
                fit_groups=groups[fit_index],
                validation_groups=groups[validation_index],
            )
        )
        print(
            f"outer fold {fold_index}/{len(splits)}: "
            f"Fs={model.get('selected_Fs_candidate', model.get('selected_S_column'))}; "
            f"R={_model_repeat_column(model)}; "
            f"alpha={float(model.get('alpha', float('nan'))):.4g}",
            flush=True,
        )
    oof = pd.concat(prediction_parts).sort_index()
    _require(len(oof) == len(compact), "nested OOF prediction count mismatch")
    _require(not oof.index.duplicated().any(), "nested OOF contains duplicate validation rows")
    _require(np.isfinite(oof["f_self_similarity"].to_numpy(float)).all(),
             "nested OOF pure Fs is incomplete")
    _require(np.isfinite(oof["f_awake_initial"].to_numpy(float)).all(),
             "nested OOF corrected f is incomplete")
    return oof.reset_index(drop=True), pd.DataFrame(stability_rows)


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0 or len(np.unique(labels[finite])) != 2:
        return None
    return float(roc_auc_score(labels[finite], values[finite]))


def pairwise_auc_rows(frame: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    zhang = frame.loc[frame["dataset"].astype(str).eq("Zhang")].copy()
    rows: list[dict[str, Any]] = []
    comparisons = [
        (negative, zhang.loc[zhang["manual_final_stage"].isin(["W", negative])])
        for negative in ("N1", "N2", "N3", "REM")
    ]
    comparisons.append(
        (
            "all non-W sleep",
            zhang.loc[zhang["manual_final_stage"].isin(STAGE_ORDER)],
        )
    )
    for negative, pair in comparisons:
        labels = pair["manual_final_stage"].eq("W").astype(int).to_numpy()
        for component, label in COMPONENTS:
            rows.append(
                {
                    "evaluation": evaluation,
                    "dataset_scope": "Zhang only",
                    "positive_stage": "W",
                    "negative_stage": negative,
                    "component": component,
                    "component_label": label.replace("$", ""),
                    "n_positive_epochs": int(np.sum(labels == 1)),
                    "n_negative_epochs": int(np.sum(labels == 0)),
                    "auc": _safe_auc(labels, pair[component].to_numpy(float)),
                }
            )
    return rows


def stage_distribution_rows(frame: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_scope, subset in (
        ("pooled", frame),
        ("Zhang", frame.loc[frame["dataset"].eq("Zhang")]),
        ("Kumral", frame.loc[frame["dataset"].eq("Kumral")]),
    ):
        for stage in STAGE_ORDER:
            stage_frame = subset.loc[subset["manual_final_stage"].eq(stage)]
            for component, _label in COMPONENTS:
                values = stage_frame[component].to_numpy(float)
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "evaluation": evaluation,
                        "dataset_scope": dataset_scope,
                        "stage": stage,
                        "component": component,
                        "n_epochs": int(len(values)),
                        "mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        "q10": float(np.quantile(values, 0.10)),
                        "median": float(np.median(values)),
                        "q90": float(np.quantile(values, 0.90)),
                        "fraction_above_0p90": float(np.mean(values > 0.90)),
                    }
                )
    return rows


def correction_magnitude_row(frame: pd.DataFrame, evaluation: str) -> dict[str, Any]:
    correction = (
        frame["f_awake_initial"].to_numpy(float)
        - frame["f_self_similarity"].to_numpy(float)
    )
    absolute = np.abs(correction)
    return {
        "evaluation": evaluation,
        "n_epochs": int(len(frame)),
        "mean_signed_correction": float(np.mean(correction)),
        "mean_absolute_correction": float(np.mean(absolute)),
        "median_absolute_correction": float(np.median(absolute)),
        "q90_absolute_correction": float(np.quantile(absolute, 0.90)),
        "maximum_absolute_correction": float(np.max(absolute)),
        "fraction_increased": float(np.mean(correction > 1.0e-12)),
        "fraction_decreased": float(np.mean(correction < -1.0e-12)),
        "all_corrected_values_in_unit_interval": bool(
            np.all(
                (frame["f_awake_initial"].to_numpy(float) >= -1.0e-12)
                & (frame["f_awake_initial"].to_numpy(float) <= 1.0 + 1.0e-12)
            )
        ),
    }


def clustered_wn1_bootstrap(
    frame: pd.DataFrame,
    evaluation: str,
    *,
    replicates: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair = frame.loc[
        frame["dataset"].eq("Zhang") & frame["manual_final_stage"].isin(["W", "N1"])
    ].reset_index(drop=True)
    groups = sorted(pair["subject_group"].astype(str).unique())
    _require(len(groups) >= 2, f"{evaluation}: fewer than two Zhang subject groups")
    group_indices = {
        group: pair.index[pair["subject_group"].astype(str).eq(group)].to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(RANDOM_STATE + (0 if "training" in evaluation else 1))
    rows: list[dict[str, Any]] = []
    attempts = 0
    maximum_attempts = max(100, replicates * 20)
    while len(rows) < replicates and attempts < maximum_attempts:
        attempts += 1
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([group_indices[str(group)] for group in sampled_groups])
        sample = pair.iloc[index]
        labels = sample["manual_final_stage"].eq("W").astype(int).to_numpy()
        if len(np.unique(labels)) != 2:
            continue
        pure = float(roc_auc_score(labels, sample["f_self_similarity"].to_numpy(float)))
        corrected = float(roc_auc_score(labels, sample["f_awake_initial"].to_numpy(float)))
        rows.append(
            {
                "evaluation": evaluation,
                "bootstrap_replicate": len(rows) + 1,
                "auc_pure_Fs": pure,
                "auc_corrected_fawake": corrected,
                "delta_auc_corrected_minus_pure": corrected - pure,
            }
        )
    _require(len(rows) == replicates, f"{evaluation}: insufficient valid bootstrap replicates")
    result = pd.DataFrame(rows)
    delta = result["delta_auc_corrected_minus_pure"].to_numpy(float)
    summary = {
        "evaluation": evaluation,
        "resampling_unit": "Zhang subject_group",
        "replicates": int(replicates),
        "mean_delta_auc_corrected_minus_pure": float(np.mean(delta)),
        "median_delta_auc_corrected_minus_pure": float(np.median(delta)),
        "percentile_95_CI": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "fraction_delta_above_zero": float(np.mean(delta > 0.0)),
    }
    return result, summary


def plot_stage_distributions(
    training_oof: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True, sharey=True)
    evaluations = (
        (training_oof, "Training: nested subject-group OOF"),
        (diagnostic_test, "Repeated held-aside diagnostic test"),
    )
    positions = np.arange(len(STAGE_ORDER), dtype=float)
    offsets = (-0.16, 0.16)
    colors = ("#4c78a8", "#e45756")
    for axis, (frame, title) in zip(axes, evaluations):
        # W endpoints exist only in Zhang.  Keep the primary visual comparison
        # within that dataset; the numeric CSV still exposes pooled, Zhang,
        # and Kumral distributions separately for audit.
        frame = frame.loc[frame["dataset"].astype(str).eq("Zhang")]
        for component_index, ((component, label), offset, color) in enumerate(
            zip(COMPONENTS, offsets, colors)
        ):
            values = [
                frame.loc[frame["manual_final_stage"].eq(stage), component].to_numpy(float)
                for stage in STAGE_ORDER
            ]
            artists = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.27,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                medianprops={"color": "white", "linewidth": 1.5},
            )
            for patch in artists["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.78)
            axis.plot([], [], color=color, linewidth=8, alpha=0.78, label=label)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("physical component value")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xticks(positions, STAGE_ORDER)
    axes[-1].set_xlabel("manual final-epoch stage")
    fig.suptitle(
        "Zhang-only stage distributions before and after the bounded repeat correction",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wn1_roc(
    training_oof: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    evaluations = (
        (training_oof, "Training: nested subject-group OOF"),
        (diagnostic_test, "Repeated held-aside diagnostic test"),
    )
    colors = ("#4c78a8", "#e45756")
    for axis, (frame, title) in zip(axes, evaluations):
        pair = frame.loc[
            frame["dataset"].eq("Zhang") & frame["manual_final_stage"].isin(["W", "N1"])
        ]
        labels = pair["manual_final_stage"].eq("W").astype(int).to_numpy()
        for (component, label), color in zip(COMPONENTS, colors):
            values = pair[component].to_numpy(float)
            false_positive, true_positive, _ = roc_curve(labels, values)
            auc = roc_auc_score(labels, values)
            axis.plot(false_positive, true_positive, color=color, linewidth=2.2,
                      label=f"{label}: AUC={auc:.3f}")
        axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
        axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        axis.set_xlabel("false-positive rate (N1 called W)")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("true-positive rate (W called W)")
    fig.suptitle("Zhang-only W versus N1: pure self-similarity and corrected f_awake", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1, dtype=float) / len(values)


def plot_wn1_ecdf(
    training_oof: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    evaluations = (
        (training_oof, "Training: nested subject-group OOF"),
        (diagnostic_test, "Repeated held-aside diagnostic test"),
    )
    for axis, (frame, title) in zip(axes, evaluations):
        subset = frame.loc[frame["dataset"].eq("Zhang")]
        for stage in ("W", "N1"):
            for component, label in COMPONENTS:
                values = subset.loc[subset["manual_final_stage"].eq(stage), component].to_numpy(float)
                x, y = _ecdf(values)
                axis.step(
                    x,
                    y,
                    where="post",
                    color=STAGE_COLORS[stage],
                    linestyle="--" if component == "f_self_similarity" else "-",
                    linewidth=1.8,
                    label=f"{stage}, {label}",
                )
        axis.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
        axis.set_xlabel("physical component value")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower right", frameon=False, fontsize=8.5)
    axes[0].set_ylabel("empirical cumulative probability")
    fig.suptitle("Zhang-only W/N1 distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _short_feature_name(name: str) -> str:
    name = str(name)
    ratio = re.fullmatch(
        r"persistence_ratio__long_d([0-9p]+)__short_d([0-9p]+)__g([0-9p]+)", name
    )
    if ratio:
        values = [part.replace("p", ".") for part in ratio.groups()]
        return rf"ratio: $\Delta_L$={values[0]}, $\Delta_S$={values[1]}, $\gamma$={values[2]}"
    prominence = re.fullmatch(r"recurrence_P_h([0-9p]+)_tau([0-9p]+)", name)
    if prominence:
        values = [part.replace("p", ".") for part in prominence.groups()]
        return rf"P: H={values[0]} s, $\tau$={values[1]} s"
    name = re.sub(r"^single__", "single S: ", name)
    name = re.sub(r"^structure_S_", "S: ", name)
    name = re.sub(r"^adjacent_repeat_", "R: ", name)
    name = re.sub(r"^recurrence_", "P: ", name)
    return name


def plot_selection_stability(stability: pd.DataFrame, output: Path) -> None:
    outer = stability.loc[stability["outer_fold"].astype(str).ne("final_full_train")].copy()
    s_counts = Counter(outer["selected_Fs_candidate"].astype(str))
    r_counts = Counter(outer["selected_repeat_column"].astype(str))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    for axis, counts, title, color in (
        (axes[0], s_counts, "Selected self-similarity-derived Fs", "#4c78a8"),
        (axes[1], r_counts, "Selected short-repeat statistic", "#59a14f"),
    ):
        labels, values = zip(*sorted(counts.items(), key=lambda item: (item[1], item[0])))
        y = np.arange(len(labels))
        axis.barh(y, values, color=color, alpha=0.82)
        axis.set_yticks(y, [_short_feature_name(label) for label in labels], fontsize=7.5)
        axis.set_xlabel("number of outer folds")
        axis.set_title(title, fontsize=10.5, fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
    fold_numeric = pd.to_numeric(outer["outer_fold"])
    axes[2].plot(fold_numeric, outer["alpha"].to_numpy(float), "o-", color="#e45756")
    axes[2].set_xticks(fold_numeric)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("outer fold")
    axes[2].set_ylabel(r"repeat-correction strength $\alpha$")
    axes[2].set_title("Correction strength", fontsize=10.5, fontweight="bold")
    axes[2].grid(alpha=0.2)
    fig.suptitle("Nested-selection stability across held-out subject groups", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metric_lookup(
    auc_frame: pd.DataFrame, evaluation: str, negative: str, component: str
) -> float:
    row = auc_frame.loc[
        auc_frame["evaluation"].eq(evaluation)
        & auc_frame["negative_stage"].eq(negative)
        & auc_frame["component"].eq(component),
        "auc",
    ]
    return float(row.iloc[0])


def build_methods_report(
    audit: dict[str, Any],
    final_model: dict[str, Any],
    auc_frame: pd.DataFrame,
    bootstrap_summaries: list[dict[str, Any]],
    stability: pd.DataFrame,
    correction_frame: pd.DataFrame,
) -> str:
    train_eval = "training_nested_subject_group_oof"
    test_eval = "repeated_held_aside_diagnostic_test"
    train_boot = next(item for item in bootstrap_summaries if item["evaluation"] == train_eval)
    test_boot = next(item for item in bootstrap_summaries if item["evaluation"] == test_eval)
    train_ci = train_boot["percentile_95_CI"]
    test_ci = test_boot["percentile_95_CI"]
    selected_s_counts = Counter(stability.iloc[:-1]["selected_Fs_candidate"].astype(str))
    selected_r_counts = Counter(stability.iloc[:-1]["selected_repeat_column"].astype(str))
    train_correction = correction_frame.loc[
        correction_frame["evaluation"].eq(train_eval)
    ].iloc[0]
    test_correction = correction_frame.loc[
        correction_frame["evaluation"].eq(test_eval)
    ].iloc[0]

    def auc_line(evaluation: str, negative: str) -> str:
        pure = _metric_lookup(auc_frame, evaluation, negative, "f_self_similarity")
        corrected = _metric_lookup(auc_frame, evaluation, negative, "f_awake_initial")
        return f"| W vs {negative} | {pure:.4f} | {corrected:.4f} | {corrected-pure:+.4f} |"

    lines = [
        "# METHODS AND RESULTS: SELF-SIMILARITY-FIRST INITIAL f_awake",
        "",
        "## Scope and hard constraints",
        "",
        "This analysis fits only the initial EEG-to-f_awake mapping. It does not read dream-report duration, does not fit g, and does not estimate apparent dream time. The main component is constructed exclusively from the local approximate self-similarity statistic S(t; Delta, gamma): for one fixed long/short Delta pair, it takes the median of the persistence ratio over all nine predeclared gamma thresholds. A single short-lag adjacent-repeat statistic is allowed only as the bounded correction implemented by `dense_repeat_fawake_model.py`; no spectral or arbitrary stage-classifier feature enters the mapping.",
        "",
        f"Input contained {audit['rows']:,} epochs from {audit['records']:,} records and {audit['subjects']} subject groups. Model selection used only {audit['train_final_labelled_epochs']} manually staged final epochs in the declared training split. The {audit['test_final_labelled_epochs']} labelled test epochs were held out from every fit. Train and test sets are disjoint by both `subject_group` and `record_uid`.",
        "",
        "## Nested validation protocol",
        "",
        "The training estimate is genuinely nested at the fitted-model level. The outer StratifiedGroupKFold holds out complete subject groups. Inside each outer fit, `fit_dense_repeat_model` reruns its own grouped selection of the long/short Delta pair, the repeat statistic, its calibration, alpha, and all N3/W anchors. Gamma is not selected or weighted: every candidate marginalizes it by the same fixed nine-value median. Only that refitted model predicts the outer validation subjects. Thus each training OOF prediction is made without using that subject's labels for feature selection or calibration.",
        "",
        "After nested OOF evaluation, one final model was fitted to all labelled training-final epochs. That locked model produced continuous trajectories for every epoch and a diagnostic prediction for the declared test subjects. Because this same held-aside test split was inspected in earlier project iterations, its result is explicitly called a **repeated held-aside diagnostic**, not an untouched confirmatory test.",
        "",
        "The gamma-median family was introduced after an audit of the earlier 288-candidate development search showed severe scale/threshold winner's curse (five different definitions in five outer folds and nested W/N1 AUC 0.400). The revised 20-pair family is physically simpler and much more stable, but the nested results below helped choose that revision; they should therefore be treated as development evidence rather than a pristine preregistered estimate. `FS_ROBUSTNESS_AUDIT.md` records the comparison and rejected alternatives.",
        "",
        "## Locked full-training model",
        "",
        f"- Selected Fs candidate: `{final_model.get('selected_Fs_candidate', final_model.get('selected_S_column'))}`",
        f"- Selected Fs specification: `{json.dumps(_json_safe(final_model.get('selected_Fs_candidate_spec', {})), ensure_ascii=False)}`",
        f"- Selected repeat column: `{_model_repeat_column(final_model)}`",
        f"- Correction strength alpha: {float(final_model.get('alpha', float('nan'))):.6g}",
        f"- Training N3/W S anchors: {float(final_model['n3_anchor']):.6g} / {float(final_model['wake_anchor']):.6g}",
        f"- Definition: {final_model.get('definition', 'see portable model JSON')}",
        "",
        "## Zhang-only stage discrimination",
        "",
        "W is explicitly labelled only in Zhang, so all W-versus-stage AUCs below are Zhang-only. W is the positive class. Pure F_S is shown next to the repeat-corrected initial f_awake.",
        "",
        "### Training: nested subject-group OOF",
        "",
        "| Comparison | pure F_S AUC | corrected f_awake AUC | delta |",
        "|---|---:|---:|---:|",
        *[
            auc_line(train_eval, negative)
            for negative in ("N1", "N2", "N3", "REM", "all non-W sleep")
        ],
        "",
        f"For the primary W/N1 contrast, the subject-cluster bootstrap mean AUC change was {train_boot['mean_delta_auc_corrected_minus_pure']:+.4f}; percentile 95% CI [{train_ci[0]:+.4f}, {train_ci[1]:+.4f}]. The subject group, not the epoch, is the bootstrap unit.",
        "",
        "### Repeated held-aside diagnostic test",
        "",
        "| Comparison | pure F_S AUC | corrected f_awake AUC | delta |",
        "|---|---:|---:|---:|",
        *[
            auc_line(test_eval, negative)
            for negative in ("N1", "N2", "N3", "REM", "all non-W sleep")
        ],
        "",
        f"For W/N1, the repeated-test subject-cluster bootstrap mean AUC change was {test_boot['mean_delta_auc_corrected_minus_pure']:+.4f}; percentile 95% CI [{test_ci[0]:+.4f}, {test_ci[1]:+.4f}]. This interval describes cluster-resampling uncertainty in this diagnostic sample; it does not restore untouched-test status.",
        "",
        "## Selection stability and interpretation",
        "",
        f"Across outer folds, {len(selected_s_counts)} distinct self-similarity scale-pair definitions and {len(selected_r_counts)} distinct repeat definitions were selected. Exact fold choices and anchors are in `selection_stability.csv`. This is substantially more stable than the previous five-of-five result, but the small labelled endpoint sample still does not identify one unique physical scale.",
        "",
        f"Alpha reached {float(final_model.get('alpha', float('nan'))):.3g} in the full fit. Despite that boundary value, the fusion remains structurally secondary: it is zero at the mathematical Fs endpoints and cannot change any epoch by more than 0.25. Observed absolute changes were mean/median/90th-percentile {train_correction['mean_absolute_correction']:.4f}/{train_correction['median_absolute_correction']:.4f}/{train_correction['q90_absolute_correction']:.4f} in nested training OOF, and {test_correction['mean_absolute_correction']:.4f}/{test_correction['median_absolute_correction']:.4f}/{test_correction['q90_absolute_correction']:.4f} in the diagnostic test. Alpha saturating at its allowed maximum nevertheless means the data favored the strongest permitted modifier, not an interior weak-correction optimum.",
        "",
        "The repeat term improved nested-development W/N1 but slightly worsened the repeated diagnostic test, including W versus all non-W sleep. It therefore has not demonstrated a stable cross-subject gain. Alternative endpoint-fixed correction shells did not materially improve strict nested W/N1 performance without allowing the correction to dominate F_S; details are in `ADJACENT_CORRECTION_AUDIT.md`. Both pure F_S and corrected f_awake are retained in every output; the scientifically conservative downstream choice is to treat pure F_S as primary and the alpha=1 correction as a sensitivity branch until new labelled subjects can confirm it. The cluster bootstrap conditions on already generated OOF predictions; it does not refit the nested pipeline inside each bootstrap replicate and therefore does not include model-selection instability.",
        "",
        "## Limitations",
        "",
        "- Kumral contains no explicitly labelled W final epochs, so the W anchor and all W-based discrimination are identified by Zhang.",
        "- N3 is sparse at the manually labelled endpoints; grouped refits can therefore be sensitive to individual subjects.",
        "- The long unlabelled portions are transformed only after the model is locked; they increase trajectory coverage but not labelled evidence for selection.",
        "- AUC measures ordering, not calibration or a literal physical proof of subjective time.",
        "- No g shell, dream report, dream-duration target, or apparent-time integral is part of this step.",
        "",
        "## Output map",
        "",
        "- `dense_repeat_fawake_portable.json`: locked physical mapping and preprocessing selections.",
        "- `code/predict_dense_repeat_fawake_initial.py` and `PREDICT_DENSE_REPEAT_FAWAKE.md`: direct EDF/epoch-CSV application without g or dream information.",
        "- `training_nested_oof_predictions.csv`: leakage-controlled training predictions.",
        "- `diagnostic_test_predictions.csv`: repeated held-aside diagnostic predictions.",
        "- `all_epoch_trajectories.csv`: locked-model trajectories for all records/epochs.",
        "- `pairwise_auc_metrics.csv`, `stage_distribution_metrics.csv`, and `wn1_subject_cluster_bootstrap.csv`: numeric audits behind the plots.",
        "- `correction_magnitude_metrics.csv`: observed size and direction of the bounded repeat modifier.",
        "- `stage_distributions.png` (Zhang only), `wn1_roc_comparison.png`, `wn1_ecdf_comparison.png`, and `selection_stability.png`: visual diagnostics.",
        "",
    ]
    return "\n".join(lines)


def build_running_report(input_path: Path, output_dir: Path, args: argparse.Namespace) -> str:
    return f"""# RUNNING THE SELF-SIMILARITY + SHORT-REPEAT f_awake AUDIT

Run from the repository root with the project virtual environment:

```bash
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \\
  .venv/bin/python -m analysis.combined_fawake.fit_dense_repeat_fawake \\
  --features {input_path} \\
  --output {output_dir} \\
  --outer-splits {args.outer_splits} \\
  --inner-splits {args.inner_splits} \\
  --bootstrap-replicates {args.bootstrap_replicates}
```

The input must be the complete S + adjacent-repeat feature table produced by
the label-independent grid extractors. The runner asserts train/test subject
and record disjointness, fits only rows where `split == train` and
`is_final_labelled_epoch == True`, and never reads a dream-duration target.

The training CSV is nested subject-group OOF. The test CSV is deliberately
named a repeated held-aside diagnostic. `all_epoch_trajectories.csv` uses the
single model fitted after evaluation to all labelled training-final epochs.

The exact source snapshot used for this result is in `code/`. To apply the
portable JSON directly to one raw Zhang/Kumral EDF or one record in an epoch
CSV, follow `PREDICT_DENSE_REPEAT_FAWAKE.md`. That command outputs both pure
`Fs` and the repeat-corrected `f_awake_initial`; it does not use `g`, dream
information, spectral features, or a sleep-stage classifier.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/dense_repeat_fawake_v2/epoch_features_S_RP_grid.csv"),
        help="Complete label-independent S + adjacent-repeat feature CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dense_repeat_fawake_v2/initial_fawake_fit"),
        help="Output directory",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(args.outer_splits >= 2, "outer-splits must be at least 2")
    _require(args.inner_splits >= 2, "inner-splits must be at least 2")
    _require(args.bootstrap_replicates >= 100, "bootstrap-replicates must be at least 100")
    args.output.mkdir(parents=True, exist_ok=True)

    epochs, input_audit = load_and_assert_feature_table(args.features)
    final = epochs.loc[epochs["is_final_labelled_epoch"]].copy()
    train_final = final.loc[final["split"].eq("train")].reset_index(drop=True)
    test_final = final.loc[final["split"].eq("test")].reset_index(drop=True)

    training_oof, stability = nested_grouped_oof(
        train_final,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
    )

    final_model = fit_dense_repeat_model(
        fitting_view(train_final),
        n_splits=args.inner_splits,
        random_state=RANDOM_STATE,
    )
    print(
        f"full training fit: Fs={final_model.get('selected_Fs_candidate', final_model.get('selected_S_column'))}; "
        f"R={_model_repeat_column(final_model)}; alpha={float(final_model.get('alpha', float('nan'))):.4g}",
        flush=True,
    )
    train_groups = train_final["subject_group"].astype(str).unique()
    stability = pd.concat(
        [
            stability,
            pd.DataFrame(
                [
                    _selection_row(
                        final_model,
                        fold="final_full_train",
                        fit_rows=len(train_final),
                        validation_rows=0,
                        fit_groups=train_groups,
                        validation_groups=[],
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    diagnostic_test = apply_dense_repeat_model(fitting_view(test_final), final_model)
    diagnostic_test["prediction_protocol"] = "repeated_held_aside_diagnostic_test"
    diagnostic_test["outer_fold"] = np.nan

    # Apply the locked full-training model to the complete timeline. No label is
    # used in this call; metadata are retained only for downstream inspection.
    full_prediction = apply_dense_repeat_model(epochs, final_model)
    full_prediction["prediction_protocol"] = "locked_full_training_model"
    fs_spec = final_model.get("selected_Fs_candidate_spec", {})
    selected_columns = [
        column
        for column in (
            final_model.get("selected_S_column"),
            fs_spec.get("column"),
            fs_spec.get("long_column"),
            fs_spec.get("short_column"),
            _model_repeat_column(final_model),
        )
        if column
    ]
    added_columns = [column for column in full_prediction.columns if column not in epochs.columns]
    trajectory_columns = [
        column
        for column in (
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
            "start_s",
            "end_s",
            "duration_s",
            *selected_columns,
            *added_columns,
        )
        if column in full_prediction.columns
    ]
    # Stable de-duplication in case a selected feature is also exposed under an
    # output alias.
    trajectory_columns = list(dict.fromkeys(trajectory_columns))

    train_eval = "training_nested_subject_group_oof"
    test_eval = "repeated_held_aside_diagnostic_test"
    auc_frame = pd.DataFrame(
        pairwise_auc_rows(training_oof, train_eval)
        + pairwise_auc_rows(diagnostic_test, test_eval)
    )
    stage_frame = pd.DataFrame(
        stage_distribution_rows(training_oof, train_eval)
        + stage_distribution_rows(diagnostic_test, test_eval)
    )
    train_boot_rows, train_boot = clustered_wn1_bootstrap(
        training_oof, train_eval, replicates=args.bootstrap_replicates
    )
    test_boot_rows, test_boot = clustered_wn1_bootstrap(
        diagnostic_test, test_eval, replicates=args.bootstrap_replicates
    )
    bootstrap_frame = pd.concat([train_boot_rows, test_boot_rows], ignore_index=True)
    correction_frame = pd.DataFrame(
        [
            correction_magnitude_row(training_oof, train_eval),
            correction_magnitude_row(diagnostic_test, test_eval),
        ]
    )

    training_oof.to_csv(args.output / "training_nested_oof_predictions.csv", index=False)
    diagnostic_test.to_csv(args.output / "diagnostic_test_predictions.csv", index=False)
    full_prediction.loc[:, trajectory_columns].to_csv(
        args.output / "all_epoch_trajectories.csv", index=False
    )
    stability.to_csv(args.output / "selection_stability.csv", index=False)
    auc_frame.to_csv(args.output / "pairwise_auc_metrics.csv", index=False)
    stage_frame.to_csv(args.output / "stage_distribution_metrics.csv", index=False)
    correction_frame.to_csv(args.output / "correction_magnitude_metrics.csv", index=False)
    bootstrap_frame.to_csv(args.output / "wn1_subject_cluster_bootstrap.csv", index=False)

    portable = export_dense_repeat_model(final_model)
    _write_json(args.output / "dense_repeat_fawake_portable.json", portable)
    results = {
        "analysis": "self-similarity-first initial f_awake with bounded short-repeat correction",
        "scope": "initial f_awake only; no g, dream duration, or apparent-time integral",
        "input_audit": input_audit,
        "training_protocol": {
            "estimate": train_eval,
            "outer_splitter": "StratifiedGroupKFold by subject_group",
            "outer_splits_requested": args.outer_splits,
            "outer_splits_realized": int(stability["outer_fold"].astype(str).ne("final_full_train").sum()),
            "inner_splits_requested": args.inner_splits,
            "entire_model_refitted_per_outer_fold": True,
            "dream_information_used": False,
        },
        "test_protocol": {
            "name": test_eval,
            "untouched_confirmatory_test": False,
            "reason": "this held-aside split was inspected in earlier project iterations",
        },
        "final_model": portable,
        "pairwise_auc_metrics": auc_frame.to_dict(orient="records"),
        "wn1_subject_cluster_bootstrap": [train_boot, test_boot],
        "correction_magnitude_metrics": correction_frame.to_dict(orient="records"),
        "selection_stability": stability.to_dict(orient="records"),
    }
    _write_json(args.output / "fit_results.json", results)

    plot_stage_distributions(
        training_oof, diagnostic_test, args.output / "stage_distributions.png"
    )
    plot_wn1_roc(training_oof, diagnostic_test, args.output / "wn1_roc_comparison.png")
    plot_wn1_ecdf(training_oof, diagnostic_test, args.output / "wn1_ecdf_comparison.png")
    plot_selection_stability(stability, args.output / "selection_stability.png")

    (args.output / "METHODS_AND_RESULTS.md").write_text(
        build_methods_report(
            input_audit,
            final_model,
            auc_frame,
            [train_boot, test_boot],
            stability,
            correction_frame,
        ),
        encoding="utf-8",
    )
    (args.output / "RUNNING.md").write_text(
        build_running_report(args.features, args.output, args), encoding="utf-8"
    )
    # Keep a self-contained, human-readable snapshot beside the numeric
    # results.  These are copies only; no source or data file is removed.
    code_output = args.output / "code"
    code_output.mkdir(parents=True, exist_ok=True)
    for source_name in (
        "__init__.py",
        "eeg_features.py",
        "augment_self_similarity_grid.py",
        "augment_adjacent_recurrence_grid.py",
        "dense_repeat_fawake_model.py",
        "fit_dense_repeat_fawake.py",
        "predict_dense_repeat_fawake_initial.py",
    ):
        shutil.copy2(SCRIPT_ROOT / source_name, code_output / source_name)
    for document_name in (
        "FS_ROBUSTNESS_AUDIT.md",
        "ADJACENT_CORRECTION_AUDIT.md",
        "PREDICT_DENSE_REPEAT_FAWAKE.md",
    ):
        shutil.copy2(SCRIPT_ROOT / document_name, args.output / document_name)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
