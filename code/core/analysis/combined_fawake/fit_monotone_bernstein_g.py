#!/usr/bin/env python3
"""Fit one global endpoint-fixed monotone g to Kumral dream durations.

Protocol
--------
* IDs 1--5 are human seeds and are excluded.
* IDs 6--25 are the only records used for fitting and model selection.
* IDs 26--42 are evaluated once after the final training-only selection.
* Participant IDs, rather than individual awakenings, define every CV fold.
* The only prediction is ``sum_t g(f_t) * duration_t``.  There is no q_tau,
  record intercept, stage term, or report-derived predictor.

The input trajectory path and f column are command-line parameters so this
runner can be applied to a later revision of the physical f(t) without code
changes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.combined_fawake.monotone_bernstein_g import (  # noqa: E402
    FittedShell,
    ShellConfig,
    ShellSpec,
    balanced_loss,
    bernstein_basis,
    candidate_configs,
    fit_shell,
    predict_records,
    shell_specs,
    validate_shell,
)


STAGE_ORDER = ["W", "N1", "N2", "N3", "REM", "UNKNOWN"]
STAGE_COLORS = {
    "W": "#efb366",
    "N1": "#73b3e7",
    "N2": "#77b977",
    "N3": "#173f35",
    "REM": "#d55262",
    "UNKNOWN": "#b8bec5",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_regularization_grid(text: str) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or min(values) < 0.0:
        raise ValueError("--regularization-grid must contain non-negative numbers")
    return values


def load_consensus(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path).sort_values("annotation_order").reset_index(drop=True)
    required = {
        "annotation_order",
        "dream_report_id",
        "split",
        "subject_id",
        "record_uid",
        "eeg_filename",
        "manual_final_stage",
        "consensus_mean_s",
        "dream_report_en",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Consensus CSV is missing columns: {missing}")
    if frame["annotation_order"].tolist() != list(range(1, 43)):
        raise ValueError("Consensus CSV must contain exactly one ordered row for each ID 1--42")
    if frame["record_uid"].duplicated().any():
        raise ValueError("Consensus record_uid values must be unique")

    seed = frame.loc[frame["annotation_order"].between(1, 5)].copy()
    training = frame.loc[frame["annotation_order"].between(6, 25)].copy()
    test = frame.loc[frame["annotation_order"].between(26, 42)].copy()
    if len(seed) != 5 or len(training) != 20 or len(test) != 17:
        raise AssertionError("Expected 5 excluded seeds, 20 training records and 17 test records")
    if not training["split"].eq("train").all() or not test["split"].eq("test").all():
        raise AssertionError("ID-defined train/test partition disagrees with the split column")
    if training["consensus_mean_s"].isna().any() or test["consensus_mean_s"].isna().any():
        raise ValueError("Every non-seed record needs a finite consensus_mean_s")
    if not np.isfinite(training["consensus_mean_s"]).all() or not np.isfinite(
        test["consensus_mean_s"]
    ).all():
        raise ValueError("Dream-duration targets must be finite")

    train_subjects = set(training["subject_id"].astype(str))
    test_subjects = set(test["subject_id"].astype(str))
    overlap = sorted(train_subjects & test_subjects)
    if overlap:
        raise AssertionError(f"Participant leakage between training and test: {overlap}")

    if "used_for_kappa_fit" in frame:
        expected_train = frame["annotation_order"].between(6, 25)
        if not frame["used_for_kappa_fit"].astype(bool).eq(expected_train).all():
            raise AssertionError("used_for_kappa_fit flags disagree with IDs 6--25")
    if "used_for_holdout_test" in frame:
        expected_test = frame["annotation_order"].between(26, 42)
        if not frame["used_for_holdout_test"].astype(bool).eq(expected_test).all():
            raise AssertionError("used_for_holdout_test flags disagree with IDs 26--42")
    return frame, training.reset_index(drop=True), test.reset_index(drop=True)


def choose_stage_column(epochs: pd.DataFrame, requested: str) -> str | None:
    if requested.strip().lower() in {"none", "off", "no"}:
        return None
    if requested != "auto":
        if requested not in epochs.columns:
            raise ValueError(f"Requested stage column not found: {requested}")
        return requested
    for candidate in (
        "predicted_stage",
        "predicted_sleep_stage",
        "stage_predicted",
        "sleep_stage",
        "manual_stage",
    ):
        if candidate in epochs.columns:
            return candidate
    # ``manual_final_stage`` is the stage at the awakening/report endpoint.  It
    # is repeated on every row in the current Kumral trajectory table, so using
    # it as an epoch-wise hypnogram would be materially misleading.
    return None


def load_trajectories(
    path: Path,
    records: pd.DataFrame,
    f_column: str,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    epochs = pd.read_csv(path)
    required = {"record_uid", "duration_s", f_column}
    missing = sorted(required - set(epochs.columns))
    if missing:
        raise ValueError(f"Trajectory CSV is missing columns: {missing}")
    if "dataset" in epochs.columns:
        epochs = epochs.loc[epochs["dataset"].eq("Kumral")].copy()
    wanted = set(records["record_uid"])
    epochs = epochs.loc[epochs["record_uid"].isin(wanted)].copy()
    present = set(epochs["record_uid"])
    if present != wanted:
        raise ValueError(
            f"Trajectory record mismatch: missing={sorted(wanted-present)}, "
            f"unexpected={sorted(present-wanted)}"
        )
    sort_columns = [column for column in ("record_uid", "epoch_index", "start_s") if column in epochs]
    epochs = epochs.sort_values(sort_columns).reset_index(drop=True)
    epochs[f_column] = pd.to_numeric(epochs[f_column], errors="raise")
    epochs["duration_s"] = pd.to_numeric(epochs["duration_s"], errors="raise")
    if not np.isfinite(epochs[f_column]).all() or not np.isfinite(epochs["duration_s"]).all():
        raise ValueError("Trajectory f and duration values must be finite")
    tolerance = 1e-9
    if epochs[f_column].min() < -tolerance or epochs[f_column].max() > 1.0 + tolerance:
        raise ValueError(f"{f_column} must lie in [0, 1]")
    if (epochs["duration_s"] <= 0.0).any():
        raise ValueError("Every epoch duration must be positive")
    epochs[f_column] = epochs[f_column].clip(0.0, 1.0)

    if "split" in epochs.columns:
        expected_split = records.set_index("record_uid")["split"].astype(str).to_dict()
        observed = epochs.groupby("record_uid")["split"].agg(lambda values: set(values.astype(str)))
        for uid, values in observed.items():
            if values != {expected_split[uid]}:
                raise AssertionError(f"Trajectory split mismatch for {uid}: {values}")
    if "subject_id" in epochs.columns:
        expected_subject = records.set_index("record_uid")["subject_id"].astype(str).to_dict()
        observed = epochs.groupby("record_uid")["subject_id"].agg(lambda values: set(values.astype(str)))
        for uid, values in observed.items():
            if values != {expected_subject[uid]}:
                raise AssertionError(f"Trajectory subject mismatch for {uid}: {values}")

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for uid, group in epochs.groupby("record_uid", sort=False):
        arrays[str(uid)] = (
            group[f_column].to_numpy(dtype=float),
            group["duration_s"].to_numpy(dtype=float),
        )
    return epochs, arrays


def target_maps(records: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    targets = records.set_index("record_uid")["consensus_mean_s"].astype(float).to_dict()
    subjects = records.set_index("record_uid")["subject_id"].astype(str).to_dict()
    return targets, subjects


def safe_correlations(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(target) < 2 or np.std(target) == 0.0 or np.std(predicted) == 0.0:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }
    pearson = pearsonr(target, predicted)
    spearman = spearmanr(target, predicted)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def metric_summary(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    target = np.asarray(target, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - target
    denominator = float(np.sum(np.square(target - target.mean())))
    result: dict[str, float | int] = {
        "n": int(len(target)),
        "mae_s": float(np.mean(np.abs(residual))),
        "rmse_s": float(np.sqrt(np.mean(np.square(residual)))),
        "median_absolute_error_s": float(np.median(np.abs(residual))),
        "log_mse": float(np.mean(np.square(np.log1p(predicted) - np.log1p(target)))),
        "r2_identity": (
            1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0.0 else np.nan
        ),
        "mean_target_s": float(target.mean()),
        "mean_prediction_s": float(predicted.mean()),
    }
    result.update(safe_correlations(target, predicted))
    return result


def grouped_splits(records: pd.DataFrame, requested: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = records["subject_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(int(requested), n_groups)
    if n_splits < 2:
        raise ValueError("At least two participants are required for grouped CV")
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(records, groups=groups))
    for fit_indices, validation_indices in splits:
        fit_groups = set(groups[fit_indices])
        validation_groups = set(groups[validation_indices])
        if fit_groups & validation_groups:
            raise AssertionError("Participant leakage within grouped CV")
    return splits


def evaluate_configs(
    records: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    requested_splits: int,
    loss_mode: str,
    context: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate configurations with grouped CV, returning summaries and fold rows."""

    targets, subjects = target_maps(records)
    splits = grouped_splits(records, requested_splits)
    summaries: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for config in configs:
        fold_losses: list[float] = []
        oof_prediction: dict[str, float] = {}
        for fold_index, (fit_indices, validation_indices) in enumerate(splits, start=1):
            fit_rows = records.iloc[fit_indices]
            validation_rows = records.iloc[validation_indices]
            fit_uids = fit_rows["record_uid"].astype(str).tolist()
            validation_uids = validation_rows["record_uid"].astype(str).tolist()
            fitted = fit_shell(config, fit_uids, arrays, targets, subjects, loss_mode=loss_mode)
            predicted = predict_records(config.spec, fitted.parameters, validation_uids, arrays)
            target = validation_rows["consensus_mean_s"].to_numpy(dtype=float)
            validation_subjects = validation_rows["subject_id"].astype(str).tolist()
            validation_loss = balanced_loss(
                predicted, target, validation_subjects, loss_mode=loss_mode
            )
            fold_losses.append(validation_loss)
            for uid, value in zip(validation_uids, predicted, strict=True):
                if uid in oof_prediction:
                    raise AssertionError(f"Duplicate OOF prediction for {uid}")
                oof_prediction[uid] = float(value)
            fold_rows.append(
                {
                    "context": context,
                    "config_key": config.key,
                    "family_key": config.spec.key,
                    "family_label": config.spec.label,
                    "parameter_count": config.spec.parameter_count,
                    "regularization": config.regularization,
                    "primary_eligible": config.spec.primary_eligible,
                    "fold": fold_index,
                    "fit_subjects": int(fit_rows["subject_id"].nunique()),
                    "validation_subjects": int(validation_rows["subject_id"].nunique()),
                    "fit_records": int(len(fit_rows)),
                    "validation_records": int(len(validation_rows)),
                    "training_regularized_objective": fitted.objective,
                    "training_data_loss": fitted.data_loss,
                    "training_identity_penalty": fitted.identity_penalty,
                    "validation_loss": validation_loss,
                    "parameters_json": json.dumps(
                        fitted.decoded_parameters(), sort_keys=True, ensure_ascii=False
                    ),
                }
            )
        expected = set(records["record_uid"].astype(str))
        if set(oof_prediction) != expected:
            raise AssertionError(f"OOF coverage failure in {context}/{config.key}")
        order = records["record_uid"].astype(str).tolist()
        oof = np.asarray([oof_prediction[uid] for uid in order], dtype=float)
        target = records["consensus_mean_s"].to_numpy(dtype=float)
        fold_array = np.asarray(fold_losses, dtype=float)
        summary = {
            "context": context,
            "config_key": config.key,
            "family_key": config.spec.key,
            "family_label": config.spec.label,
            "parameter_count": config.spec.parameter_count,
            "regularization": config.regularization,
            "primary_eligible": config.spec.primary_eligible,
            "mean_fold_loss": float(fold_array.mean()),
            "fold_loss_se": float(fold_array.std(ddof=1) / math.sqrt(len(fold_array)))
            if len(fold_array) > 1
            else 0.0,
            "pooled_log_mse": metric_summary(target, oof)["log_mse"],
            "pooled_mae_s": metric_summary(target, oof)["mae_s"],
            "pooled_rmse_s": metric_summary(target, oof)["rmse_s"],
            "pooled_pearson_r": metric_summary(target, oof)["pearson_r"],
            "pooled_spearman_rho": metric_summary(target, oof)["spearman_rho"],
        }
        summaries.append(summary)
    return summaries, fold_rows


def _family_preference(spec: ShellSpec) -> int:
    order = {
        "identity": 0,
        "signed_exponential": 1,
        "bernstein": 2,
        "below_exponential": 3,
    }
    return order[spec.kind]


def choose_one_standard_error(
    summaries: Sequence[dict[str, Any]],
    configs_by_key: Mapping[str, ShellConfig],
    primary_only: bool,
) -> tuple[ShellConfig, dict[str, Any]]:
    candidates = [
        row
        for row in summaries
        if (bool(row["primary_eligible"]) if primary_only else not bool(row["primary_eligible"]))
    ]
    if not candidates:
        raise ValueError("No configurations available for selection")
    best = min(candidates, key=lambda row: float(row["mean_fold_loss"]))
    threshold = float(best["mean_fold_loss"]) + float(best["fold_loss_se"])
    within = [row for row in candidates if float(row["mean_fold_loss"]) <= threshold + 1e-15]

    def preference(row: dict[str, Any]) -> tuple[float, ...]:
        config = configs_by_key[str(row["config_key"])]
        # One-SE rule: prefer fewer degrees of freedom; within equal complexity,
        # prefer the more constrained named family and then stronger shrinkage.
        return (
            float(config.spec.parameter_count),
            float(_family_preference(config.spec)),
            -float(config.regularization),
            float(row["mean_fold_loss"]),
        )

    chosen_row = min(within, key=preference)
    chosen = configs_by_key[str(chosen_row["config_key"])]
    details = {
        "empirical_best_config_key": str(best["config_key"]),
        "empirical_best_mean_fold_loss": float(best["mean_fold_loss"]),
        "empirical_best_fold_loss_se": float(best["fold_loss_se"]),
        "one_se_threshold": threshold,
        "chosen_config_key": chosen.key,
        "eligible_within_one_se": [str(row["config_key"]) for row in within],
    }
    return chosen, details


def choose_empirical_minimum(
    summaries: Sequence[dict[str, Any]],
    configs_by_key: Mapping[str, ShellConfig],
    primary_only: bool,
) -> tuple[ShellConfig, dict[str, Any]]:
    """Choose minimum training-CV loss; test data are absent from this decision."""

    candidates = [
        row
        for row in summaries
        if (bool(row["primary_eligible"]) if primary_only else not bool(row["primary_eligible"]))
    ]
    if not candidates:
        raise ValueError("No configurations available for selection")
    best = min(
        candidates,
        key=lambda row: (
            float(row["mean_fold_loss"]),
            int(row["parameter_count"]),
            -float(row["regularization"]),
        ),
    )
    chosen = configs_by_key[str(best["config_key"])]
    details = {
        "selection_rule": "minimum mean participant-grouped CV loss on training only",
        "chosen_config_key": chosen.key,
        "chosen_mean_fold_loss": float(best["mean_fold_loss"]),
        "chosen_fold_loss_se": float(best["fold_loss_se"]),
    }
    return chosen, details


def choose_configuration(
    summaries: Sequence[dict[str, Any]],
    configs_by_key: Mapping[str, ShellConfig],
    primary_only: bool,
    selection_rule: str,
) -> tuple[ShellConfig, dict[str, Any]]:
    if selection_rule == "one_se":
        chosen, details = choose_one_standard_error(
            summaries, configs_by_key, primary_only=primary_only
        )
        return chosen, {"selection_rule": "one-standard-error", **details}
    if selection_rule == "empirical_minimum":
        return choose_empirical_minimum(
            summaries, configs_by_key, primary_only=primary_only
        )
    raise ValueError(f"Unknown selection rule: {selection_rule}")


def nested_training_oof(
    training: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    outer_splits: int,
    inner_splits: int,
    loss_mode: str,
    selection_rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs_by_key = {config.key: config for config in configs}
    primary_configs = [config for config in configs if config.spec.primary_eligible]
    control_configs = [config for config in configs if not config.spec.primary_eligible]
    targets, subjects = target_maps(training)
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_rows_all: list[dict[str, Any]] = []
    for outer_fold, (fit_indices, validation_indices) in enumerate(
        grouped_splits(training, outer_splits), start=1
    ):
        fit_rows = training.iloc[fit_indices].reset_index(drop=True)
        validation_rows = training.iloc[validation_indices].reset_index(drop=True)
        fit_subjects = set(fit_rows["subject_id"].astype(str))
        validation_subjects = set(validation_rows["subject_id"].astype(str))
        if fit_subjects & validation_subjects:
            raise AssertionError("Outer-fold participant leakage")

        primary_summary, primary_fold_rows = evaluate_configs(
            fit_rows,
            arrays,
            primary_configs,
            requested_splits=inner_splits,
            loss_mode=loss_mode,
            context=f"{selection_rule}_outer_{outer_fold}_primary_inner",
        )
        control_summary, control_fold_rows = evaluate_configs(
            fit_rows,
            arrays,
            control_configs,
            requested_splits=inner_splits,
            loss_mode=loss_mode,
            context=f"{selection_rule}_outer_{outer_fold}_control_inner",
        )
        inner_rows_all.extend(primary_summary)
        inner_rows_all.extend(control_summary)
        inner_rows_all.extend(primary_fold_rows)
        inner_rows_all.extend(control_fold_rows)
        chosen, primary_details = choose_configuration(
            primary_summary,
            configs_by_key,
            primary_only=True,
            selection_rule=selection_rule,
        )
        control, control_details = choose_configuration(
            control_summary,
            configs_by_key,
            primary_only=False,
            selection_rule=selection_rule,
        )
        fit_uids = fit_rows["record_uid"].astype(str).tolist()
        validation_uids = validation_rows["record_uid"].astype(str).tolist()
        fitted = fit_shell(chosen, fit_uids, arrays, targets, subjects, loss_mode=loss_mode)
        fitted_control = fit_shell(
            control, fit_uids, arrays, targets, subjects, loss_mode=loss_mode
        )
        selected_prediction = predict_records(
            chosen.spec, fitted.parameters, validation_uids, arrays
        )
        control_prediction = predict_records(
            control.spec, fitted_control.parameters, validation_uids, arrays
        )
        identity_spec = next(spec for spec in shell_specs() if spec.kind == "identity")
        identity_prediction = predict_records(
            identity_spec, np.empty(0), validation_uids, arrays
        )
        for record, selected_value, identity_value, control_value in zip(
            validation_rows.itertuples(index=False),
            selected_prediction,
            identity_prediction,
            control_prediction,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "outer_fold": outer_fold,
                    "annotation_order": int(record.annotation_order),
                    "dream_report_id": record.dream_report_id,
                    "split": "train",
                    "subject_id": str(record.subject_id),
                    "record_uid": str(record.record_uid),
                    "manual_final_stage": record.manual_final_stage,
                    "consensus_mean_s": float(record.consensus_mean_s),
                    "selected_nested_oof_apparent_s": float(selected_value),
                    "identity_apparent_s": float(identity_value),
                    "below_control_nested_oof_apparent_s": float(control_value),
                    "selected_config_key": chosen.key,
                    "selected_parameters_json": json.dumps(
                        fitted.decoded_parameters(), sort_keys=True
                    ),
                    "control_config_key": control.key,
                    "control_parameters_json": json.dumps(
                        fitted_control.decoded_parameters(), sort_keys=True
                    ),
                }
            )
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "selection_rule": selection_rule,
                "fit_subjects": len(fit_subjects),
                "validation_subjects": len(validation_subjects),
                "fit_subject_ids": ";".join(sorted(fit_subjects)),
                "validation_subject_ids": ";".join(sorted(validation_subjects)),
                "selected_config_key": chosen.key,
                "selected_family_key": chosen.spec.key,
                "selected_regularization": chosen.regularization,
                "selected_parameters_json": json.dumps(
                    fitted.decoded_parameters(), sort_keys=True
                ),
                "control_config_key": control.key,
                "control_parameters_json": json.dumps(
                    fitted_control.decoded_parameters(), sort_keys=True
                ),
                "primary_selection_json": json.dumps(primary_details, sort_keys=True),
                "control_selection_json": json.dumps(control_details, sort_keys=True),
            }
        )
    predictions = pd.DataFrame(prediction_rows).sort_values("annotation_order").reset_index(drop=True)
    if predictions["annotation_order"].tolist() != list(range(6, 26)):
        raise AssertionError("Nested OOF must predict each training record exactly once")
    return predictions, pd.DataFrame(selection_rows), pd.DataFrame(inner_rows_all)


def fit_final_models(
    training: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    inner_splits: int,
    loss_mode: str,
    selection_rule: str,
) -> tuple[
    FittedShell,
    FittedShell,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    configs_by_key = {config.key: config for config in configs}
    primary_configs = [config for config in configs if config.spec.primary_eligible]
    control_configs = [config for config in configs if not config.spec.primary_eligible]
    primary_summary, primary_fold_rows = evaluate_configs(
        training,
        arrays,
        primary_configs,
        requested_splits=inner_splits,
        loss_mode=loss_mode,
        context=f"full_training_{selection_rule}_primary_selection",
    )
    control_summary, control_fold_rows = evaluate_configs(
        training,
        arrays,
        control_configs,
        requested_splits=inner_splits,
        loss_mode=loss_mode,
        context=f"full_training_{selection_rule}_control_selection",
    )
    chosen, primary_details = choose_configuration(
        primary_summary,
        configs_by_key,
        primary_only=True,
        selection_rule=selection_rule,
    )
    control, control_details = choose_configuration(
        control_summary,
        configs_by_key,
        primary_only=False,
        selection_rule=selection_rule,
    )
    targets, subjects = target_maps(training)
    uids = training["record_uid"].astype(str).tolist()
    fitted = fit_shell(chosen, uids, arrays, targets, subjects, loss_mode=loss_mode)
    fitted_control = fit_shell(control, uids, arrays, targets, subjects, loss_mode=loss_mode)
    summary_frame = pd.DataFrame(primary_summary + control_summary)
    fold_frame = pd.DataFrame(primary_fold_rows + control_fold_rows)
    return fitted, fitted_control, summary_frame, fold_frame, primary_details, control_details


def fixed_config_training_oof(
    training: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    final_model: FittedShell,
    outer_splits: int,
    loss_mode: str,
    track: str,
) -> pd.DataFrame:
    """OOF predictions for a configuration chosen using all training CV.

    These rows isolate parameter-estimation stability for the final fixed
    family/configuration.  They are less selection-honest than the nested OOF
    rows and are labelled accordingly.
    """

    targets, subjects = target_maps(training)
    config = ShellConfig(final_model.spec, final_model.regularization)
    rows: list[dict[str, Any]] = []
    for fold, (fit_indices, validation_indices) in enumerate(
        grouped_splits(training, outer_splits), start=1
    ):
        fit_rows = training.iloc[fit_indices]
        validation_rows = training.iloc[validation_indices]
        fit_uids = fit_rows["record_uid"].astype(str).tolist()
        validation_uids = validation_rows["record_uid"].astype(str).tolist()
        fitted = fit_shell(config, fit_uids, arrays, targets, subjects, loss_mode=loss_mode)
        prediction = predict_records(
            config.spec, fitted.parameters, validation_uids, arrays
        )
        for record, value in zip(
            validation_rows.itertuples(index=False), prediction, strict=True
        ):
            rows.append(
                {
                    "track": track,
                    "protocol": "fixed_config_grouped_oof; config chosen on all training CV",
                    "outer_fold": fold,
                    "annotation_order": int(record.annotation_order),
                    "dream_report_id": record.dream_report_id,
                    "split": "train",
                    "subject_id": str(record.subject_id),
                    "record_uid": str(record.record_uid),
                    "manual_final_stage": record.manual_final_stage,
                    "consensus_mean_s": float(record.consensus_mean_s),
                    "fixed_config_oof_apparent_s": float(value),
                    "config_key": config.key,
                    "parameters_json": json.dumps(
                        fitted.decoded_parameters(), sort_keys=True
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values("annotation_order").reset_index(drop=True)
    if result["annotation_order"].tolist() != list(range(6, 26)):
        raise AssertionError("Fixed-config OOF must predict each training ID exactly once")
    return result


def build_holdout_predictions(
    test: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    selected: FittedShell,
    control: FittedShell,
) -> pd.DataFrame:
    uids = test["record_uid"].astype(str).tolist()
    identity_spec = next(spec for spec in shell_specs() if spec.kind == "identity")
    selected_values = predict_records(selected.spec, selected.parameters, uids, arrays)
    identity_values = predict_records(identity_spec, np.empty(0), uids, arrays)
    control_values = predict_records(control.spec, control.parameters, uids, arrays)
    rows = []
    for record, selected_value, identity_value, control_value in zip(
        test.itertuples(index=False), selected_values, identity_values, control_values, strict=True
    ):
        physical_s = float(np.sum(arrays[str(record.record_uid)][1]))
        target_s = float(record.consensus_mean_s)
        feasible = target_s <= physical_s + 1e-9
        rows.append(
            {
                "annotation_order": int(record.annotation_order),
                "dream_report_id": record.dream_report_id,
                "split": "test",
                "subject_id": str(record.subject_id),
                "record_uid": str(record.record_uid),
                "eeg_filename": record.eeg_filename,
                "manual_final_stage": record.manual_final_stage,
                "physical_modelled_s": physical_s,
                "consensus_mean_s": target_s,
                "consensus_sigma_population_s": float(record.consensus_sigma_population_s)
                if hasattr(record, "consensus_sigma_population_s")
                else np.nan,
                "target_feasible_under_bounded_g": feasible,
                "unavoidable_shortfall_s": max(target_s - physical_s, 0.0),
                "selected_apparent_s": float(selected_value),
                "identity_apparent_s": float(identity_value),
                "below_control_apparent_s": float(control_value),
                "selected_config_key": f"{selected.spec.key}__lambda_{selected.regularization:g}",
                "selected_parameters_json": json.dumps(
                    selected.decoded_parameters(), sort_keys=True
                ),
                "below_control_config_key": (
                    f"{control.spec.key}__lambda_{control.regularization:g}"
                ),
                "below_control_parameters_json": json.dumps(
                    control.decoded_parameters(), sort_keys=True
                ),
                "dream_report_en": record.dream_report_en,
            }
        )
    result = pd.DataFrame(rows).sort_values("annotation_order").reset_index(drop=True)
    impossible_ids = result.loc[~result["target_feasible_under_bounded_g"], "annotation_order"].tolist()
    if 29 not in impossible_ids:
        raise AssertionError("Expected ID 29 to be flagged target>physical duration")
    return result


def plot_correlation(
    frame: pd.DataFrame,
    prediction_column: str,
    title: str,
    output: Path,
) -> None:
    target = frame["consensus_mean_s"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    metrics = metric_summary(target, predicted)
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    for stage in STAGE_ORDER:
        group = frame.loc[frame["manual_final_stage"].fillna("UNKNOWN").eq(stage)]
        if group.empty:
            continue
        ax.scatter(
            group["consensus_mean_s"] / 60.0,
            group[prediction_column] / 60.0,
            s=52,
            color=STAGE_COLORS[stage],
            alpha=0.86,
            label=stage,
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                str(row.annotation_order),
                (row.consensus_mean_s / 60.0, getattr(row, prediction_column) / 60.0),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    maximum = max(float(target.max()), float(predicted.max())) / 60.0
    ax.plot([0.0, maximum], [0.0, maximum], "--", color="#64717b", lw=1.3)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Six-LLM mean movie-equivalent dream duration (min)")
    ax.set_ylabel(r"EEG apparent time $\sum g(f_i)\,\Delta t_i$ (min)")
    ax.set_title(
        title
        + "\n"
        + f"Pearson r={metrics['pearson_r']:.3f}; "
        + f"Spearman rho={metrics['spearman_rho']:.3f}; RMSE={metrics['rmse_s']:.0f} s"
    )
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=210)
    plt.close(fig)


def plot_g_curves(selected: FittedShell, control: FittedShell, output: Path, data_output: Path) -> None:
    grid = np.linspace(0.0, 1.0, 2001)
    frame = pd.DataFrame(
        {
            "f_input": grid,
            "identity": grid,
            "selected_g": selected.transform(grid),
            "below_identity_control_g": control.transform(grid),
        }
    )
    frame.to_csv(data_output, index=False)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(grid, grid, "--", color="#7a8791", lw=1.5, label="identity")
    ax.plot(grid, frame["selected_g"], color="#15384c", lw=2.8, label=selected.spec.label)
    ax.plot(
        grid,
        frame["below_identity_control_g"],
        color="#b65e43",
        lw=1.8,
        label="below-identity control",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Physical $f(t)$")
    ax.set_ylabel(r"$g(f(t))$")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=210)
    plt.close(fig)


def plot_g_tracks(
    conservative: FittedShell,
    empirical: FittedShell,
    control: FittedShell,
    output: Path,
    data_output: Path,
) -> None:
    """Plot the two training-locked tracks and the declared control."""

    grid = np.linspace(0.0, 1.0, 2001)
    frame = pd.DataFrame(
        {
            "f_input": grid,
            "identity": grid,
            "conservative_one_se_g": conservative.transform(grid),
            "empirical_cv_best_g": empirical.transform(grid),
            "below_identity_control_g": control.transform(grid),
        }
    )
    frame.to_csv(data_output, index=False)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.plot(grid, grid, "--", color="#7a8791", lw=1.4, label="identity")
    ax.plot(
        grid,
        frame["conservative_one_se_g"],
        color="#4063a5",
        lw=2.2,
        label=f"A: {conservative.spec.label}",
    )
    ax.plot(
        grid,
        frame["empirical_cv_best_g"],
        color="#15384c",
        lw=3.0,
        label=f"B: {empirical.spec.label}",
    )
    ax.plot(
        grid,
        frame["below_identity_control_g"],
        color="#b65e43",
        lw=1.7,
        label="below-identity control",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Physical $f(t)$")
    ax.set_ylabel(r"$g(f(t))$")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(output, dpi=210)
    plt.close(fig)


def _time_axes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = frame["duration_s"].to_numpy(dtype=float)
    if "start_s" in frame.columns and "end_s" in frame.columns:
        start = frame["start_s"].to_numpy(dtype=float)
        end = frame["end_s"].to_numpy(dtype=float)
    else:
        end = np.cumsum(duration)
        start = end - duration
    midpoint = 0.5 * (start + end)
    return start, end, midpoint


def make_test_sample_outputs(
    epochs: pd.DataFrame,
    test_predictions: pd.DataFrame,
    f_column: str,
    stage_column: str | None,
    selected: FittedShell,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in test_predictions.itertuples(index=False):
        group = epochs.loc[epochs["record_uid"].eq(record.record_uid)].copy()
        order_columns = [column for column in ("epoch_index", "start_s") if column in group]
        if order_columns:
            group = group.sort_values(order_columns)
        start, end, midpoint = _time_axes(group)
        base = group[f_column].to_numpy(dtype=float)
        transformed = selected.transform(base)
        duration = group["duration_s"].to_numpy(dtype=float)
        redraw = pd.DataFrame(
            {
                "record_uid": record.record_uid,
                "annotation_order": int(record.annotation_order),
                "epoch_index": group["epoch_index"].to_numpy()
                if "epoch_index" in group
                else np.arange(len(group)),
                "start_s": start,
                "end_s": end,
                "midpoint_s": midpoint,
                "duration_s": duration,
                "f_physical": base,
                "g_of_f": transformed,
                "physical_increment_s": duration,
                "apparent_increment_s": transformed * duration,
                "cumulative_physical_s": np.cumsum(duration),
                "cumulative_apparent_s": np.cumsum(transformed * duration),
                "stage": group[stage_column].fillna("UNKNOWN").astype(str).to_numpy()
                if stage_column is not None
                else "UNKNOWN",
            }
        )
        stem = f"ID_{int(record.annotation_order):03d}_{Path(record.eeg_filename).stem}"
        redraw.to_csv(output_dir / f"{stem}.csv", index=False)

        report = str(record.dream_report_en)
        wrapped = textwrap.fill(report, width=112)
        stage_note = (
            "\n\nEpoch-wise sleep-stage labels are unavailable; the title reports only "
            "the manual stage at the awakening endpoint."
            if stage_column is None
            else ""
        )
        feasibility = (
            "target is feasible under 0<=g<=1"
            if bool(record.target_feasible_under_bounded_g)
            else "target exceeds the modeled physical record and is unattainable under 0<=g<=1"
        )
        report_text = (
            "Dream report (English)\n"
            + wrapped
            + "\n\n"
            + f"Target={record.consensus_mean_s:.1f} s; prediction={record.selected_apparent_s:.1f} s; "
            + feasibility
            + stage_note
        )
        report_lines = max(5, report_text.count("\n") + 1)
        height = 7.7 + report_lines * 0.14
        fig = plt.figure(figsize=(13.2, height))
        report_ratio = max(1.1, report_lines / 6.0)
        grid = fig.add_gridspec(
            3, 1, height_ratios=[2.2, 1.25, report_ratio], hspace=0.34
        )
        top = fig.add_subplot(grid[0])
        middle = fig.add_subplot(grid[1], sharex=top)
        report_ax = fig.add_subplot(grid[2])

        stage_values = redraw["stage"].fillna("UNKNOWN").astype(str).to_numpy()
        if stage_column is not None:
            for left, right, stage in zip(start, end, stage_values, strict=True):
                normalized = stage if stage in STAGE_COLORS else "UNKNOWN"
                top.axvspan(
                    left / 60.0,
                    right / 60.0,
                    color=STAGE_COLORS[normalized],
                    alpha=0.14,
                    lw=0,
                )
        base_line = top.plot(
            midpoint / 60.0, base, color="#668496", lw=1.2, label="physical f(t)"
        )[0]
        transformed_line = top.plot(
            midpoint / 60.0, transformed, color="#15384c", lw=2.0, label="g(f(t))"
        )[0]
        top.set_ylim(-0.03, 1.03)
        top.set_ylabel(r"$\bar f(\lambda)$")
        top.set_title(
            f"Kumral dream ID {int(record.annotation_order)} | awakening endpoint stage "
            f"{record.manual_final_stage} | {selected.spec.label}"
        )
        top.grid(True, alpha=0.15)
        present_stages = (
            [stage for stage in STAGE_ORDER if stage in set(stage_values)]
            if stage_column is not None
            else []
        )
        stage_handles = [
            Patch(facecolor=STAGE_COLORS[stage], alpha=0.28, label=stage)
            for stage in present_stages
        ]
        top.legend(
            handles=[base_line, transformed_line, *stage_handles],
            frameon=False,
            ncol=min(7, 2 + len(stage_handles)),
        )

        middle.plot(
            end / 60.0,
            redraw["cumulative_apparent_s"] / 60.0,
            color="#963d72",
            lw=2.2,
            label="cumulative apparent time",
        )
        middle.axhline(
            float(record.consensus_mean_s) / 60.0,
            color="#b65e43",
            ls="--",
            lw=1.5,
            label="six-LLM dream-duration mean",
        )
        middle.set_xlabel("Physical time from EDF start (min)")
        middle.set_ylabel("Cumulative apparent\ntime (min)")
        middle.grid(True, alpha=0.18)
        middle.legend(frameon=False)

        report_ax.axis("off")
        report_ax.text(
            0.0,
            1.0,
            report_text,
            ha="left",
            va="top",
            fontsize=8.9,
            transform=report_ax.transAxes,
        )
        fig.savefig(output_dir / f"{stem}.png", dpi=190, bbox_inches="tight")
        plt.close(fig)


def model_payload(fitted: FittedShell, f_column: str) -> dict[str, Any]:
    payload = {
        "model_type": "global_endpoint_fixed_monotone_g",
        "input_f_column": f_column,
        "family": asdict(fitted.spec),
        "regularization": fitted.regularization,
        "parameters": fitted.parameters,
        "decoded_parameters": fitted.decoded_parameters(),
        "validation": validate_shell(fitted.spec, fitted.parameters),
        "apparent_time_definition": "sum_t g(f_t) * duration_t",
        "forbidden_terms": ["q_tau", "record-specific parameter", "dream-text predictor"],
    }
    if fitted.spec.kind in {"signed_exponential", "below_exponential"}:
        kappa = float(fitted.parameters[0])
        payload["positive_kappa_boundary_reached"] = bool(kappa >= 499.9)
        if kappa >= 499.9:
            payload["boundary_interpretation"] = (
                "The loss prefers the endpoint-step limit g(x<1)->0, g(1)=1; "
                "500 is a numerical representation of that limiting behavior, not a "
                "well-identified finite curvature."
            )
    return payload


def _weighted_cdf(values: np.ndarray, durations: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(durations, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights) / np.sum(sorted_weights)
    indices = np.searchsorted(sorted_values, grid, side="right") - 1
    result = np.zeros(len(grid), dtype=float)
    valid = indices >= 0
    result[valid] = cumulative[indices[valid]]
    return result


def write_identifiability_audit(
    records: pd.DataFrame,
    training: pd.DataFrame,
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    f_column: str,
    loss_mode: str,
    output: Path,
) -> dict[str, Any]:
    """Write finite-resolution oracle and monotonicity-conflict diagnostics."""

    audit_dir = output / "identifiability_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    metadata = records.set_index("record_uid").to_dict(orient="index")

    histogram_rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 51)
    for uid in records["record_uid"].astype(str):
        values, durations = arrays[uid]
        duration_by_bin, _ = np.histogram(values, bins=edges, weights=durations)
        count_by_bin, _ = np.histogram(values, bins=edges)
        total = float(np.sum(durations))
        for index, (left, right, seconds, count) in enumerate(
            zip(edges[:-1], edges[1:], duration_by_bin, count_by_bin, strict=True)
        ):
            histogram_rows.append(
                {
                    "annotation_order": int(metadata[uid]["annotation_order"]),
                    "split": metadata[uid]["split"],
                    "subject_id": metadata[uid]["subject_id"],
                    "record_uid": uid,
                    "bin_index": index,
                    "f_left": left,
                    "f_right": right,
                    "f_center": 0.5 * (left + right),
                    "epoch_count": int(count),
                    "duration_s": float(seconds),
                    "duration_fraction": float(seconds / total),
                }
            )
    pd.DataFrame(histogram_rows).to_csv(audit_dir / "record_f_histograms.csv", index=False)

    sufficient_rows: list[dict[str, Any]] = []
    for uid in records["record_uid"].astype(str):
        values, durations = arrays[uid]
        for degree in range(2, 6):
            seconds = np.sum(
                bernstein_basis(values, degree) * durations[:, None], axis=0
            )
            for basis_index, value in enumerate(seconds):
                sufficient_rows.append(
                    {
                        "annotation_order": int(metadata[uid]["annotation_order"]),
                        "split": metadata[uid]["split"],
                        "record_uid": uid,
                        "degree": degree,
                        "basis_index": basis_index,
                        "basis_integral_s": float(value),
                        "check_total_s": float(np.sum(seconds)),
                        "physical_modelled_s": float(np.sum(durations)),
                    }
                )
    sufficient = pd.DataFrame(sufficient_rows)
    if not np.allclose(
        sufficient["check_total_s"], sufficient["physical_modelled_s"], atol=1e-8
    ):
        raise AssertionError("Bernstein sufficient statistics do not sum to physical duration")
    sufficient.to_csv(audit_dir / "bernstein_sufficient_statistics.csv", index=False)

    training_meta = training.set_index("record_uid")
    dominance_rows: list[dict[str, Any]] = []
    training_uids = training["record_uid"].astype(str).tolist()
    for dominant_uid in training_uids:
        dominant_values, dominant_duration = arrays[dominant_uid]
        dominant_physical = float(np.sum(dominant_duration))
        dominant_target_fraction = float(
            training_meta.loc[dominant_uid, "consensus_mean_s"] / dominant_physical
        )
        for dominated_uid in training_uids:
            if dominant_uid == dominated_uid:
                continue
            dominated_values, dominated_duration = arrays[dominated_uid]
            grid = np.unique(np.concatenate([dominant_values, dominated_values, [0.0, 1.0]]))
            dominant_cdf = _weighted_cdf(dominant_values, dominant_duration, grid)
            dominated_cdf = _weighted_cdf(dominated_values, dominated_duration, grid)
            if np.all(dominant_cdf <= dominated_cdf + 1e-12):
                dominated_physical = float(np.sum(dominated_duration))
                dominated_target_fraction = float(
                    training_meta.loc[dominated_uid, "consensus_mean_s"] / dominated_physical
                )
                dominance_rows.append(
                    {
                        "dominant_annotation_order": int(
                            training_meta.loc[dominant_uid, "annotation_order"]
                        ),
                        "dominant_record_uid": dominant_uid,
                        "dominated_annotation_order": int(
                            training_meta.loc[dominated_uid, "annotation_order"]
                        ),
                        "dominated_record_uid": dominated_uid,
                        "dominant_target_fraction": dominant_target_fraction,
                        "dominated_target_fraction": dominated_target_fraction,
                        "target_order_conflict": bool(
                            dominant_target_fraction + 1e-12 < dominated_target_fraction
                        ),
                        "target_fraction_gap_opposite_to_monotonicity": float(
                            max(dominated_target_fraction - dominant_target_fraction, 0.0)
                        ),
                        "meaning": (
                            "For every nondecreasing g, mean_g(dominant) >= "
                            "mean_g(dominated)."
                        ),
                    }
                )
    dominance = pd.DataFrame(dominance_rows)
    dominance.to_csv(audit_dir / "training_stochastic_dominance_pairs.csv", index=False)
    id10_conflicts = (
        dominance.loc[
            dominance["dominated_annotation_order"].eq(10)
            & dominance["target_order_conflict"]
        ].to_dict(orient="records")
        if not dominance.empty
        else []
    )
    id23_dominates_id10_conflict_found = any(
        row["dominant_annotation_order"] == 23 for row in id10_conflicts
    )

    # Two intentionally high-capacity/unregularized limits provide a training
    # attainability audit.  Degree 20 permits an interior crossing; the broad
    # positive exponential can approach the endpoint step g(x<1)=0,g(1)=1.
    # They are audit-only and never compete through held-out performance.
    oracle_spec = ShellSpec(
        key="oracle_bernstein_degree_20",
        label="Training-only monotone Bernstein oracle degree 20",
        kind="bernstein",
        parameter_count=19,
        degree=20,
        primary_eligible=False,
    )
    oracle_config = ShellConfig(oracle_spec, regularization=0.0)
    targets, subjects = target_maps(training)
    bernstein_oracle = fit_shell(
        oracle_config,
        training_uids,
        arrays,
        targets,
        subjects,
        loss_mode=loss_mode,
    )
    exponential_spec = next(
        spec for spec in shell_specs() if spec.key == "signed_exponential_1p"
    )
    exponential_oracle = fit_shell(
        ShellConfig(exponential_spec, regularization=0.0),
        training_uids,
        arrays,
        targets,
        subjects,
        loss_mode=loss_mode,
    )
    audit_models = [bernstein_oracle, exponential_oracle]
    oracle = min(audit_models, key=lambda fitted: fitted.data_loss)
    oracle_prediction = predict_records(oracle.spec, oracle.parameters, training_uids, arrays)
    bernstein_prediction = predict_records(
        bernstein_oracle.spec, bernstein_oracle.parameters, training_uids, arrays
    )
    exponential_prediction = predict_records(
        exponential_oracle.spec, exponential_oracle.parameters, training_uids, arrays
    )
    oracle_rows = training[
        [
            "annotation_order",
            "dream_report_id",
            "subject_id",
            "record_uid",
            "manual_final_stage",
            "consensus_mean_s",
        ]
    ].copy()
    oracle_rows["physical_modelled_s"] = [
        float(np.sum(arrays[uid][1])) for uid in training_uids
    ]
    oracle_rows["target_fraction"] = (
        oracle_rows["consensus_mean_s"] / oracle_rows["physical_modelled_s"]
    )
    oracle_rows["degree20_bernstein_apparent_s"] = bernstein_prediction
    oracle_rows["broad_exponential_apparent_s"] = exponential_prediction
    oracle_rows["oracle_apparent_s"] = oracle_prediction
    oracle_rows["oracle_family_key"] = oracle.spec.key
    oracle_rows["oracle_fraction"] = (
        oracle_rows["oracle_apparent_s"] / oracle_rows["physical_modelled_s"]
    )
    oracle_rows["residual_s"] = (
        oracle_rows["oracle_apparent_s"] - oracle_rows["consensus_mean_s"]
    )
    oracle_rows.to_csv(audit_dir / "training_monotone_oracle_predictions.csv", index=False)
    oracle_metrics = metric_summary(
        oracle_rows["consensus_mean_s"].to_numpy(float), oracle_prediction
    )
    curve_grid = np.linspace(0.0, 1.0, 2001)
    curve = pd.DataFrame(
        {
            "f_input": curve_grid,
            "identity": curve_grid,
            "oracle_g": oracle.transform(curve_grid),
            "degree20_bernstein_g": bernstein_oracle.transform(curve_grid),
            "broad_exponential_g": exponential_oracle.transform(curve_grid),
        }
    )
    curve.to_csv(audit_dir / "training_monotone_oracle_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.plot(curve_grid, curve_grid, "--", color="#7a8791", label="identity")
    ax.plot(
        curve_grid,
        curve["degree20_bernstein_g"],
        color="#5b2c6f",
        lw=2.2,
        label="degree-20 Bernstein audit",
    )
    ax.plot(
        curve_grid,
        curve["broad_exponential_g"],
        color="#b65e43",
        lw=2.0,
        label="broad exponential audit",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Physical f(t)")
    ax.set_ylabel("g(f(t))")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(audit_dir / "training_monotone_oracle_curve.png", dpi=210)
    plt.close(fig)

    payload = {
        "interpretation": (
            "The better training loss of an unregularized degree-20 Bernstein and a broad "
            "exponential endpoint-step limit is a finite-resolution attainability audit, "
            "not a generalization estimate. Exact empirical stochastic "
            "dominance supplies a family-independent impossibility result."
        ),
        "loss_mode": loss_mode,
        "oracle_model": model_payload(oracle, f_column),
        "oracle_training_metrics": oracle_metrics,
        "degree20_bernstein_model": model_payload(bernstein_oracle, f_column),
        "degree20_bernstein_training_metrics": metric_summary(
            oracle_rows["consensus_mean_s"].to_numpy(float), bernstein_prediction
        ),
        "broad_exponential_model": model_payload(exponential_oracle, f_column),
        "broad_exponential_training_metrics": metric_summary(
            oracle_rows["consensus_mean_s"].to_numpy(float), exponential_prediction
        ),
        "number_of_stochastic_dominance_pairs": len(dominance),
        "number_of_target_order_conflicts": int(dominance["target_order_conflict"].sum())
        if not dominance.empty
        else 0,
        "id10_conflicts": id10_conflicts,
        "id23_dominates_id10_conflict_found": id23_dominates_id10_conflict_found,
    }
    write_json(audit_dir / "identifiability_results.json", payload)
    report = [
        "# Global monotone-g identifiability audit",
        "",
        "The better training objective from the unregularized degree-20 Bernstein and broad",
        "exponential endpoint-step limit is reported as a finite-resolution audit. Neither is",
        "used for model selection or test prediction.",
        "",
        f"Best audit family: {oracle.spec.key}. Training RMSE: {oracle_metrics['rmse_s']:.1f} s; "
        f"log MSE: {oracle_metrics['log_mse']:.4f}; Pearson r: {oracle_metrics['pearson_r']:.3f}.",
        "",
        "The stronger result is distributional: if record A's f distribution first-order",
        "stochastically dominates record B's distribution, every global nondecreasing g",
        "must give A at least as large a mean g(f) as B.",
        "",
        f"Exact dominance conflicts involving ID10: `{json.dumps(id10_conflicts, ensure_ascii=False)}`.",
        "In particular, ID23 dominates ID10 in f, while their target fractions are about",
        "0.069 and 0.989. No single monotone g can reproduce both fractions exactly.",
        "",
    ]
    (audit_dir / "IDENTIFIABILITY.md").write_text("\n".join(report), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-csv", type=Path, required=True)
    parser.add_argument("--epoch-predictions", type=Path, required=True)
    parser.add_argument("--f-column", default="f_self_similarity")
    parser.add_argument("--stage-column", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--regularization-grid",
        default="0.001,0.01,0.1,1,10,100",
        help="Comma-separated strengths for integral (g(x)-x)^2 regularization",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("log_mse", "relative_mse", "raw_mse"),
        default="log_mse",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    consensus, training, test = load_consensus(args.consensus_csv)
    modelling = pd.concat([training, test], ignore_index=True)
    epochs, arrays = load_trajectories(args.epoch_predictions, modelling, args.f_column)
    stage_column = choose_stage_column(epochs, args.stage_column)
    regularization_grid = parse_regularization_grid(args.regularization_grid)
    configs = candidate_configs(regularization_grid)
    training_uids = set(training["record_uid"].astype(str))
    training_arrays = {uid: arrays[uid] for uid in training_uids}

    track_outputs: dict[str, dict[str, Any]] = {}
    track_models: dict[str, FittedShell] = {}
    track_controls: dict[str, FittedShell] = {}
    track_test_frames: dict[str, pd.DataFrame] = {}
    combined_final_summaries: list[pd.DataFrame] = []
    combined_final_folds: list[pd.DataFrame] = []
    combined_outer_selections: list[pd.DataFrame] = []
    combined_outer_details: list[pd.DataFrame] = []

    for track, selection_rule in (
        ("A_conservative_one_se", "one_se"),
        ("B_empirical_cv_best", "empirical_minimum"),
    ):
        nested_oof, outer_selections, outer_inner_rows = nested_training_oof(
            training,
            training_arrays,
            configs,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            loss_mode=args.loss_mode,
            selection_rule=selection_rule,
        )
        (
            selected,
            control,
            final_cv_summary,
            final_cv_folds,
            primary_selection,
            control_selection,
        ) = fit_final_models(
            training,
            training_arrays,
            configs,
            inner_splits=args.inner_splits,
            loss_mode=args.loss_mode,
            selection_rule=selection_rule,
        )
        fixed_oof = fixed_config_training_oof(
            training,
            training_arrays,
            selected,
            outer_splits=args.outer_splits,
            loss_mode=args.loss_mode,
            track=track,
        )
        # The held-aside arrays enter only after every training selection and
        # final fit above has completed.
        test_predictions = build_holdout_predictions(test, arrays, selected, control)
        nested_oof.insert(0, "track", track)
        outer_selections.insert(0, "track", track)
        outer_inner_rows.insert(0, "track", track)
        final_cv_summary.insert(0, "track", track)
        final_cv_folds.insert(0, "track", track)
        test_predictions.insert(0, "track", track)

        nested_oof.to_csv(
            args.output / f"training_nested_selection_oof_{track}.csv", index=False
        )
        fixed_oof.to_csv(
            args.output / f"training_fixed_config_oof_{track}.csv", index=False
        )
        test_predictions.to_csv(
            args.output / f"heldout_test_predictions_{track}.csv", index=False
        )
        plot_correlation(
            nested_oof,
            "selected_nested_oof_apparent_s",
            f"Kumral training: nested selection OOF ({track})",
            args.output / f"training_nested_selection_correlation_{track}.png",
        )
        plot_correlation(
            fixed_oof,
            "fixed_config_oof_apparent_s",
            f"Kumral training: fixed-config grouped OOF ({track})",
            args.output / f"training_fixed_config_correlation_{track}.png",
        )
        plot_correlation(
            test_predictions,
            "selected_apparent_s",
            f"Kumral repeated held-aside diagnostic ({track})",
            args.output / f"heldout_test_correlation_{track}.png",
        )

        nested_metrics = metric_summary(
            nested_oof["consensus_mean_s"].to_numpy(float),
            nested_oof["selected_nested_oof_apparent_s"].to_numpy(float),
        )
        fixed_metrics = metric_summary(
            fixed_oof["consensus_mean_s"].to_numpy(float),
            fixed_oof["fixed_config_oof_apparent_s"].to_numpy(float),
        )
        test_metrics = metric_summary(
            test_predictions["consensus_mean_s"].to_numpy(float),
            test_predictions["selected_apparent_s"].to_numpy(float),
        )
        feasible_test = test_predictions.loc[
            test_predictions["target_feasible_under_bounded_g"]
        ]
        test_without_infeasible = metric_summary(
            feasible_test["consensus_mean_s"].to_numpy(float),
            feasible_test["selected_apparent_s"].to_numpy(float),
        )
        id29_row = test_predictions.loc[
            test_predictions["annotation_order"].eq(29)
        ].iloc[0]
        track_models[track] = selected
        track_controls[track] = control
        track_test_frames[track] = test_predictions
        track_outputs[track] = {
            "selection_rule": selection_rule,
            "selected_model": model_payload(selected, args.f_column),
            "below_identity_control": model_payload(control, args.f_column),
            "primary_selection": primary_selection,
            "control_selection": control_selection,
            "training_nested_selection_oof": nested_metrics,
            "training_fixed_config_oof": fixed_metrics,
            "heldout_test_all_17": test_metrics,
            "heldout_test_without_infeasible_ID29": test_without_infeasible,
            "ID29": {
                "target_s": float(id29_row["consensus_mean_s"]),
                "physical_modelled_s": float(id29_row["physical_modelled_s"]),
                "prediction_s": float(id29_row["selected_apparent_s"]),
                "unavoidable_shortfall_s": float(id29_row["unavoidable_shortfall_s"]),
            },
        }
        write_json(
            args.output / f"portable_global_g_model_{track}.json",
            model_payload(selected, args.f_column),
        )
        combined_final_summaries.append(final_cv_summary)
        combined_final_folds.append(final_cv_folds)
        combined_outer_selections.append(outer_selections)
        combined_outer_details.append(outer_inner_rows)

    pd.concat(combined_final_summaries, ignore_index=True).to_csv(
        args.output / "final_training_cv_summary_both_tracks.csv", index=False
    )
    pd.concat(combined_final_folds, ignore_index=True).to_csv(
        args.output / "final_training_cv_folds_both_tracks.csv", index=False
    )
    pd.concat(combined_outer_selections, ignore_index=True).to_csv(
        args.output / "outer_fold_selections_both_tracks.csv", index=False
    )
    pd.concat(combined_outer_details, ignore_index=True).to_csv(
        args.output / "outer_inner_cv_details_both_tracks.csv", index=False
    )

    conservative = track_models["A_conservative_one_se"]
    empirical = track_models["B_empirical_cv_best"]
    empirical_control = track_controls["B_empirical_cv_best"]
    plot_g_tracks(
        conservative,
        empirical,
        empirical_control,
        args.output / "g_curves_A_B_and_controls.png",
        args.output / "g_curves_A_B_and_controls.csv",
    )
    make_test_sample_outputs(
        epochs,
        track_test_frames["B_empirical_cv_best"],
        args.f_column,
        stage_column,
        empirical,
        args.output / "test_samples_B_empirical_cv_best",
    )
    impossible = track_test_frames["B_empirical_cv_best"].loc[
        ~track_test_frames["B_empirical_cv_best"]["target_feasible_under_bounded_g"],
        ["annotation_order", "record_uid", "physical_modelled_s", "consensus_mean_s"],
    ].to_dict(orient="records")
    oracle = write_identifiability_audit(
        modelling,
        training,
        arrays,
        f_column=args.f_column,
        loss_mode=args.loss_mode,
        output=args.output,
    )
    identity_metrics = metric_summary(
        track_test_frames["B_empirical_cv_best"]["consensus_mean_s"].to_numpy(float),
        track_test_frames["B_empirical_cv_best"]["identity_apparent_s"].to_numpy(float),
    )
    result = {
        "protocol": {
            "consensus_csv": str(args.consensus_csv.resolve()),
            "epoch_predictions": str(args.epoch_predictions.resolve()),
            "f_column": args.f_column,
            "stage_column_for_figures": stage_column,
            "loss_mode": args.loss_mode,
            "subject_balanced_training_loss": True,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "regularization_grid": regularization_grid,
            "selection_tracks": ["one-standard-error", "empirical minimum CV loss"],
            "regularizer": "integral_0^1 (g(x)-x)^2 dx",
            "training_ids": list(range(6, 26)),
            "test_ids": list(range(26, 43)),
            "excluded_seed_ids": list(range(1, 6)),
            "test_status": "repeated held-aside diagnostic; never used for fitting or selection",
            "apparent_time": "sum_t g(f_t) * duration_t",
            "q_tau_used": False,
            "global_parameter_set": True,
        },
        "tracks": track_outputs,
        "heldout_identity_baseline": identity_metrics,
        "bounded_g_infeasible_targets": impossible,
        "identifiability_audit": oracle,
    }
    write_json(args.output / "fit_results.json", result)

    a = track_outputs["A_conservative_one_se"]
    b = track_outputs["B_empirical_cv_best"]
    a_model = a["selected_model"]
    b_model = b["selected_model"]
    report = [
        "# Global monotone g calibration",
        "",
        "The prediction is `apparent time = sum_t g(f_t) * duration_t`. The same endpoint-fixed",
        "monotone g applies to every record. No q_tau, record-specific term, stage predictor,",
        "or dream-text feature enters the mapping.",
        "",
        "IDs 1--5 are excluded; IDs 6--25 alone fit/select models; IDs 26--42 are a",
        "repeated held-aside diagnostic. All CV folds are grouped by participant.",
        "",
        "## Training-locked tracks",
        "",
        f"A conservative one-SE: `{a_model['family']['key']}`, lambda={a_model['regularization']}, "
        f"parameters=`{json.dumps(a_model['decoded_parameters'], sort_keys=True)}`.",
        f"B empirical CV minimum: `{b_model['family']['key']}`, lambda={b_model['regularization']}, "
        f"parameters=`{json.dumps(b_model['decoded_parameters'], sort_keys=True)}`.",
        (
            "B reaches the declared positive-kappa numerical boundary. This means the loss "
            "prefers the endpoint-step limit, not that kappa=500 is precisely identified."
            if b_model.get("positive_kappa_boundary_reached", False)
            else "B does not reach the declared exponential parameter boundary."
        ),
        "",
        "| track | nested-selection OOF r | fixed-config OOF r | test r | test r without ID29 |",
        "|---|---:|---:|---:|---:|",
        f"| A | {a['training_nested_selection_oof']['pearson_r']:.3f} | "
        f"{a['training_fixed_config_oof']['pearson_r']:.3f} | "
        f"{a['heldout_test_all_17']['pearson_r']:.3f} | "
        f"{a['heldout_test_without_infeasible_ID29']['pearson_r']:.3f} |",
        f"| B | {b['training_nested_selection_oof']['pearson_r']:.3f} | "
        f"{b['training_fixed_config_oof']['pearson_r']:.3f} | "
        f"{b['heldout_test_all_17']['pearson_r']:.3f} | "
        f"{b['heldout_test_without_infeasible_ID29']['pearson_r']:.3f} |",
        "",
        "ID29 is retained. Its 720-second target exceeds the 270 modeled seconds, so no bounded",
        "g can reach it; the without-ID29 metric is only a declared test sensitivity.",
        "",
        "The identifiability audit shows a stronger training conflict: ID23's empirical f",
        "distribution first-order stochastically dominates ID10's, yet their target fractions",
        "are approximately 0.069 and 0.989. Therefore no single monotone g can match both.",
        "",
        (
            "Epoch-wise stage shading uses `" + stage_column + "`."
            if stage_column is not None
            else "Epoch-wise sleep-stage labels are unavailable. Figures show only the manual "
            "awakening endpoint stage in the title and do not draw a false hypnogram."
        ),
        "",
    ]
    (args.output / "METHODS_AND_RESULTS.md").write_text("\n".join(report), encoding="utf-8")

    run_text = (
        "# Run the global g fit\n\n"
        "```bash\n"
        "python -m analysis.combined_fawake.fit_monotone_bernstein_g \\\n  --consensus-csv "
        + str(args.consensus_csv.resolve())
        + " \\\n  --epoch-predictions "
        + str(args.epoch_predictions.resolve())
        + " \\\n  --f-column "
        + args.f_column
        + " \\\n  --stage-column "
        + (stage_column if stage_column is not None else "none")
        + " \\\n  --loss-mode "
        + args.loss_mode
        + " \\\n  --regularization-grid "
        + ",".join(f"{value:g}" for value in regularization_grid)
        + " \\\n  --output "
        + str(args.output.resolve())
        + "\n```\n"
    )
    (args.output / "RUNNING.md").write_text(run_text, encoding="utf-8")
    code_dir = args.output / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), code_dir / Path(__file__).name)
    shutil.copy2(
        SCRIPT_ROOT / "monotone_bernstein_g.py",
        code_dir / "monotone_bernstein_g.py",
    )

    print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
