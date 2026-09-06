#!/usr/bin/env python3
"""Fit a single global endpoint-fixed monotone g using REM-gated dream time.

The physical ``f(t)`` is read unchanged from the locked adjacent-repeat v3
trajectory.  An independently produced, exploratory hard sleep-stage label is
joined one-to-one at epoch resolution.  The calibration target is

    T_dream = sum_i 1[predicted_stage_i == "REM"] * g(f_i) * duration_i.

Only Kumral IDs 6--25 participate in parameter fitting and configuration
selection.  IDs 26--42 are evaluated after the final training-only selection;
IDs 1--5 are excluded.  All cross-validation folds are grouped by participant.
The same fitted scalar g is also integrated over every epoch to report total
apparent time, but total apparent time is not part of the fitting objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.combined_fawake.fit_monotone_bernstein_g import (
    choose_empirical_minimum,
    evaluate_configs,
    grouped_splits,
    load_consensus,
    metric_summary,
    parse_regularization_grid,
    target_maps,
)
from analysis.combined_fawake.monotone_bernstein_g import (
    FittedShell,
    ShellConfig,
    candidate_configs,
    fit_shell,
    predict_records,
    transform,
    validate_shell,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONSENSUS = (
    REPO_ROOT
    / "outputs/combined_fawake_v2/kappa_fit_v1/"
    "kumral_dream_duration_6llm_consensus.csv"
)
DEFAULT_PHYSICAL = (
    REPO_ROOT
    / "outputs/dense_repeat_fawake_v3/rstar_micro_fit/all_epoch_trajectories.csv"
)
DEFAULT_STAGES = (
    REPO_ROOT
    / "outputs/sleep_staging_kumral_rem_probability_v1/"
    "pipeline_stage_trajectories.csv.gz"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/dense_repeat_fawake_v3/rem_only_g_gssc_v2_confirmed"
F_COLUMN = "f_repeat_micro_v3"
JOIN_KEYS = ["record_uid", "epoch_index", "start_s", "end_s", "duration_s"]
STAGE_COLUMNS = ["predicted_stage", "dominant_band", "p_W", "p_N1", "p_N2", "p_N3", "p_REM"]


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_epoch_join(
    physical_path: Path,
    stage_path: Path,
    records: pd.DataFrame,
    f_column: str = F_COLUMN,
) -> pd.DataFrame:
    """Join locked f and exploratory stage labels exactly, one epoch to one epoch."""

    physical = pd.read_csv(physical_path)
    stages = pd.read_csv(stage_path)
    required_physical = set(JOIN_KEYS + ["dataset", f_column])
    required_stages = set(JOIN_KEYS + ["dataset"] + STAGE_COLUMNS)
    missing_physical = sorted(required_physical - set(physical.columns))
    missing_stages = sorted(required_stages - set(stages.columns))
    if missing_physical:
        raise ValueError(f"Physical trajectory is missing columns: {missing_physical}")
    if missing_stages:
        raise ValueError(f"Stage trajectory is missing columns: {missing_stages}")

    wanted = set(records["record_uid"].astype(str))
    physical = physical.loc[
        physical["dataset"].eq("Kumral") & physical["record_uid"].isin(wanted)
    ].copy()
    stages = stages.loc[
        stages["dataset"].eq("Kumral") & stages["record_uid"].isin(wanted),
        JOIN_KEYS + STAGE_COLUMNS,
    ].copy()
    if physical.duplicated(JOIN_KEYS).any() or stages.duplicated(JOIN_KEYS).any():
        raise AssertionError("Epoch join keys must be unique in both source tables")

    joined = physical.merge(
        stages,
        on=JOIN_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    counts = joined["_merge"].value_counts().to_dict()
    if set(joined["_merge"].astype(str)) != {"both"}:
        raise AssertionError(f"Locked-f/stage epoch mismatch: {counts}")
    joined = joined.drop(columns="_merge")
    if set(joined["record_uid"].astype(str)) != wanted:
        missing = sorted(wanted - set(joined["record_uid"].astype(str)))
        raise AssertionError(f"Missing modelled records after epoch join: {missing}")

    joined[f_column] = pd.to_numeric(joined[f_column], errors="raise")
    joined["duration_s"] = pd.to_numeric(joined["duration_s"], errors="raise")
    if not np.isfinite(joined[f_column]).all() or not np.isfinite(joined["duration_s"]).all():
        raise ValueError("f and duration must be finite")
    if joined[f_column].min() < -1e-10 or joined[f_column].max() > 1.0 + 1e-10:
        raise ValueError(f"{f_column} must lie in [0, 1]")
    if (joined["duration_s"] <= 0.0).any():
        raise ValueError("Every epoch duration must be positive")
    if joined["predicted_stage"].isna().any():
        raise ValueError("Every joined epoch needs an exploratory predicted stage")
    allowed_stages = {"W", "N1", "N2", "N3", "REM"}
    unexpected = sorted(set(joined["predicted_stage"].astype(str)) - allowed_stages)
    if unexpected:
        raise ValueError(f"Unexpected predicted stages: {unexpected}")

    metadata = records.set_index("record_uid")
    for column in ("annotation_order", "dream_report_id", "subject_id", "split"):
        expected = joined["record_uid"].map(metadata[column])
        if expected.isna().any():
            raise AssertionError(f"Missing record metadata for {column}")
        # Source columns are checked rather than silently trusted.
        if column in joined.columns:
            observed = joined[column].astype(str)
            if not observed.eq(expected.astype(str)).all():
                raise AssertionError(f"Source/consensus metadata mismatch in {column}")
        joined[column] = expected.to_numpy()

    joined[f_column] = joined[f_column].clip(0.0, 1.0)
    joined["is_REM"] = joined["predicted_stage"].eq("REM").astype(np.int8)
    return joined.sort_values(["annotation_order", "epoch_index", "start_s"]).reset_index(
        drop=True
    )


def epoch_arrays(
    epochs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    rem_only: bool,
    f_column: str = F_COLUMN,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return arrays for every record, retaining explicit empty REM sequences."""

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    grouped = {str(uid): group for uid, group in epochs.groupby("record_uid", sort=False)}
    for uid in records["record_uid"].astype(str):
        group = grouped[uid]
        if rem_only:
            group = group.loc[group["is_REM"].eq(1)]
        arrays[uid] = (
            group[f_column].to_numpy(dtype=float),
            group["duration_s"].to_numpy(dtype=float),
        )
    return arrays


def eligible_configs(regularization_grid: Sequence[float]) -> list[ShellConfig]:
    """Use endpoint-fixed monotone candidates with zero through four parameters."""

    return [
        config
        for config in candidate_configs(regularization_grid)
        if config.spec.primary_eligible and config.spec.parameter_count <= 4
    ]


def _fit_one(
    config: ShellConfig,
    fitting_records: pd.DataFrame,
    rem_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    all_training: pd.DataFrame,
) -> FittedShell:
    targets, subjects = target_maps(all_training)
    return fit_shell(
        config,
        fitting_records["record_uid"].astype(str).tolist(),
        rem_arrays,
        targets,
        subjects,
        loss_mode="log_mse",
    )


def nested_training_oof(
    training: pd.DataFrame,
    rem_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    outer_splits: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested, participant-grouped OOF estimates with selection inside each fold."""

    by_key = {config.key: config for config in configs}
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    inner_summary_rows: list[dict[str, Any]] = []
    inner_fold_rows: list[dict[str, Any]] = []
    for outer_fold, (fit_idx, validation_idx) in enumerate(
        grouped_splits(training, outer_splits), start=1
    ):
        fit_records = training.iloc[fit_idx].reset_index(drop=True)
        validation = training.iloc[validation_idx].reset_index(drop=True)
        summaries, fold_rows = evaluate_configs(
            fit_records,
            rem_arrays,
            configs,
            requested_splits=inner_splits,
            loss_mode="log_mse",
            context=f"outer_{outer_fold}_inner_training_only",
        )
        selected_config, details = choose_empirical_minimum(
            summaries, by_key, primary_only=True
        )
        fitted = _fit_one(selected_config, fit_records, rem_arrays, training)
        validation_uids = validation["record_uid"].astype(str).tolist()
        dream = predict_records(
            fitted.spec, fitted.parameters, validation_uids, rem_arrays
        )
        apparent = predict_records(
            fitted.spec, fitted.parameters, validation_uids, full_arrays
        )
        for record, dream_s, apparent_s in zip(
            validation.itertuples(index=False), dream, apparent, strict=True
        ):
            uid = str(record.record_uid)
            rem_duration = float(np.sum(rem_arrays[uid][1]))
            prediction_rows.append(
                {
                    "outer_fold": outer_fold,
                    "annotation_order": int(record.annotation_order),
                    "dream_report_id": record.dream_report_id,
                    "split": "train",
                    "subject_id": str(record.subject_id),
                    "record_uid": uid,
                    "consensus_mean_s": float(record.consensus_mean_s),
                    "predicted_dream_s": float(dream_s),
                    "apparent_time_s": float(apparent_s),
                    "predicted_REM_duration_s": rem_duration,
                    "target_feasible_under_REM_and_bounded_g": bool(
                        float(record.consensus_mean_s) <= rem_duration + 1e-9
                    ),
                    "selected_config_key": selected_config.key,
                    "selected_parameters_json": json.dumps(
                        fitted.decoded_parameters(), sort_keys=True
                    ),
                    "protocol": (
                        "conditional nested participant-grouped OOF for g only; "
                        "g selection and fit exclude this subject, while frozen upstream "
                        "physical-f and stage predictions are not re-cross-fitted here"
                    ),
                }
            )
        fit_subjects = sorted(set(fit_records["subject_id"].astype(str)))
        validation_subjects = sorted(set(validation["subject_id"].astype(str)))
        if set(fit_subjects) & set(validation_subjects):
            raise AssertionError("Participant leakage in outer fold")
        selection_rows.append(
            {
                "outer_fold": outer_fold,
                "fit_subject_ids": ";".join(fit_subjects),
                "validation_subject_ids": ";".join(validation_subjects),
                "selected_config_key": selected_config.key,
                "selected_family_key": selected_config.spec.key,
                "parameter_count": selected_config.spec.parameter_count,
                "regularization": selected_config.regularization,
                "parameters_json": json.dumps(fitted.decoded_parameters(), sort_keys=True),
                "selection_details_json": json.dumps(details, sort_keys=True),
            }
        )
        inner_summary_rows.extend({"outer_fold": outer_fold, **row} for row in summaries)
        inner_fold_rows.extend({"outer_fold": outer_fold, **row} for row in fold_rows)

    predictions = pd.DataFrame(prediction_rows).sort_values("annotation_order").reset_index(
        drop=True
    )
    if predictions["annotation_order"].tolist() != list(range(6, 26)):
        raise AssertionError("Nested OOF must cover training IDs 6--25 exactly once")
    return (
        predictions,
        pd.DataFrame(selection_rows),
        pd.DataFrame(inner_summary_rows),
        pd.DataFrame(inner_fold_rows),
    )


def fit_final_model(
    training: pd.DataFrame,
    rem_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    inner_splits: int,
) -> tuple[FittedShell, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries, folds = evaluate_configs(
        training,
        rem_arrays,
        configs,
        requested_splits=inner_splits,
        loss_mode="log_mse",
        context="full_training_selection",
    )
    by_key = {config.key: config for config in configs}
    selected_config, details = choose_empirical_minimum(summaries, by_key, primary_only=True)
    fitted = _fit_one(selected_config, training, rem_arrays, training)
    return fitted, pd.DataFrame(summaries), pd.DataFrame(folds), details


def fixed_config_oof(
    training: pd.DataFrame,
    rem_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    final_model: FittedShell,
    outer_splits: int,
) -> pd.DataFrame:
    """Estimate parameter stability after the config was selected on all training data."""

    config = ShellConfig(final_model.spec, final_model.regularization)
    rows: list[dict[str, Any]] = []
    for fold, (fit_idx, validation_idx) in enumerate(
        grouped_splits(training, outer_splits), start=1
    ):
        fit_records = training.iloc[fit_idx]
        validation = training.iloc[validation_idx]
        fitted = _fit_one(config, fit_records, rem_arrays, training)
        uids = validation["record_uid"].astype(str).tolist()
        dreams = predict_records(fitted.spec, fitted.parameters, uids, rem_arrays)
        apparent = predict_records(fitted.spec, fitted.parameters, uids, full_arrays)
        for record, dream_s, apparent_s in zip(
            validation.itertuples(index=False), dreams, apparent, strict=True
        ):
            rows.append(
                {
                    "outer_fold": fold,
                    "annotation_order": int(record.annotation_order),
                    "dream_report_id": record.dream_report_id,
                    "split": "train",
                    "subject_id": str(record.subject_id),
                    "record_uid": str(record.record_uid),
                    "consensus_mean_s": float(record.consensus_mean_s),
                    "predicted_dream_s": float(dream_s),
                    "apparent_time_s": float(apparent_s),
                    "config_key": config.key,
                    "parameters_json": json.dumps(fitted.decoded_parameters(), sort_keys=True),
                    "protocol": "fixed-config participant-grouped OOF; config selected using all training CV",
                }
            )
    return pd.DataFrame(rows).sort_values("annotation_order").reset_index(drop=True)


def record_predictions(
    records: pd.DataFrame,
    rem_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    model: FittedShell,
    protocol: str,
) -> pd.DataFrame:
    uids = records["record_uid"].astype(str).tolist()
    dream = predict_records(model.spec, model.parameters, uids, rem_arrays)
    apparent = predict_records(model.spec, model.parameters, uids, full_arrays)
    rows: list[dict[str, Any]] = []
    for record, dream_s, apparent_s in zip(
        records.itertuples(index=False), dream, apparent, strict=True
    ):
        uid = str(record.record_uid)
        rem_s = float(np.sum(rem_arrays[uid][1]))
        physical_s = float(np.sum(full_arrays[uid][1]))
        target_s = float(record.consensus_mean_s)
        rows.append(
            {
                "annotation_order": int(record.annotation_order),
                "dream_report_id": record.dream_report_id,
                "split": str(record.split),
                "subject_id": str(record.subject_id),
                "record_uid": uid,
                "eeg_filename": record.eeg_filename,
                "manual_final_stage": record.manual_final_stage,
                "physical_modelled_s": physical_s,
                "predicted_REM_duration_s": rem_s,
                "consensus_mean_s": target_s,
                "consensus_sigma_population_s": float(record.consensus_sigma_population_s),
                "target_feasible_under_REM_and_bounded_g": bool(target_s <= rem_s + 1e-9),
                "unavoidable_REM_shortfall_s": max(target_s - rem_s, 0.0),
                "predicted_dream_s": float(dream_s),
                "dream_residual_s": float(dream_s - target_s),
                "apparent_time_s": float(apparent_s),
                "config_key": f"{model.spec.key}__lambda_{model.regularization:g}",
                "protocol": protocol,
            }
        )
    return pd.DataFrame(rows)


def add_final_epoch_outputs(
    epochs: pd.DataFrame,
    model: FittedShell,
    records: pd.DataFrame,
    f_column: str = F_COLUMN,
) -> pd.DataFrame:
    result = epochs.copy()
    result["bar_f_lambda"] = transform(
        model.spec, result[f_column].to_numpy(dtype=float), model.parameters
    )
    result["apparent_increment_s"] = result["bar_f_lambda"] * result["duration_s"]
    result["dream_increment_s"] = (
        result["is_REM"] * result["bar_f_lambda"] * result["duration_s"]
    )
    result["cumulative_apparent_s"] = result.groupby("record_uid", sort=False)[
        "apparent_increment_s"
    ].cumsum()
    result["cumulative_dream_s"] = result.groupby("record_uid", sort=False)[
        "dream_increment_s"
    ].cumsum()
    targets = records.set_index("record_uid")["consensus_mean_s"].astype(float)
    result["consensus_mean_s"] = result["record_uid"].map(targets)
    result["g_config_key"] = f"{model.spec.key}__lambda_{model.regularization:g}"
    result["stage_role"] = (
        "exploratory hard prediction; REM gates dream time; not a manual hypnogram"
    )
    return result


def metrics_for(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    return metric_summary(
        frame["consensus_mean_s"].to_numpy(dtype=float),
        frame[prediction].to_numpy(dtype=float),
    )


def portable_payload(
    model: FittedShell,
    args: argparse.Namespace,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tpmt.rem_only_global_g.v1",
        "description": "One global endpoint-fixed monotone g calibrated to REM-gated dream time",
        "formula": "T_dream = sum 1[predicted_stage == REM] * g(f_repeat_micro_v3) * duration_s",
        "apparent_time_formula": "T_apparent = sum g(f_repeat_micro_v3) * duration_s",
        "physical_f_refit": False,
        "stage_status": "exploratory predicted hard label, not manual hypnogram",
        "stage_used_as_g_input": False,
        "stage_used_as_dream_mask": True,
        "training_ids": list(range(6, 26)),
        "heldout_ids": list(range(26, 43)),
        "excluded_ids": list(range(1, 6)),
        "selection": "minimum mean participant-grouped CV log-MSE on training only",
        "g_only_cv_conditional_on_frozen_upstream": True,
        "upstream_stage_cross_fitted_within_g_cv": False,
        "family": {
            "key": model.spec.key,
            "label": model.spec.label,
            "kind": model.spec.kind,
            "degree": model.spec.degree,
            "parameter_count": model.spec.parameter_count,
        },
        "regularization": model.regularization,
        "parameters": [float(value) for value in model.parameters],
        "decoded_parameters": model.decoded_parameters(),
        "validation": dict(validation),
        "inputs": {
            "physical_f_column": args.f_column,
            "stage_column": "predicted_stage",
            "rem_label": "REM",
        },
        "source_files": {
            "consensus_csv": str(args.consensus_csv.resolve()),
            "consensus_sha256": sha256(args.consensus_csv),
            "locked_physical_epochs": str(args.physical_epochs.resolve()),
            "locked_physical_epochs_sha256": sha256(args.physical_epochs),
            "exploratory_stage_epochs": str(args.stage_epochs.resolve()),
            "exploratory_stage_epochs_sha256": sha256(args.stage_epochs),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus-csv", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--physical-epochs", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--stage-epochs", type=Path, default=DEFAULT_STAGES)
    parser.add_argument("--f-column", default=F_COLUMN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--regularization-grid",
        default="0,0.001,0.01,0.1,1,10",
        help="Comma-separated weights for integral (g(x)-x)^2",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    consensus, training, test = load_consensus(args.consensus_csv)
    records = pd.concat([training, test], ignore_index=True)
    epochs = load_epoch_join(
        args.physical_epochs, args.stage_epochs, records, f_column=args.f_column
    )
    rem_arrays = epoch_arrays(
        epochs, records, rem_only=True, f_column=args.f_column
    )
    full_arrays = epoch_arrays(
        epochs, records, rem_only=False, f_column=args.f_column
    )
    regularization_grid = parse_regularization_grid(args.regularization_grid)
    configs = eligible_configs(regularization_grid)

    nested_oof, outer_selections, outer_inner_summary, outer_inner_folds = (
        nested_training_oof(
            training,
            rem_arrays,
            full_arrays,
            configs,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
        )
    )
    final_model, final_cv_summary, final_cv_folds, selection_details = fit_final_model(
        training, rem_arrays, configs, inner_splits=args.inner_splits
    )
    fixed_oof = fixed_config_oof(
        training, rem_arrays, full_arrays, final_model, outer_splits=args.outer_splits
    )
    train_final = record_predictions(
        training,
        rem_arrays,
        full_arrays,
        final_model,
        protocol="final global g fitted on all training records; in-sample training description",
    )
    test_final = record_predictions(
        test,
        rem_arrays,
        full_arrays,
        final_model,
        protocol="held-aside test; global g locked from IDs 6--25",
    )
    per_epoch = add_final_epoch_outputs(epochs, final_model, records, f_column=args.f_column)

    nested_oof.to_csv(args.output / "training_nested_oof_predictions.csv", index=False)
    fixed_oof.to_csv(args.output / "training_fixed_config_oof_predictions.csv", index=False)
    train_final.to_csv(args.output / "training_final_model_in_sample.csv", index=False)
    test_final.to_csv(args.output / "heldout_test_predictions.csv", index=False)
    pd.concat([train_final, test_final], ignore_index=True).to_csv(
        args.output / "all_final_model_record_predictions.csv", index=False
    )
    per_epoch.to_csv(args.output / "all_epoch_trajectories.csv", index=False)
    outer_selections.to_csv(args.output / "outer_fold_selections.csv", index=False)
    outer_inner_summary.to_csv(args.output / "outer_inner_cv_summaries.csv", index=False)
    outer_inner_folds.to_csv(args.output / "outer_inner_cv_folds.csv", index=False)
    final_cv_summary.to_csv(args.output / "final_training_cv_summary.csv", index=False)
    final_cv_folds.to_csv(args.output / "final_training_cv_folds.csv", index=False)

    grid = np.linspace(0.0, 1.0, 2001)
    curve = pd.DataFrame(
        {"f_repeat_micro_v3": grid, "identity": grid, "g_of_f": final_model.transform(grid)}
    )
    curve.to_csv(args.output / "g_curve.csv", index=False)
    validation = validate_shell(final_model.spec, final_model.parameters)
    portable = portable_payload(final_model, args, validation)
    write_json(args.output / "portable_rem_only_global_g.json", portable)

    metrics = {
        "dream_time": {
            "training_nested_selection_oof": metrics_for(nested_oof, "predicted_dream_s"),
            "training_fixed_config_oof": metrics_for(fixed_oof, "predicted_dream_s"),
            "training_final_model_in_sample": metrics_for(train_final, "predicted_dream_s"),
            "heldout_test": metrics_for(test_final, "predicted_dream_s"),
        },
        "apparent_time_vs_dream_estimate": {
            "training_nested_selection_oof": metrics_for(nested_oof, "apparent_time_s"),
            "training_fixed_config_oof": metrics_for(fixed_oof, "apparent_time_s"),
            "training_final_model_in_sample": metrics_for(train_final, "apparent_time_s"),
            "heldout_test": metrics_for(test_final, "apparent_time_s"),
        },
    }
    feasibility = {
        "training_records": len(training),
        "training_with_any_predicted_REM": int(
            (train_final["predicted_REM_duration_s"] > 0).sum()
        ),
        "training_target_feasible": int(
            train_final["target_feasible_under_REM_and_bounded_g"].sum()
        ),
        "test_records": len(test),
        "test_with_any_predicted_REM": int((test_final["predicted_REM_duration_s"] > 0).sum()),
        "test_target_feasible": int(
            test_final["target_feasible_under_REM_and_bounded_g"].sum()
        ),
    }
    fit_results = {
        "protocol": portable,
        "selected_configuration": {
            "config_key": f"{final_model.spec.key}__lambda_{final_model.regularization:g}",
            "objective": final_model.objective,
            "data_loss": final_model.data_loss,
            "identity_penalty": final_model.identity_penalty,
            "selection_details": selection_details,
        },
        "epoch_join": {
            "joined_epochs": len(epochs),
            "records": epochs["record_uid"].nunique(),
            "REM_epochs": int(epochs["is_REM"].sum()),
            "non_REM_epochs": int((1 - epochs["is_REM"]).sum()),
            "join_keys": JOIN_KEYS,
            "exact_one_to_one": True,
        },
        "feasibility": feasibility,
        "metrics": metrics,
    }
    write_json(args.output / "fit_results.json", fit_results)

    train_metric = metrics["dream_time"]["training_nested_selection_oof"]
    test_metric = metrics["dream_time"]["heldout_test"]
    methods = rf"""# REM-only global g calibration

## Definition

The locked physical quantity `{args.f_column}` is never refitted.  The exploratory
epoch-wise stage table is joined to it exactly on `{', '.join(JOIN_KEYS)}`.  Dream
time is defined as

\[
T_{{dream}}=\sum_i {1}[\widehat{{stage}}_i=REM]\,g(f_i)\,\Delta t_i,
\]

whereas total apparent time is reported separately as

\[
T_{{apparent}}=\sum_i g(f_i)\,\Delta t_i.
\]

The stage labels are exploratory classifier predictions, not manual epoch-wise
hypnograms.  They gate the REM-only dream integral but do not modify physical f.

## Leakage controls and fitting

* IDs 1--5: excluded human calibration seeds.
* IDs 6--25: training and model selection only.
* IDs 26--42: held-aside diagnostic evaluation after locking g.
* Every outer and inner fold is grouped by participant.
* The nested OOF result is conditional validation of `g`: its selection and fitting
  exclude each validation participant, but the already frozen upstream physical-f
  and exploratory stage models are not re-cross-fitted inside this run.  It must not
  be described as fully end-to-end OOF performance.
* Candidate g families have zero through four free parameters and are monotone,
  bounded in [0,1], with exact endpoints g(0)=0 and g(1)=1.
* Selection minimizes participant-balanced inner-CV log-MSE of REM-gated dream
  time.  Raw-second MAE/RMSE and correlations are reported without entering the
  selection rule.

Selected configuration: `{final_model.spec.key}__lambda_{final_model.regularization:g}`.
Decoded parameters: `{json.dumps(final_model.decoded_parameters(), sort_keys=True)}`.

## Results

Nested training OOF dream-time result: n={train_metric['n']}, RMSE={train_metric['rmse_s']:.2f} s,
MAE={train_metric['mae_s']:.2f} s, Pearson r={train_metric['pearson_r']:.4f},
Spearman rho={train_metric['spearman_rho']:.4f}.

Held-aside test dream-time result: n={test_metric['n']}, RMSE={test_metric['rmse_s']:.2f} s,
MAE={test_metric['mae_s']:.2f} s, Pearson r={test_metric['pearson_r']:.4f},
Spearman rho={test_metric['spearman_rho']:.4f}.

REM-duration feasibility is a hard upper bound because 0 <= g <= 1.  Only
{feasibility['training_target_feasible']}/{feasibility['training_records']} training and
{feasibility['test_target_feasible']}/{feasibility['test_records']} test targets are at or below
their predicted REM duration.  This limitation is preserved rather than hidden.
"""
    (args.output / "METHODS_AND_RESULTS.md").write_text(methods, encoding="utf-8")
    running = f"""# Reproduce REM-only g calibration

```bash
cd {REPO_ROOT}
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \\
  .venv/bin/python -m analysis.combined_fawake.fit_rem_only_global_g
```

The command writes only to `{args.output}` and does not alter the locked physical-f
or exploratory-stage source files.
"""
    (args.output / "RUNNING.md").write_text(running, encoding="utf-8")

    print(json.dumps(_json_value({
        "output": args.output,
        "selected_config": f"{final_model.spec.key}__lambda_{final_model.regularization:g}",
        "parameters": final_model.decoded_parameters(),
        "training_nested_oof": train_metric,
        "heldout_test": test_metric,
        "feasibility": feasibility,
    }), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
