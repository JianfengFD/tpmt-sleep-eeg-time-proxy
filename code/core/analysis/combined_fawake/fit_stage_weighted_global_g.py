#!/usr/bin/env python3
"""Fit a global monotone ``g`` to stage-probability-weighted dream time.

The locked physical trajectory ``f(t)`` and exploratory hard GSSC sleep-stage
prediction are joined exactly as in the prior REM-only analysis.  This analysis
is deliberately separate from that historical result.  Its calibration target is

    T_dream = sum_i p(predicted_stage_i) * g(f_i) * duration_i.

The five fixed stage weights are command-line configurable and are recorded in
every portable result. The main literature-mean profile is W=0, N1=0.849,
N2=0.532, N3=0.508, and REM=0.834; the originally requested heuristic profile
remains available as a sensitivity analysis.
Kumral annotation-order IDs 10 and 19 are omitted from training; IDs 29 and 33
are omitted from the held-aside test because their recordings are too short.
Only the remaining training records select and fit ``g``.  Test targets are not
used until the locked model is evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from analysis.combined_fawake.fit_monotone_bernstein_g import (
    choose_empirical_minimum,
    evaluate_configs,
    grouped_splits,
    load_consensus,
    parse_regularization_grid,
    target_maps,
)
from analysis.combined_fawake.fit_rem_only_global_g import (
    DEFAULT_CONSENSUS,
    DEFAULT_PHYSICAL,
    DEFAULT_STAGES,
    F_COLUMN,
    JOIN_KEYS,
    REPO_ROOT,
    _json_value,
    eligible_configs,
    epoch_arrays,
    load_epoch_join,
    metrics_for,
    sha256,
    write_json,
)
from analysis.combined_fawake.monotone_bernstein_g import (
    FittedShell,
    ShellConfig,
    fit_shell,
    predict_records,
    transform,
    validate_shell,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/dense_repeat_fawake_v3/stage_weighted_g_literature_v1"
STAGE_ORDER = ("W", "N1", "N2", "N3", "REM")
LITERATURE_MEAN_STAGE_WEIGHTS = {
    "W": 0.0,
    "N1": 0.849,
    "N2": 0.532,
    "N3": 0.508,
    "REM": 0.834,
}
REQUESTED_HEURISTIC_STAGE_WEIGHTS = {
    "W": 0.0,
    "N1": 0.35,
    "N2": 0.15,
    "N3": 0.0,
    "REM": 0.85,
}
# Backward-friendly name for callers that only need the analysis default.
DEFAULT_STAGE_WEIGHTS = LITERATURE_MEAN_STAGE_WEIGHTS
TRAIN_EXCLUDED_IDS = (10, 19)
TEST_EXCLUDED_IDS = (29, 33)
TRAIN_IDS = tuple(item for item in range(6, 26) if item not in TRAIN_EXCLUDED_IDS)
TEST_IDS = tuple(item for item in range(26, 43) if item not in TEST_EXCLUDED_IDS)


def parse_stage_weights(text: str) -> dict[str, float]:
    """Parse and validate ``STAGE=value`` pairs for all five sleep stages."""

    weights: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "--stage-weights must use STAGE=value pairs, for example "
                "W=0,N1=0.35,N2=0.15,N3=0,REM=0.85"
            )
        stage, raw_value = item.split("=", 1)
        stage = stage.strip().upper()
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown sleep stage in --stage-weights: {stage}")
        if stage in weights:
            raise ValueError(f"Duplicate sleep stage in --stage-weights: {stage}")
        value = float(raw_value.strip())
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Stage weight for {stage} must be finite and in [0, 1]")
        weights[stage] = value
    missing = [stage for stage in STAGE_ORDER if stage not in weights]
    if missing:
        raise ValueError(f"--stage-weights is missing stages: {missing}")
    if not any(value > 0.0 for value in weights.values()):
        raise ValueError("At least one stage weight must be positive")
    return {stage: weights[stage] for stage in STAGE_ORDER}


def resolve_stage_weights(
    profile: str,
    custom_text: str | None,
) -> tuple[str, dict[str, float]]:
    """Resolve a named reproducible profile or an explicit CLI override."""

    if custom_text is not None:
        return "custom", parse_stage_weights(custom_text)
    if profile == "literature_mean":
        return profile, dict(LITERATURE_MEAN_STAGE_WEIGHTS)
    if profile == "requested_heuristic":
        return profile, dict(REQUESTED_HEURISTIC_STAGE_WEIGHTS)
    raise ValueError("--weight-profile=custom requires --stage-weights")


def select_analysis_records(
    training: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the prespecified short-record exclusions without changing splits."""

    selected_training = training.loc[
        ~training["annotation_order"].isin(TRAIN_EXCLUDED_IDS)
    ].copy()
    selected_test = test.loc[~test["annotation_order"].isin(TEST_EXCLUDED_IDS)].copy()
    selected_training = selected_training.sort_values("annotation_order").reset_index(drop=True)
    selected_test = selected_test.sort_values("annotation_order").reset_index(drop=True)
    if selected_training["annotation_order"].astype(int).tolist() != list(TRAIN_IDS):
        raise AssertionError("Stage-weighted training IDs do not match the prespecified set")
    if selected_test["annotation_order"].astype(int).tolist() != list(TEST_IDS):
        raise AssertionError("Stage-weighted test IDs do not match the prespecified set")
    return selected_training, selected_test


def stage_weighted_epoch_arrays(
    epochs: pd.DataFrame,
    records: pd.DataFrame,
    stage_weights: Mapping[str, float],
    f_column: str = F_COLUMN,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``f`` and positive effective durations ``p(stage)*duration``.

    Epochs with zero stage probability are absent from the compact fitting
    arrays, which is exactly equivalent to retaining them with zero contribution.
    Every requested record remains present, including a possible empty sequence.
    """

    if set(stage_weights) != set(STAGE_ORDER):
        raise ValueError(f"stage_weights must contain exactly {list(STAGE_ORDER)}")
    for stage, value in stage_weights.items():
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Invalid stage weight for {stage}: {value}")

    grouped = {str(uid): group for uid, group in epochs.groupby("record_uid", sort=False)}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for uid in records["record_uid"].astype(str):
        group = grouped[uid]
        probabilities = group["predicted_stage"].astype(str).map(stage_weights)
        if probabilities.isna().any():
            unexpected = sorted(set(group.loc[probabilities.isna(), "predicted_stage"].astype(str)))
            raise ValueError(f"No stage weight for predicted stages: {unexpected}")
        effective_duration = (
            group["duration_s"].to_numpy(dtype=float)
            * probabilities.to_numpy(dtype=float)
        )
        active = effective_duration > 0.0
        arrays[uid] = (
            group.loc[active, f_column].to_numpy(dtype=float),
            effective_duration[active],
        )
    return arrays


def _fit_one(
    config: ShellConfig,
    fitting_records: pd.DataFrame,
    dream_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    all_training: pd.DataFrame,
) -> FittedShell:
    targets, subjects = target_maps(all_training)
    return fit_shell(
        config,
        fitting_records["record_uid"].astype(str).tolist(),
        dream_arrays,
        targets,
        subjects,
        loss_mode="log_mse",
    )


def nested_training_oof(
    training: pd.DataFrame,
    dream_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    outer_splits: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nested participant-grouped OOF estimates with selection inside each fold."""

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
            dream_arrays,
            configs,
            requested_splits=inner_splits,
            loss_mode="log_mse",
            context=f"outer_{outer_fold}_inner_training_only",
        )
        selected_config, details = choose_empirical_minimum(
            summaries, by_key, primary_only=True
        )
        fitted = _fit_one(selected_config, fit_records, dream_arrays, training)
        validation_uids = validation["record_uid"].astype(str).tolist()
        dream = predict_records(fitted.spec, fitted.parameters, validation_uids, dream_arrays)
        apparent = predict_records(fitted.spec, fitted.parameters, validation_uids, full_arrays)
        for record, dream_s, apparent_s in zip(
            validation.itertuples(index=False), dream, apparent, strict=True
        ):
            uid = str(record.record_uid)
            maximum_s = float(np.sum(dream_arrays[uid][1]))
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
                    "stage_weighted_maximum_s": maximum_s,
                    "target_feasible_under_stage_weights_and_bounded_g": bool(
                        float(record.consensus_mean_s) <= maximum_s + 1e-9
                    ),
                    "selected_config_key": selected_config.key,
                    "selected_parameters_json": json.dumps(
                        fitted.decoded_parameters(), sort_keys=True
                    ),
                    "protocol": (
                        "conditional nested participant-grouped OOF for g only; "
                        "g selection and fit exclude this subject, while frozen upstream "
                        "physical-f and GSSC stage predictions are not re-cross-fitted here"
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
    if predictions["annotation_order"].astype(int).tolist() != list(TRAIN_IDS):
        raise AssertionError("Nested OOF must cover each selected training ID exactly once")
    return (
        predictions,
        pd.DataFrame(selection_rows),
        pd.DataFrame(inner_summary_rows),
        pd.DataFrame(inner_fold_rows),
    )


def fit_final_model(
    training: pd.DataFrame,
    dream_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    configs: Sequence[ShellConfig],
    inner_splits: int,
) -> tuple[FittedShell, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summaries, folds = evaluate_configs(
        training,
        dream_arrays,
        configs,
        requested_splits=inner_splits,
        loss_mode="log_mse",
        context="full_training_selection",
    )
    by_key = {config.key: config for config in configs}
    selected_config, details = choose_empirical_minimum(summaries, by_key, primary_only=True)
    fitted = _fit_one(selected_config, training, dream_arrays, training)
    return fitted, pd.DataFrame(summaries), pd.DataFrame(folds), details


def fixed_config_oof(
    training: pd.DataFrame,
    dream_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    final_model: FittedShell,
    outer_splits: int,
) -> pd.DataFrame:
    """Estimate parameter stability after selecting the configuration on training."""

    config = ShellConfig(final_model.spec, final_model.regularization)
    rows: list[dict[str, Any]] = []
    for fold, (fit_idx, validation_idx) in enumerate(
        grouped_splits(training, outer_splits), start=1
    ):
        fit_records = training.iloc[fit_idx]
        validation = training.iloc[validation_idx]
        fitted = _fit_one(config, fit_records, dream_arrays, training)
        uids = validation["record_uid"].astype(str).tolist()
        dreams = predict_records(fitted.spec, fitted.parameters, uids, dream_arrays)
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
                    "protocol": (
                        "fixed-config participant-grouped OOF; configuration selected "
                        "using all selected training records"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("annotation_order").reset_index(drop=True)


def _stage_duration_fields(epochs: pd.DataFrame) -> dict[str, float]:
    durations = epochs.groupby("predicted_stage")["duration_s"].sum().to_dict()
    return {
        f"predicted_{stage}_duration_s": float(durations.get(stage, 0.0))
        for stage in STAGE_ORDER
    }


def record_predictions(
    records: pd.DataFrame,
    epochs: pd.DataFrame,
    dream_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    full_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    model: FittedShell,
    stage_weights: Mapping[str, float],
    protocol: str,
) -> pd.DataFrame:
    uids = records["record_uid"].astype(str).tolist()
    dream = predict_records(model.spec, model.parameters, uids, dream_arrays)
    apparent = predict_records(model.spec, model.parameters, uids, full_arrays)
    epoch_groups = {str(uid): group for uid, group in epochs.groupby("record_uid", sort=False)}
    rows: list[dict[str, Any]] = []
    for record, dream_s, apparent_s in zip(
        records.itertuples(index=False), dream, apparent, strict=True
    ):
        uid = str(record.record_uid)
        maximum_s = float(np.sum(dream_arrays[uid][1]))
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
                **_stage_duration_fields(epoch_groups[uid]),
                "stage_weighted_maximum_s": maximum_s,
                "consensus_mean_s": target_s,
                "consensus_sigma_population_s": float(record.consensus_sigma_population_s),
                "target_feasible_under_stage_weights_and_bounded_g": bool(
                    target_s <= maximum_s + 1e-9
                ),
                "unavoidable_stage_weighted_shortfall_s": max(target_s - maximum_s, 0.0),
                "predicted_dream_s": float(dream_s),
                "dream_residual_s": float(dream_s - target_s),
                "apparent_time_s": float(apparent_s),
                "stage_weights_json": json.dumps(stage_weights, sort_keys=True),
                "config_key": f"{model.spec.key}__lambda_{model.regularization:g}",
                "protocol": protocol,
            }
        )
    return pd.DataFrame(rows)


def add_final_epoch_outputs(
    epochs: pd.DataFrame,
    model: FittedShell,
    records: pd.DataFrame,
    stage_weights: Mapping[str, float],
    f_column: str = F_COLUMN,
) -> pd.DataFrame:
    result = epochs.copy()
    result["stage_dream_probability"] = (
        result["predicted_stage"].astype(str).map(stage_weights).astype(float)
    )
    result["dream_weight"] = result["stage_dream_probability"]
    result["bar_f_lambda"] = transform(
        model.spec, result[f_column].to_numpy(dtype=float), model.parameters
    )
    result["apparent_increment_s"] = result["bar_f_lambda"] * result["duration_s"]
    result["weighted_dream_increment_s"] = (
        result["stage_dream_probability"]
        * result["bar_f_lambda"]
        * result["duration_s"]
    )
    result["cumulative_apparent_s"] = result.groupby("record_uid", sort=False)[
        "apparent_increment_s"
    ].cumsum()
    result["cumulative_weighted_dream_s"] = result.groupby("record_uid", sort=False)[
        "weighted_dream_increment_s"
    ].cumsum()
    targets = records.set_index("record_uid")["consensus_mean_s"].astype(float)
    result["consensus_mean_s"] = result["record_uid"].map(targets)
    result["g_config_key"] = f"{model.spec.key}__lambda_{model.regularization:g}"
    result["stage_weights_json"] = json.dumps(stage_weights, sort_keys=True)
    result["stage_role"] = (
        "exploratory GSSC hard prediction mapped to fixed dream probabilities; "
        "not a manual epoch-wise hypnogram"
    )
    return result


def portable_payload(
    model: FittedShell,
    args: argparse.Namespace,
    weight_profile: str,
    stage_weights: Mapping[str, float],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "tpmt.stage_weighted_global_g.v1",
        "description": "One global endpoint-fixed monotone g calibrated to stage-weighted dream time",
        "formula": (
            "T_dream = sum p(predicted_stage) * g(f_repeat_micro_v3) * duration_s"
        ),
        "apparent_time_formula": "T_apparent = sum g(f_repeat_micro_v3) * duration_s",
        "stage_weights": dict(stage_weights),
        "stage_weight_profile": weight_profile,
        "stage_weights_fixed_during_g_optimization": True,
        "physical_f_refit": False,
        "stage_status": (
            "exploratory hard prediction from pretrained GSSC v0.0.9; "
            "not a manual epoch-wise hypnogram"
        ),
        "stage_used_as_g_input": False,
        "stage_used_as_dream_weight": True,
        "training_ids": list(TRAIN_IDS),
        "heldout_ids": list(TEST_IDS),
        "excluded_ids": {
            "human_calibration_seeds": list(range(1, 6)),
            "training_too_short": list(TRAIN_EXCLUDED_IDS),
            "test_too_short": list(TEST_EXCLUDED_IDS),
        },
        "selection": "minimum mean participant-grouped CV log-MSE on selected training only",
        "test_used_for_selection_or_fitting": False,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to the literature main directory; the "
            "requested-heuristic profile defaults to a sensitivity subdirectory."
        ),
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--regularization-grid",
        default="0,0.001,0.01,0.1,1,10",
        help="Comma-separated weights for integral (g(x)-x)^2",
    )
    parser.add_argument(
        "--weight-profile",
        choices=("literature_mean", "requested_heuristic", "custom"),
        default="literature_mean",
        help=(
            "Named fixed weights. literature_mean is the main analysis; "
            "requested_heuristic reproduces W=0,N1=.35,N2=.15,N3=0,REM=.85."
        ),
    )
    parser.add_argument(
        "--stage-weights",
        default=None,
        help=(
            "Optional override as five comma-separated STAGE=value pairs; supplying "
            "this records the profile as custom."
        ),
    )
    args = parser.parse_args(argv)
    weight_profile, stage_weights = resolve_stage_weights(
        args.weight_profile, args.stage_weights
    )
    if args.output is None:
        if weight_profile == "literature_mean":
            args.output = DEFAULT_OUTPUT
        elif weight_profile == "requested_heuristic":
            args.output = DEFAULT_OUTPUT / "sensitivity_requested_heuristic"
        else:
            raise ValueError("Custom --stage-weights requires an explicit --output directory")
    args.output.mkdir(parents=True, exist_ok=True)

    consensus, original_training, original_test = load_consensus(args.consensus_csv)
    training, test = select_analysis_records(original_training, original_test)
    records = pd.concat([training, test], ignore_index=True)
    epochs = load_epoch_join(
        args.physical_epochs, args.stage_epochs, records, f_column=args.f_column
    )
    dream_arrays = stage_weighted_epoch_arrays(
        epochs, records, stage_weights, f_column=args.f_column
    )
    full_arrays = epoch_arrays(epochs, records, rem_only=False, f_column=args.f_column)
    configs = eligible_configs(parse_regularization_grid(args.regularization_grid))

    nested_oof, outer_selections, outer_inner_summary, outer_inner_folds = (
        nested_training_oof(
            training,
            dream_arrays,
            full_arrays,
            configs,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
        )
    )
    final_model, final_cv_summary, final_cv_folds, selection_details = fit_final_model(
        training, dream_arrays, configs, inner_splits=args.inner_splits
    )
    fixed_oof = fixed_config_oof(
        training, dream_arrays, full_arrays, final_model, outer_splits=args.outer_splits
    )
    weights_text = ", ".join(f"{stage}={stage_weights[stage]:g}" for stage in STAGE_ORDER)
    train_final = record_predictions(
        training,
        epochs,
        dream_arrays,
        full_arrays,
        final_model,
        stage_weights,
        protocol=(
            "final global g fitted on selected training IDs; in-sample training description; "
            f"fixed stage weights {weights_text}"
        ),
    )
    test_final = record_predictions(
        test,
        epochs,
        dream_arrays,
        full_arrays,
        final_model,
        stage_weights,
        protocol=(
            "held-aside test; global g locked from selected training IDs; "
            f"fixed stage weights {weights_text}"
        ),
    )
    per_epoch = add_final_epoch_outputs(
        epochs, final_model, records, stage_weights, f_column=args.f_column
    )

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
    pd.DataFrame(
        {"f_repeat_micro_v3": grid, "identity": grid, "g_of_f": final_model.transform(grid)}
    ).to_csv(args.output / "g_curve.csv", index=False)
    validation = validate_shell(final_model.spec, final_model.parameters)
    portable = portable_payload(
        final_model, args, weight_profile, stage_weights, validation
    )
    write_json(args.output / "portable_stage_weighted_global_g.json", portable)

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
        "training_target_feasible": int(
            train_final["target_feasible_under_stage_weights_and_bounded_g"].sum()
        ),
        "test_records": len(test),
        "test_target_feasible": int(
            test_final["target_feasible_under_stage_weights_and_bounded_g"].sum()
        ),
    }
    stage_counts = epochs["predicted_stage"].value_counts().to_dict()
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
            "predicted_stage_epoch_counts": {
                stage: int(stage_counts.get(stage, 0)) for stage in STAGE_ORDER
            },
            "join_keys": JOIN_KEYS,
            "exact_one_to_one": True,
        },
        "feasibility": feasibility,
        "metrics": metrics,
    }
    write_json(args.output / "fit_results.json", fit_results)

    train_metric = metrics["dream_time"]["training_nested_selection_oof"]
    test_metric = metrics["dream_time"]["heldout_test"]
    methods = rf"""# Stage-weighted global g calibration

## Definition

The locked physical quantity `{args.f_column}` is not refitted. The exploratory
GSSC hard stage prediction is mapped to a fixed stage-level probability, and
dream time is

\[
T_{{dream}}=\sum_i p(\widehat{{stage}}_i)\,g(f_i)\,\Delta t_i.
\]

This run used the `{weight_profile}` profile: `{weights_text}`. These weights are
fixed inputs, not parameters estimated from Kumral. Total apparent time remains

\[
T_{{apparent}}=\sum_i g(f_i)\,\Delta t_i.
\]

## Records and leakage controls

* IDs 1--5: excluded human-calibration seeds.
* Training IDs 10 and 19: excluded because the recordings are too short.
* Test IDs 29 and 33: excluded because the recordings are too short.
* Remaining training IDs ({len(training)} records): model selection and fitting only.
* Remaining held-aside test IDs ({len(test)} records): evaluated only after locking g.
* Every inner and outer fold is grouped by participant.
* GSSC stages and the physical f trajectory are frozen upstream inputs and are not
  re-cross-fitted within this conditional validation of g.

Selected configuration: `{final_model.spec.key}__lambda_{final_model.regularization:g}`.
Decoded parameters: `{json.dumps(final_model.decoded_parameters(), sort_keys=True)}`.

## Results

Nested training OOF: n={train_metric['n']}, RMSE={train_metric['rmse_s']:.2f} s,
MAE={train_metric['mae_s']:.2f} s, Pearson r={train_metric['pearson_r']:.4f},
Spearman rho={train_metric['spearman_rho']:.4f}.

Held-aside test: n={test_metric['n']}, RMSE={test_metric['rmse_s']:.2f} s,
MAE={test_metric['mae_s']:.2f} s, Pearson r={test_metric['pearson_r']:.4f},
Spearman rho={test_metric['spearman_rho']:.4f}.

Because `0 <= g <= 1`, `sum p(stage) * duration` is a hard record-level upper
bound. {feasibility['training_target_feasible']}/{feasibility['training_records']}
training and {feasibility['test_target_feasible']}/{feasibility['test_records']} test
targets are at or below that bound.

## Interpretation and limitations

The weights are stage-conditioned priors for obtaining a reportable experience at
an awakening; they are **not** measured, epoch-by-epoch dream occupancy fractions.
The full GSSC trajectory has no manual epoch-wise hypnogram in this dataset. Only
awakening endpoints were checked: five-class endpoint accuracy was 13/22 (59.1%;
balanced accuracy 66.7%), while the REM-versus-non-REM endpoint decision was 21/22
(95.45%). The similar literature priors assigned to N1, N2, N3, and REM reduce the
impact of confusion among sleep subclasses, but the W=0 assumption remains a
potentially sensitive boundary. Results must therefore be described as exploratory
and conditional on the frozen GSSC stage predictions and fixed prior weights.
"""
    (args.output / "METHODS_AND_RESULTS.md").write_text(methods, encoding="utf-8")
    running = f"""# Reproduce stage-weighted g calibration

```bash
cd {REPO_ROOT}
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \\
  .venv/bin/python -m analysis.combined_fawake.fit_stage_weighted_global_g \\
  --weight-profile {weight_profile}
```

To rerun with the originally requested heuristic weights, use
`--weight-profile requested_heuristic`. To use revised literature values, supply
all five values with `--stage-weights` and an explicit `--output`. The command
writes to `{args.output}`; it does not modify the
REM-only output or any `paper_candidate...` directory.
"""
    (args.output / "RUNNING.md").write_text(running, encoding="utf-8")

    print(
        json.dumps(
            _json_value(
                {
                    "output": args.output,
                    "stage_weights": stage_weights,
                    "stage_weight_profile": weight_profile,
                    "training_ids": list(TRAIN_IDS),
                    "heldout_ids": list(TEST_IDS),
                    "selected_config": (
                        f"{final_model.spec.key}__lambda_{final_model.regularization:g}"
                    ),
                    "parameters": final_model.decoded_parameters(),
                    "training_nested_oof": train_metric,
                    "heldout_test": test_metric,
                    "feasibility": feasibility,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
