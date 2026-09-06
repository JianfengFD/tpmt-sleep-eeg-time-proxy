#!/usr/bin/env python3
"""Nested audit for the fixed R* logit micro-correction of physical F_S."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve

from analysis.combined_fawake.adjacent_repeat_v3_model import (
    RSTAR_BETA,
    RSTAR_COLUMNS,
    apply_adjacent_repeat_v3,
    calibrate_rstar,
    export_adjacent_repeat_v3,
    fit_adjacent_repeat_v3,
    rstar_wake_nonwake_anchors,
)
from analysis.combined_fawake.dense_repeat_fawake_model import fit_dense_repeat_model
from analysis.combined_fawake.fit_dense_repeat_fawake import (
    RANDOM_STATE,
    STAGE_COLORS,
    STAGE_ORDER,
    _outer_splits,
    _require,
    _write_json,
    fitting_view,
    load_and_assert_feature_table,
)


COMPONENTS = (
    ("f_self_similarity", r"pure $F_S$"),
    ("f_old_return_prominence", "old single-P correction"),
    ("f_repeat_micro_v3", r"fixed $R^*$ micro-correction"),
)
COMPARISONS = ("N1", "N2", "N3", "REM", "all non-W sleep")
RSTAR_SENSITIVITY_QUANTILES = (0.70, 0.75, 0.80, 0.90)
RSTAR_SENSITIVITY_BETAS = (0.20, 0.25, 0.275, 0.50, 1.00)


def nested_grouped_oof(
    train_final: pd.DataFrame, *, outer_splits: int, inner_splits: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the full F_S/old-P model and R* anchors inside every outer fold."""
    compact = fitting_view(train_final.reset_index(drop=True))
    splits = _outer_splits(compact, outer_splits)
    groups = compact["subject_group"].astype(str).to_numpy()
    predictions: list[pd.DataFrame] = []
    stability: list[dict[str, Any]] = []
    for fold_index, (fit_index, validation_index) in enumerate(splits, start=1):
        fit = compact.iloc[fit_index].reset_index(drop=True)
        validation = compact.iloc[validation_index].copy()
        seed = RANDOM_STATE + 1009 * fold_index
        base_model = fit_dense_repeat_model(
            fit, n_splits=inner_splits, random_state=seed
        )
        model = fit_adjacent_repeat_v3(
            fit,
            n_splits=inner_splits,
            random_state=seed,
            base_model=base_model,
        )
        predicted = apply_adjacent_repeat_v3(validation, model)
        predicted["outer_fold"] = fold_index
        predicted["prediction_protocol"] = "nested_subject_group_oof"
        predictions.append(predicted)
        stability.append(
            {
                "outer_fold": fold_index,
                "fit_epochs": int(len(fit_index)),
                "validation_epochs": int(len(validation_index)),
                "fit_subjects": int(len(set(groups[fit_index]))),
                "validation_subjects": int(len(set(groups[validation_index]))),
                "validation_groups": "|".join(sorted(set(groups[validation_index]))),
                "selected_Fs_candidate": base_model["selected_Fs_candidate"],
                "selected_old_P_column": base_model["selected_repeat_column"],
                "selected_old_alpha": float(base_model["alpha"]),
                "Rstar_nonwake_anchor": float(model["nonwake_anchor"]),
                "Rstar_wake_anchor": float(model["wake_anchor"]),
                "Rstar_anchor_gap": float(model["wake_anchor"] - model["nonwake_anchor"]),
                "Rstar_beta_fixed": float(model["beta"]),
                "Rstar_direction_valid": bool(model["wake_anchor"] > model["nonwake_anchor"]),
                "Rstar_definition_selected_in_fold": False,
                "Fs_refitted_inside_outer_fold": True,
                "Rstar_calibration_refitted_inside_outer_fold": True,
            }
        )
        print(
            f"outer fold {fold_index}/{len(splits)}: "
            f"Fs={base_model['selected_Fs_candidate']}; "
            f"oldP={base_model['selected_repeat_column']}; "
            f"R* anchors={model['nonwake_anchor']:.5f}/{model['wake_anchor']:.5f}",
            flush=True,
        )
    oof = pd.concat(predictions).sort_index()
    _require(len(oof) == len(compact), "nested OOF prediction count mismatch")
    _require(not oof.index.duplicated().any(), "nested OOF contains duplicate rows")
    for column, _ in COMPONENTS:
        values = oof[column].to_numpy(float)
        _require(np.isfinite(values).all(), f"incomplete nested OOF component {column}")
        _require(np.all((values >= 0.0) & (values <= 1.0)), f"invalid {column}")
    return oof.reset_index(drop=True), pd.DataFrame(stability)


def _comparison_subset(frame: pd.DataFrame, negative: str) -> pd.DataFrame:
    zhang = frame.loc[frame["dataset"].astype(str).eq("Zhang")]
    if negative == "all non-W sleep":
        return zhang.loc[zhang["manual_final_stage"].isin(STAGE_ORDER)].copy()
    return zhang.loc[zhang["manual_final_stage"].isin(["W", negative])].copy()


def auc_rows(
    frame: pd.DataFrame, evaluation: str, *, outer_fold: int | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for negative in COMPARISONS:
        subset = _comparison_subset(frame, negative)
        labels = subset["manual_final_stage"].eq("W").astype(int).to_numpy()
        for component, label in COMPONENTS:
            auc = None
            if len(subset) and len(np.unique(labels)) == 2:
                auc = float(roc_auc_score(labels, subset[component].to_numpy(float)))
            rows.append(
                {
                    "evaluation": evaluation,
                    "outer_fold": outer_fold,
                    "dataset_scope": "Zhang only",
                    "positive_stage": "W",
                    "negative_stage": negative,
                    "component": component,
                    "component_label": label.replace("$", ""),
                    "n_positive_epochs": int(np.sum(labels == 1)),
                    "n_negative_epochs": int(np.sum(labels == 0)),
                    "auc": auc,
                }
            )
    return rows


def stage_rows(frame: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope, subset in (
        ("pooled", frame),
        ("Zhang", frame.loc[frame["dataset"].eq("Zhang")]),
        ("Kumral", frame.loc[frame["dataset"].eq("Kumral")]),
    ):
        for stage in STAGE_ORDER:
            stage_frame = subset.loc[subset["manual_final_stage"].eq(stage)]
            for component, label in COMPONENTS:
                values = stage_frame[component].to_numpy(float)
                if len(values) == 0:
                    continue
                rows.append(
                    {
                        "evaluation": evaluation,
                        "dataset_scope": scope,
                        "stage": stage,
                        "component": component,
                        "component_label": label.replace("$", ""),
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


def correction_rows(frame: pd.DataFrame, evaluation: str) -> list[dict[str, Any]]:
    base = frame["f_self_similarity"].to_numpy(float)
    rows = []
    for component, label in COMPONENTS[1:]:
        change = frame[component].to_numpy(float) - base
        rows.append(
            {
                "evaluation": evaluation,
                "component": component,
                "component_label": label.replace("$", ""),
                "n_epochs": int(len(change)),
                "mean_signed_change": float(np.mean(change)),
                "mean_absolute_change": float(np.mean(np.abs(change))),
                "median_absolute_change": float(np.median(np.abs(change))),
                "q90_absolute_change": float(np.quantile(np.abs(change), 0.90)),
                "maximum_absolute_change": float(np.max(np.abs(change))),
                "fraction_increased": float(np.mean(change > 1.0e-12)),
                "fraction_decreased": float(np.mean(change < -1.0e-12)),
            }
        )
    return rows


def _raw_rstar_at_quantile(frame: pd.DataFrame, quantile: float) -> np.ndarray:
    """Return the declared two-column P quantile for a sensitivity value."""
    matrix = frame.loc[:, RSTAR_COLUMNS].to_numpy(float)
    _require(np.isfinite(matrix).all(), "non-finite R* sensitivity input")
    return np.quantile(matrix, float(quantile), axis=1, method="linear")


def _sensitivity_component(
    frame: pd.DataFrame,
    raw: np.ndarray,
    nonwake_anchor: float,
    wake_anchor: float,
    beta: float,
) -> np.ndarray:
    repeat = calibrate_rstar(raw, nonwake_anchor, wake_anchor)
    fs = frame["f_self_similarity"].to_numpy(float)
    output = fs.copy()
    interior = (fs > 0.0) & (fs < 1.0)
    output[interior] = expit(
        np.log(fs[interior])
        - np.log1p(-fs[interior])
        + float(beta) * (2.0 * repeat[interior] - 1.0)
    )
    return output


def _sensitivity_summary_row(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    evaluation: str,
    quantile: float,
    beta: float,
    anchors: list[tuple[float, float]],
) -> dict[str, Any]:
    base = frame["f_self_similarity"].to_numpy(float)
    change = np.asarray(values, dtype=float) - base
    row: dict[str, Any] = {
        "evaluation": evaluation,
        "quantile": float(quantile),
        "beta": float(beta),
        "is_locked_preferred": bool(
            abs(float(quantile) - 0.75) <= 1.0e-12
            and abs(float(beta) - RSTAR_BETA) <= 1.0e-12
        ),
        "status": "descriptive_sensitivity_not_used_for_selection",
        "n_endpoint_epochs": int(len(frame)),
        "calibration_folds": int(len(anchors)),
        "nonwake_anchor_min": float(min(pair[0] for pair in anchors)),
        "nonwake_anchor_max": float(max(pair[0] for pair in anchors)),
        "wake_anchor_min": float(min(pair[1] for pair in anchors)),
        "wake_anchor_max": float(max(pair[1] for pair in anchors)),
        "mean_signed_correction": float(np.mean(change)),
        "mean_absolute_correction": float(np.mean(np.abs(change))),
        "maximum_absolute_correction": float(np.max(np.abs(change))),
    }
    all_four_safe = True
    for negative in COMPARISONS:
        subset = _comparison_subset(frame, negative)
        index = subset.index.to_numpy(dtype=int)
        labels = subset["manual_final_stage"].eq("W").astype(int).to_numpy()
        pure_auc = float(
            roc_auc_score(labels, subset["f_self_similarity"].to_numpy(float))
        )
        sensitivity_auc = float(roc_auc_score(labels, np.asarray(values)[index]))
        suffix = negative.replace("all non-W sleep", "all_nonW").replace(" ", "_")
        row[f"auc_W_vs_{suffix}"] = sensitivity_auc
        row[f"delta_auc_W_vs_{suffix}"] = sensitivity_auc - pure_auc
        if negative != "all non-W sleep":
            all_four_safe &= sensitivity_auc >= pure_auc - 1.0e-12
    row["all_four_stage_pairs_non_decreasing"] = bool(all_four_safe)
    return row


def rstar_weight_sensitivity(
    train_final: pd.DataFrame,
    training_oof: pd.DataFrame,
    diagnostic_test: pd.DataFrame,
) -> pd.DataFrame:
    """Descriptive q/beta audit with fold-local training calibrations.

    This grid is intentionally downstream of the locked q=.75/beta=.25 design.
    It is reported to show local robustness and is never consulted by the fit.
    """
    compact = fitting_view(train_final.reset_index(drop=True))
    _require(len(training_oof) == len(compact), "sensitivity/OOF row mismatch")
    _require(
        training_oof.index.equals(pd.RangeIndex(len(training_oof))),
        "training OOF must have a simple row index for sensitivity audit",
    )
    rows: list[dict[str, Any]] = []
    for quantile in RSTAR_SENSITIVITY_QUANTILES:
        train_values_by_beta = {
            beta: np.full(len(training_oof), np.nan, dtype=float)
            for beta in RSTAR_SENSITIVITY_BETAS
        }
        fold_anchors: list[tuple[float, float]] = []
        for fold in sorted(training_oof["outer_fold"].astype(int).unique()):
            validation_mask = training_oof["outer_fold"].astype(int).eq(fold).to_numpy()
            validation_groups = set(
                training_oof.loc[validation_mask, "subject_group"].astype(str)
            )
            fit = compact.loc[
                ~compact["subject_group"].astype(str).isin(validation_groups)
            ].copy()
            raw_fit = _raw_rstar_at_quantile(fit, quantile)
            nonwake_anchor, wake_anchor = rstar_wake_nonwake_anchors(fit, raw_fit)
            fold_anchors.append((nonwake_anchor, wake_anchor))
            validation = training_oof.loc[validation_mask]
            raw_validation = _raw_rstar_at_quantile(validation, quantile)
            for beta in RSTAR_SENSITIVITY_BETAS:
                train_values_by_beta[beta][validation_mask] = _sensitivity_component(
                    validation,
                    raw_validation,
                    nonwake_anchor,
                    wake_anchor,
                    beta,
                )
        raw_full = _raw_rstar_at_quantile(compact, quantile)
        full_anchors = rstar_wake_nonwake_anchors(compact, raw_full)
        raw_test = _raw_rstar_at_quantile(diagnostic_test, quantile)
        for beta in RSTAR_SENSITIVITY_BETAS:
            train_values = train_values_by_beta[beta]
            _require(np.isfinite(train_values).all(), "incomplete training sensitivity grid")
            test_values = _sensitivity_component(
                diagnostic_test,
                raw_test,
                full_anchors[0],
                full_anchors[1],
                beta,
            )
            rows.append(
                _sensitivity_summary_row(
                    training_oof,
                    train_values,
                    evaluation="training_nested_subject_group_oof",
                    quantile=quantile,
                    beta=beta,
                    anchors=fold_anchors,
                )
            )
            rows.append(
                _sensitivity_summary_row(
                    diagnostic_test,
                    test_values,
                    evaluation="repeated_held_aside_diagnostic_test",
                    quantile=quantile,
                    beta=beta,
                    anchors=[full_anchors],
                )
            )

            if abs(quantile - 0.75) <= 1.0e-12 and abs(beta - RSTAR_BETA) <= 1.0e-12:
                np.testing.assert_allclose(
                    train_values,
                    training_oof["f_repeat_micro_v3"].to_numpy(float),
                    rtol=0.0,
                    atol=2.0e-14,
                )
                np.testing.assert_allclose(
                    test_values,
                    diagnostic_test["f_repeat_micro_v3"].to_numpy(float),
                    rtol=0.0,
                    atol=2.0e-14,
                )
    output = pd.DataFrame(rows).sort_values(
        ["quantile", "beta", "evaluation"]
    ).reset_index(drop=True)
    _require(len(output) == 2 * len(RSTAR_SENSITIVITY_QUANTILES) * len(RSTAR_SENSITIVITY_BETAS),
             "unexpected R* sensitivity row count")
    return output


def clustered_auc_bootstrap(
    frame: pd.DataFrame,
    evaluation: str,
    *,
    replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap subject groups for old-minus-pure and v3-minus-pure AUC."""
    rng = np.random.default_rng(RANDOM_STATE + (0 if "training" in evaluation else 1))
    replicate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for comparison_index, negative in enumerate(COMPARISONS):
        pair = _comparison_subset(frame, negative).reset_index(drop=True)
        groups = sorted(pair["subject_group"].astype(str).unique())
        group_indices = {
            group: pair.index[pair["subject_group"].astype(str).eq(group)].to_numpy()
            for group in groups
        }
        collected: dict[str, list[float]] = {component: [] for component, _ in COMPONENTS[1:]}
        attempts = 0
        while min((len(values) for values in collected.values()), default=0) < replicates:
            attempts += 1
            if attempts > max(100, replicates * 30):
                raise RuntimeError(f"bootstrap failed for {evaluation} {negative}")
            sampled = rng.choice(groups, size=len(groups), replace=True)
            index = np.concatenate([group_indices[str(group)] for group in sampled])
            sample = pair.iloc[index]
            labels = sample["manual_final_stage"].eq("W").astype(int).to_numpy()
            if len(np.unique(labels)) != 2:
                continue
            pure = float(roc_auc_score(labels, sample["f_self_similarity"].to_numpy(float)))
            for component, _ in COMPONENTS[1:]:
                value = float(roc_auc_score(labels, sample[component].to_numpy(float)))
                collected[component].append(value - pure)
        for component, label in COMPONENTS[1:]:
            values = np.asarray(collected[component], dtype=float)
            for replicate, value in enumerate(values, start=1):
                replicate_rows.append(
                    {
                        "evaluation": evaluation,
                        "negative_stage": negative,
                        "component": component,
                        "bootstrap_replicate": replicate,
                        "delta_auc_minus_pure_Fs": float(value),
                    }
                )
            summary_rows.append(
                {
                    "evaluation": evaluation,
                    "negative_stage": negative,
                    "component": component,
                    "component_label": label.replace("$", ""),
                    "resampling_unit": "Zhang subject_group",
                    "replicates": replicates,
                    "mean_delta_auc_minus_pure_Fs": float(np.mean(values)),
                    "median_delta_auc_minus_pure_Fs": float(np.median(values)),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "fraction_delta_above_zero": float(np.mean(values > 0.0)),
                }
            )
    return pd.DataFrame(replicate_rows), pd.DataFrame(summary_rows)


def plot_auc_comparison(auc: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), sharex=True, sharey=True)
    evaluations = (
        ("training_nested_subject_group_oof", "Training: nested subject-group OOF"),
        ("repeated_held_aside_diagnostic_test", "Repeated held-aside diagnostic test"),
    )
    x = np.arange(len(COMPARISONS), dtype=float)
    colors = ("#4c78a8", "#e45756", "#2a9d8f")
    markers = ("o", "s", "D")
    for axis, (evaluation, title) in zip(axes, evaluations, strict=True):
        subset = auc.loc[auc["evaluation"].eq(evaluation) & auc["outer_fold"].isna()]
        for (component, label), color, marker in zip(COMPONENTS, colors, markers, strict=True):
            values = [
                float(
                    subset.loc[
                        subset["negative_stage"].eq(negative)
                        & subset["component"].eq(component),
                        "auc",
                    ].iloc[0]
                )
                for negative in COMPARISONS
            ]
            axis.plot(x, values, marker=marker, color=color, lw=2.0, ms=6, label=label)
        axis.axhline(0.5, color="#777777", ls="--", lw=1.0)
        axis.set_ylim(0.45, 1.01)
        axis.set_ylabel("ROC AUC")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, ncol=3, fontsize=9)
    axes[-1].set_xticks(x, COMPARISONS)
    axes[-1].set_xlabel("Zhang final-epoch comparison; W is positive")
    fig.suptitle("Pure self-similarity, old P correction, and fixed R* micro-correction")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_stage_distributions(
    training: pd.DataFrame, test: pd.DataFrame, output: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), sharex=True, sharey=True)
    evaluations = (
        (training, "Training: nested subject-group OOF"),
        (test, "Repeated held-aside diagnostic test"),
    )
    positions = np.arange(len(STAGE_ORDER), dtype=float)
    offsets = (-0.22, 0.0, 0.22)
    colors = ("#4c78a8", "#e45756", "#2a9d8f")
    for axis, (frame, title) in zip(axes, evaluations, strict=True):
        zhang = frame.loc[frame["dataset"].eq("Zhang")]
        for (component, label), offset, color in zip(COMPONENTS, offsets, colors, strict=True):
            values = [
                zhang.loc[zhang["manual_final_stage"].eq(stage), component].to_numpy(float)
                for stage in STAGE_ORDER
            ]
            artists = axis.boxplot(
                values,
                positions=positions + offset,
                widths=0.19,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                medianprops={"color": "white", "linewidth": 1.2},
            )
            for patch in artists["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.78)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel(r"$\bar f(\lambda)$ component")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(positions, STAGE_ORDER)
    axes[-1].set_xlabel("manual final-epoch stage")
    handles = [
        plt.Line2D([0], [0], color=color, lw=8, alpha=0.78, label=label)
        for (_component, label), color in zip(COMPONENTS, colors, strict=True)
    ]
    fig.suptitle(
        "Zhang-only distributions: self-similarity remains the primary mapping",
        y=0.995,
    )
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.965), fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_outer_fold_deltas(auc: pd.DataFrame, output: Path) -> None:
    subset = auc.loc[
        auc["evaluation"].eq("training_outer_fold")
        & auc["component"].isin(["f_old_return_prominence", "f_repeat_micro_v3"])
    ].copy()
    pure = auc.loc[
        auc["evaluation"].eq("training_outer_fold")
        & auc["component"].eq("f_self_similarity"),
        ["outer_fold", "negative_stage", "auc"],
    ].rename(columns={"auc": "pure_auc"})
    subset = subset.merge(pure, on=["outer_fold", "negative_stage"], how="left")
    subset["delta"] = subset["auc"] - subset["pure_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for axis, (component, label, color) in zip(
        axes,
        (
            ("f_old_return_prominence", "old single-P correction", "#e45756"),
            ("f_repeat_micro_v3", "fixed R* micro-correction", "#2a9d8f"),
        ),
        strict=True,
    ):
        data = subset.loc[subset["component"].eq(component)]
        for fold in sorted(data["outer_fold"].dropna().unique()):
            fold_data = data.loc[data["outer_fold"].eq(fold)].set_index("negative_stage")
            axis.plot(
                np.arange(len(COMPARISONS)),
                [fold_data.loc[name, "delta"] for name in COMPARISONS],
                marker="o",
                lw=1.2,
                alpha=0.72,
                label=f"fold {int(fold)}",
            )
        axis.axhline(0.0, color="#333333", lw=1.0)
        axis.set_xticks(np.arange(len(COMPARISONS)), COMPARISONS, rotation=24, ha="right")
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.set_ylabel(r"$\Delta$AUC versus pure $F_S$")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Outer-fold instability audit; each fold refits F_S and recurrence calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def metric_lookup(auc: pd.DataFrame, evaluation: str, negative: str, component: str) -> float:
    row = auc.loc[
        auc["evaluation"].eq(evaluation)
        & auc["outer_fold"].isna()
        & auc["negative_stage"].eq(negative)
        & auc["component"].eq(component),
        "auc",
    ]
    return float(row.iloc[0])


def build_report(
    input_audit: dict[str, Any],
    final_model: dict[str, Any],
    auc: pd.DataFrame,
    stage: pd.DataFrame,
    stability: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    train_eval = "training_nested_subject_group_oof"
    test_eval = "repeated_held_aside_diagnostic_test"

    def table(evaluation: str) -> list[str]:
        lines = [
            "| comparison | pure F_S | old P | fixed R* | R* - pure |",
            "|---|---:|---:|---:|---:|",
        ]
        for negative in COMPARISONS:
            pure = metric_lookup(auc, evaluation, negative, "f_self_similarity")
            old = metric_lookup(auc, evaluation, negative, "f_old_return_prominence")
            new = metric_lookup(auc, evaluation, negative, "f_repeat_micro_v3")
            lines.append(
                f"| W vs {negative} | {pure:.4f} | {old:.4f} | {new:.4f} | {new-pure:+.4f} |"
            )
        return lines

    zhang_train = stage.loc[
        stage["evaluation"].eq(train_eval) & stage["dataset_scope"].eq("Zhang")
    ]

    def median(stage_name: str, component: str) -> float:
        return float(
            zhang_train.loc[
                zhang_train["stage"].eq(stage_name)
                & zhang_train["component"].eq(component),
                "median",
            ].iloc[0]
        )

    outer = stability.loc[stability["outer_fold"].astype(str).ne("final_full_train")]
    fold_auc = auc.loc[
        auc["evaluation"].eq("training_outer_fold")
        & auc["negative_stage"].eq("all non-W sleep")
    ]
    fold_wide = fold_auc.pivot(index="outer_fold", columns="component", values="auc")
    positive_folds = int(
        (fold_wide["f_repeat_micro_v3"] > fold_wide["f_self_similarity"]).sum()
    )
    fs_counts = Counter(outer["selected_Fs_candidate"].astype(str))
    p_counts = Counter(outer["selected_old_P_column"].astype(str))
    preferred_sensitivity = sensitivity.loc[sensitivity["is_locked_preferred"]]
    _require(len(preferred_sensitivity) == 2, "preferred sensitivity rows missing")
    preferred_safe = bool(
        preferred_sensitivity["all_four_stage_pairs_non_decreasing"].all()
    )
    return "\n".join(
        [
            "# Fixed R* micro-correction: methods and results",
            "",
            "## Definition and scope",
            "",
            "The primary mapping is the robust gamma-marginalized self-similarity F_S from v2. "
            "Its training N3/W anchors remain mapped to 0.05/0.95. The separate 0.02 in the "
            "persistence-ratio formula is only denominator stabilization, not an endpoint target. "
            "The new modifier uses P*=q75(P at H=0.5 s and tau=0.08/0.10 s), with NumPy's "
            "linear two-value quantile. Inside every fit, the Zhang all-non-W median and W "
            "median map smoothly to R=0.05 and 0.95. The physical sign is fixed and every "
            "fit must have median(P*|W)>median(P*|non-W).",
            "",
            "The final formula is `f=expit(logit(F_S)+0.25*(2R-1))`. Thus recurrence changes "
            "the F_S log-odds by at most 0.25 and fixes the exact endpoints. P*, q=.75, and "
            "beta=.25 are fixed development choices, not parameters claimed to have been "
            "selected by an unbiased nested procedure. Dream reports, dream durations, and "
            "spectral/stage-classifier features were not read.",
            "",
            "## Validation protocol",
            "",
            f"Input has {input_audit['rows']:,} epochs, {input_audit['records']} records, and "
            f"{input_audit['subjects']} subject groups. The outer StratifiedGroupKFold predicts "
            "each training subject exactly once. Within each outer fit, the complete robust F_S "
            "feature selection and N3/W calibration are rerun, the old P comparator is rerun, "
            "and the R* W/non-W calibration uses only outer-training Zhang rows. Test rows never "
            "enter fitting.",
            "",
            "Because this R* definition was developed after inspection of this study and the "
            "held-aside subjects were seen in earlier iterations, the outer result is a "
            "cross-fitted development audit and the test result is a repeated held-aside "
            "diagnostic—not a pristine confirmatory estimate.",
            "",
            "## Cross-fitted training audit",
            "",
            *table(train_eval),
            "",
            "## Repeated held-aside diagnostic",
            "",
            *table(test_eval),
            "",
            "## Endpoint and stability audit",
            "",
            f"Training nested-OOF medians for the new f are W={median('W','f_repeat_micro_v3'):.4f}, "
            f"N1={median('N1','f_repeat_micro_v3'):.4f}, N3={median('N3','f_repeat_micro_v3'):.4f}, "
            f"and REM={median('REM','f_repeat_micro_v3'):.4f}. The new all-non-W AUC exceeded "
            f"pure F_S in {positive_folds}/{len(fold_wide)} outer folds.",
            "",
            f"Outer folds selected {len(fs_counts)} F_S definitions and {len(p_counts)} old-P "
            "definitions. That instability belongs to the freely selected base comparator; "
            "P*, q=.75, and beta=.25 do not vary by fold. Exact choices and fold-local anchors "
            "are in `selection_stability.csv`.",
            "",
            "The old single-P correction gives a larger cross-fitted development gain but loses "
            "AUC in the repeated test. R* gives smaller changes but was directionally consistent "
            "across the aggregate and the principal N1/REM comparisons in the combined audit. "
            "For downstream g fitting, `f_repeat_micro_v3` is the recommended main trajectory; "
            "retain pure `f_self_similarity` as the required sensitivity baseline.",
            "",
            "## R* weight sensitivity (descriptive only)",
            "",
            "`R_weight_sensitivity.csv` compares q=.70/.75/.80/.90 and "
            "beta=.20/.25/.275/.5/1.0. For training rows, every q is recalibrated from only "
            "the corresponding outer-training Zhang subjects; the repeated-test rows use the "
            "full-training calibration. The locked q=.75, beta=.25 setting "
            + ("does" if preferred_safe else "does not")
            + " avoid AUC decreases for all four W-versus-stage pairs in both summaries. It is "
            "a simple, small correction with balanced training/diagnostic behavior; q=.70 and "
            "q=.80 are nearby comparators. Larger beta can look stronger in development while "
            "reducing held-aside W-N1 or W-REM separation. This post-lock audit was not used to "
            "reselect q, beta, F_S, or any model parameter.",
            "",
            "## Locked full-training parameters",
            "",
            f"- F_S: `{final_model['base_model']['selected_Fs_candidate']}`",
            f"- F_S calibration targets: {final_model['base_model']['endpoint_epsilon']:.2f}/"
            f"{1-final_model['base_model']['endpoint_epsilon']:.2f} for N3/W anchors",
            f"- old P comparator: `{final_model['base_model']['selected_repeat_column']}`, "
            f"alpha={float(final_model['base_model']['alpha']):.4g}",
            f"- P* source columns: `{', '.join(RSTAR_COLUMNS)}`",
            f"- full-training R* anchors: non-W={final_model['nonwake_anchor']:.8f}, "
            f"W={final_model['wake_anchor']:.8f}",
            f"- fixed beta: {RSTAR_BETA}",
            "",
            "## Output map",
            "",
            "- `training_nested_oof_predictions.csv`: one leakage-controlled prediction per training endpoint.",
            "- `diagnostic_test_predictions.csv`: locked repeated-test endpoint predictions.",
            "- `all_epoch_trajectories.csv`: locked trajectories for all 9,197 epochs.",
            "- `pairwise_auc_metrics.csv` and `outer_fold_auc_metrics.csv`: aggregate and fold audits.",
            "- `stage_distribution_metrics.csv`: pooled and dataset-specific summaries.",
            "- `subject_cluster_bootstrap_summary.csv`: subject-level uncertainty audit.",
            "- `R_weight_sensitivity.csv`: post-lock q/beta robustness audit; never a selector.",
            "- `adjacent_repeat_v3_portable.json`: parameters needed for later application and g fitting.",
            "- `PREDICTING.md` and `code/predict_adjacent_repeat_v3.py`: standalone CSV/EDF application.",
            "",
        ]
    )


def build_running(features: Path, output: Path, args: argparse.Namespace) -> str:
    return f"""# Running the fixed R* micro-correction audit

From the project root:

```bash
XDG_CACHE_HOME=.cache_runtime MPLCONFIGDIR=.cache_runtime/matplotlib \\
.venv/bin/python -m analysis.combined_fawake.fit_adjacent_repeat_v3 \\
  --features {features} \\
  --output {output} \\
  --outer-splits {args.outer_splits} \\
  --inner-splits {args.inner_splits} \\
  --bootstrap-replicates {args.bootstrap_replicates}
```

The input must contain the complete self-similarity grid, the return-prominence
grid, and the declared train/test metadata. The script does not read dream
reports or durations and does not fit g. The output directory is independent
of v2 and never overwrites it.

Use `predict_adjacent_repeat_v3.py` with the exported portable JSON to apply
the locked mapping to one existing feature-table record or directly to a raw
Zhang/Kumral EDF. See `PREDICTING.md` in the output directory.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("outputs/dense_repeat_fawake_v2/epoch_features_S_RP_grid.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/dense_repeat_fawake_v3/rstar_micro_fit"),
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require(args.outer_splits >= 2, "outer-splits must be at least 2")
    _require(args.inner_splits >= 2, "inner-splits must be at least 2")
    _require(args.bootstrap_replicates >= 100, "bootstrap-replicates must be at least 100")
    args.output.mkdir(parents=True, exist_ok=True)

    epochs, input_audit = load_and_assert_feature_table(args.features)
    for column in RSTAR_COLUMNS:
        _require(column in epochs, f"missing fixed R* column {column}")
    final = epochs.loc[epochs["is_final_labelled_epoch"]].copy()
    train_final = final.loc[final["split"].eq("train")].reset_index(drop=True)
    test_final = final.loc[final["split"].eq("test")].reset_index(drop=True)

    training_oof, stability = nested_grouped_oof(
        train_final, outer_splits=args.outer_splits, inner_splits=args.inner_splits
    )
    base_model = fit_dense_repeat_model(
        fitting_view(train_final), n_splits=args.inner_splits, random_state=RANDOM_STATE
    )
    final_model = fit_adjacent_repeat_v3(
        fitting_view(train_final),
        n_splits=args.inner_splits,
        random_state=RANDOM_STATE,
        base_model=base_model,
    )
    stability = pd.concat(
        [
            stability,
            pd.DataFrame(
                [
                    {
                        "outer_fold": "final_full_train",
                        "fit_epochs": len(train_final),
                        "validation_epochs": 0,
                        "fit_subjects": train_final["subject_group"].nunique(),
                        "validation_subjects": 0,
                        "validation_groups": "",
                        "selected_Fs_candidate": base_model["selected_Fs_candidate"],
                        "selected_old_P_column": base_model["selected_repeat_column"],
                        "selected_old_alpha": float(base_model["alpha"]),
                        "Rstar_nonwake_anchor": float(final_model["nonwake_anchor"]),
                        "Rstar_wake_anchor": float(final_model["wake_anchor"]),
                        "Rstar_anchor_gap": float(
                            final_model["wake_anchor"] - final_model["nonwake_anchor"]
                        ),
                        "Rstar_beta_fixed": float(final_model["beta"]),
                        "Rstar_direction_valid": True,
                        "Rstar_definition_selected_in_fold": False,
                        "Fs_refitted_inside_outer_fold": True,
                        "Rstar_calibration_refitted_inside_outer_fold": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    print(
        f"full fit: Fs={base_model['selected_Fs_candidate']}; "
        f"oldP={base_model['selected_repeat_column']}; "
        f"R* anchors={final_model['nonwake_anchor']:.5f}/{final_model['wake_anchor']:.5f}",
        flush=True,
    )

    diagnostic_test = apply_adjacent_repeat_v3(fitting_view(test_final), final_model)
    diagnostic_test["outer_fold"] = np.nan
    diagnostic_test["prediction_protocol"] = "repeated_held_aside_diagnostic_test"
    full_prediction = apply_adjacent_repeat_v3(epochs, final_model)
    full_prediction["prediction_protocol"] = "locked_full_training_model"

    train_eval = "training_nested_subject_group_oof"
    test_eval = "repeated_held_aside_diagnostic_test"
    aggregate_auc = auc_rows(training_oof, train_eval) + auc_rows(diagnostic_test, test_eval)
    outer_auc: list[dict[str, Any]] = []
    for fold in sorted(training_oof["outer_fold"].unique()):
        outer_auc.extend(
            auc_rows(
                training_oof.loc[training_oof["outer_fold"].eq(fold)],
                "training_outer_fold",
                outer_fold=int(fold),
            )
        )
    auc_frame = pd.DataFrame(aggregate_auc + outer_auc)
    stage_frame = pd.DataFrame(
        stage_rows(training_oof, train_eval) + stage_rows(diagnostic_test, test_eval)
    )
    correction_frame = pd.DataFrame(
        correction_rows(training_oof, train_eval) + correction_rows(diagnostic_test, test_eval)
    )
    sensitivity_frame = rstar_weight_sensitivity(
        train_final, training_oof, diagnostic_test
    )
    train_boot, train_boot_summary = clustered_auc_bootstrap(
        training_oof, train_eval, replicates=args.bootstrap_replicates
    )
    test_boot, test_boot_summary = clustered_auc_bootstrap(
        diagnostic_test, test_eval, replicates=args.bootstrap_replicates
    )
    bootstrap = pd.concat([train_boot, test_boot], ignore_index=True)
    bootstrap_summary = pd.concat(
        [train_boot_summary, test_boot_summary], ignore_index=True
    )

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
            *RSTAR_COLUMNS,
            "self_similarity_Fs_raw",
            "f_self_similarity",
            "return_prominence_P_old_raw",
            "return_prominence_R_old",
            "f_old_return_prominence",
            "return_prominence_P_star",
            "return_prominence_R_star",
            "f_repeat_micro_v3",
            "prediction_protocol",
        )
        if column in full_prediction
    ]
    training_oof.to_csv(args.output / "training_nested_oof_predictions.csv", index=False)
    diagnostic_test.to_csv(args.output / "diagnostic_test_predictions.csv", index=False)
    full_prediction.loc[:, trajectory_columns].to_csv(
        args.output / "all_epoch_trajectories.csv", index=False
    )
    stability.to_csv(args.output / "selection_stability.csv", index=False)
    auc_frame.loc[auc_frame["outer_fold"].isna()].to_csv(
        args.output / "pairwise_auc_metrics.csv", index=False
    )
    auc_frame.loc[auc_frame["outer_fold"].notna()].to_csv(
        args.output / "outer_fold_auc_metrics.csv", index=False
    )
    stage_frame.to_csv(args.output / "stage_distribution_metrics.csv", index=False)
    correction_frame.to_csv(args.output / "correction_magnitude_metrics.csv", index=False)
    sensitivity_frame.to_csv(args.output / "R_weight_sensitivity.csv", index=False)
    bootstrap.to_csv(args.output / "subject_cluster_bootstrap.csv", index=False)
    bootstrap_summary.to_csv(
        args.output / "subject_cluster_bootstrap_summary.csv", index=False
    )

    portable = export_adjacent_repeat_v3(final_model)
    _write_json(args.output / "adjacent_repeat_v3_portable.json", portable)
    results = {
        "analysis": "fixed R* symmetric-logit micro-correction of self-similarity F_S",
        "scope": "initial f only; no g, dream report, or dream duration",
        "input_audit": input_audit,
        "training_protocol": {
            "estimate": train_eval,
            "outer_splitter": "StratifiedGroupKFold by subject_group",
            "outer_splits_requested": args.outer_splits,
            "entire_Fs_model_refitted_per_outer_fold": True,
            "Rstar_calibration_refitted_per_outer_fold": True,
            "Rstar_definition_and_beta_fixed": True,
            "crossfit_is_preregistered_unbiased_evidence": False,
            "reason": "R* design was developed after inspection of this study",
        },
        "test_protocol": {
            "name": test_eval,
            "untouched_confirmatory_test": False,
            "reason": "held-aside subjects were inspected in prior project iterations",
        },
        "final_model": portable,
        "pairwise_auc_metrics": pd.DataFrame(aggregate_auc).to_dict(orient="records"),
        "outer_fold_auc_metrics": pd.DataFrame(outer_auc).to_dict(orient="records"),
        "bootstrap_summary": bootstrap_summary.to_dict(orient="records"),
        "correction_magnitude_metrics": correction_frame.to_dict(orient="records"),
        "R_weight_sensitivity": {
            "status": "descriptive_post_lock_audit_not_used_for_selection",
            "grid_csv": "R_weight_sensitivity.csv",
            "quantiles": list(RSTAR_SENSITIVITY_QUANTILES),
            "betas": list(RSTAR_SENSITIVITY_BETAS),
        },
        "selection_stability": stability.to_dict(orient="records"),
    }
    _write_json(args.output / "fit_results.json", results)

    plot_auc_comparison(auc_frame, args.output / "pairwise_auc_comparison.png")
    plot_stage_distributions(
        training_oof, diagnostic_test, args.output / "stage_distributions.png"
    )
    plot_outer_fold_deltas(auc_frame, args.output / "outer_fold_auc_deltas.png")
    (args.output / "METHODS_AND_RESULTS.md").write_text(
        build_report(
            input_audit,
            final_model,
            auc_frame,
            stage_frame,
            stability,
            sensitivity_frame,
        ),
        encoding="utf-8",
    )
    (args.output / "RUNNING.md").write_text(
        build_running(args.features, args.output, args), encoding="utf-8"
    )
    shutil.copy2(
        Path(__file__).resolve().parent / "ADJACENT_REPEAT_V3_PREDICTING.md",
        args.output / "PREDICTING.md",
    )

    code_output = args.output / "code"
    code_output.mkdir(parents=True, exist_ok=True)
    for source_name in (
        "__init__.py",
        "dense_repeat_fawake_model.py",
        "adjacent_repeat_v3_model.py",
        "augment_adjacent_recurrence_grid.py",
        "eeg_features.py",
        "fit_dense_repeat_fawake.py",
        "fit_adjacent_repeat_v3.py",
        "predict_dense_repeat_fawake_initial.py",
        "predict_adjacent_repeat_v3.py",
    ):
        shutil.copy2(Path(__file__).resolve().parent / source_name, code_output / source_name)
    print(f"wrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
