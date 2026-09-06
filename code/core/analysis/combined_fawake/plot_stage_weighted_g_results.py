#!/usr/bin/env python3
"""Plot the independent stage-weighted ``g`` analysis and replay trajectories.

Training correlations use conditional nested participant-grouped OOF estimates
for ``g``. Test correlations and all per-record test trajectories use the global
``g`` locked from the selected training records. Per-record trajectories display
the cumulative integral

    sum p(predicted_stage) * g(f) * duration.

The upstream physical trajectory and GSSC stage predictions remain frozen.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analysis.combined_fawake.fit_stage_weighted_global_g import (
    DEFAULT_CONSENSUS,
    DEFAULT_OUTPUT,
    F_COLUMN,
    STAGE_ORDER,
    TEST_IDS,
    TRAIN_IDS,
)
from analysis.combined_fawake.plot_rem_only_g_results import (
    BAND_COLORS,
    STAGE_COLORS,
    label_long_spans,
    metric_summary,
    run_length_spans,
    sha256,
    wrap_report,
    write_json,
)


DEFAULT_ROOT = DEFAULT_OUTPUT
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "final_plots"
STAGE_PROVENANCE = (
    "exploratory hard prediction from pretrained GSSC v0.0.9 using C3-TP10/C4-TP9, "
    "official five-class argmax and loudest-vote; not a manual epoch-wise hypnogram"
)
TRAJECTORY_COLUMNS_TO_PRESERVE = [
    "bar_f_lambda",
    "dream_weight",
    "stage_dream_probability",
    "apparent_increment_s",
    "weighted_dream_increment_s",
    "cumulative_apparent_s",
    "cumulative_weighted_dream_s",
    "g_config_key",
]


def load_inputs(
    root: Path,
    consensus_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    epochs = pd.read_csv(root / "all_epoch_trajectories.csv", low_memory=False)
    training = pd.read_csv(root / "training_nested_oof_predictions.csv")
    test = pd.read_csv(root / "heldout_test_predictions.csv")
    final_records = pd.read_csv(root / "all_final_model_record_predictions.csv")
    portable = json.loads(
        (root / "portable_stage_weighted_global_g.json").read_text(encoding="utf-8")
    )
    consensus = pd.read_csv(consensus_path, low_memory=False)
    selected_ids = set(TRAIN_IDS) | set(TEST_IDS)
    consensus = consensus.loc[consensus["annotation_order"].isin(selected_ids)].copy()
    consensus = consensus.sort_values("annotation_order").reset_index(drop=True)

    if training["annotation_order"].astype(int).tolist() != list(TRAIN_IDS):
        raise ValueError("Training nested OOF table has the wrong selected IDs")
    if test["annotation_order"].astype(int).tolist() != list(TEST_IDS):
        raise ValueError("Held-out table has the wrong selected IDs")
    expected_ids = sorted(selected_ids)
    if final_records["annotation_order"].astype(int).tolist() != expected_ids:
        raise ValueError("Final record table has the wrong selected IDs")
    if consensus["annotation_order"].astype(int).tolist() != expected_ids:
        raise ValueError("Consensus selection has the wrong IDs")
    if epochs["record_uid"].nunique() != len(expected_ids):
        raise ValueError(f"Expected epoch trajectories for exactly {len(expected_ids)} records")
    required_epochs = {
        "record_uid",
        "epoch_index",
        "start_s",
        "end_s",
        "duration_s",
        F_COLUMN,
        "bar_f_lambda",
        "predicted_stage",
        "dominant_band",
        "dream_weight",
        "stage_dream_probability",
        "weighted_dream_increment_s",
        "cumulative_weighted_dream_s",
    }
    missing = sorted(required_epochs - set(epochs.columns))
    if missing:
        raise ValueError(f"Epoch trajectory is missing columns: {missing}")
    if portable.get("schema") != "tpmt.stage_weighted_global_g.v1":
        raise ValueError("Unexpected portable model schema")
    return epochs, training, test, final_records, consensus, portable


def make_record_trajectory(
    source: pd.DataFrame,
    record: pd.Series,
    prediction: pd.Series,
    stage_weights: dict[str, float],
) -> pd.DataFrame:
    """Build a replayable weighted trajectory matching its record-level result."""

    frame = source.loc[source["record_uid"].eq(record["record_uid"])].copy()
    frame = frame.sort_values(["epoch_index", "start_s"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No trajectory for {record['record_uid']}")
    if frame.duplicated(["record_uid", "epoch_index"]).any():
        raise ValueError(f"Duplicate epoch index for {record['record_uid']}")

    for column in TRAJECTORY_COLUMNS_TO_PRESERVE:
        if column in frame.columns:
            frame[f"source_final_global_g_{column}"] = frame[column]

    frame["bar_f_lambda"] = np.clip(
        frame["source_final_global_g_bar_f_lambda"].to_numpy(dtype=float), 0.0, 1.0
    )
    expected_weights = frame["predicted_stage"].astype(str).map(stage_weights)
    if expected_weights.isna().any():
        raise ValueError(f"Unweighted stage in {record['record_uid']}")
    if not np.allclose(
        expected_weights.to_numpy(float),
        frame["source_final_global_g_dream_weight"].to_numpy(float),
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(f"Stage-weight replay mismatch for {record['record_uid']}")
    frame["dream_weight"] = expected_weights.to_numpy(float)
    frame["stage_dream_probability"] = frame["dream_weight"]
    durations = frame["duration_s"].to_numpy(dtype=float)
    frame["apparent_increment_s"] = frame["bar_f_lambda"] * durations
    frame["weighted_dream_increment_s"] = (
        frame["dream_weight"] * frame["bar_f_lambda"] * durations
    )
    frame["cumulative_apparent_s"] = frame["apparent_increment_s"].cumsum()
    frame["cumulative_weighted_dream_s"] = frame[
        "weighted_dream_increment_s"
    ].cumsum()
    frame["g_config_key"] = str(prediction["config_key"])
    frame["trajectory_protocol"] = (
        "descriptive training trajectory using final global g fitted on all selected training IDs"
        if str(record["split"]) == "train"
        else "held-aside test using global g locked from selected training IDs"
    )
    frame["dream_target_s"] = float(record["consensus_mean_s"])
    frame["dream_target_sigma_population_s"] = float(
        record["consensus_sigma_population_s"]
    )
    frame["dream_report_en"] = str(record["dream_report_en"])
    frame["predicted_stage_provenance"] = STAGE_PROVENANCE
    frame["predicted_stage_is_manual_hypnogram"] = False
    frame["predicted_stage_used_by_physical_f"] = False
    frame["predicted_stage_weight_used_for_dream_integral"] = True
    frame["predicted_stage_weight_used_to_calibrate_g"] = True
    frame["dream_time_formula"] = (
        "sum dream_weight[predicted_stage] * bar_f_lambda * duration_s"
    )
    frame["apparent_time_formula"] = "sum bar_f_lambda * duration_s"

    apparent = float(frame["apparent_increment_s"].sum())
    dream = float(frame["weighted_dream_increment_s"].sum())
    if abs(apparent - float(prediction["apparent_time_s"])) > 1e-7:
        raise ValueError(f"Apparent-time replay mismatch for {record['record_uid']}")
    if abs(dream - float(prediction["predicted_dream_s"])) > 1e-7:
        raise ValueError(f"Dream-time replay mismatch for {record['record_uid']}")
    physical = float(durations.sum())
    if not (-1e-9 <= dream <= apparent + 1e-9 <= physical + 1e-9):
        raise ValueError(f"Integral bounds fail for {record['record_uid']}")
    return frame


def plot_record(
    record: pd.Series,
    frame: pd.DataFrame,
    stage_weights: dict[str, float],
    output: Path,
    dpi: int,
) -> None:
    midpoint_min = (frame["start_s"] + frame["end_s"]) / 120.0
    end_min = frame["end_s"] / 60.0
    stage_spans = run_length_spans(frame, "predicted_stage")
    band_spans = run_length_spans(frame, "dominant_band")
    total_width = max(float(frame["end_s"].max() - frame["start_s"].min()) / 60.0, 0.5)
    report, report_font = wrap_report(str(record["dream_report_en"]))

    fig = plt.figure(figsize=(18.0, 10.4))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.82, 1.0], wspace=0.10)
    left = GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[0],
        height_ratios=[4.35, 0.48, 0.48, 2.05],
        hspace=0.08,
    )
    f_ax = fig.add_subplot(left[0])
    stage_ax = fig.add_subplot(left[1], sharex=f_ax)
    band_ax = fig.add_subplot(left[2], sharex=f_ax)
    cumulative_ax = fig.add_subplot(left[3], sharex=f_ax)
    report_ax = fig.add_subplot(outer[1])

    prefix = float(frame["start_s"].min()) / 60.0
    if prefix > 0:
        for axis in (f_ax, stage_ax, band_ax, cumulative_ax):
            axis.axvspan(0.0, prefix, facecolor="#dddddd", hatch="////", alpha=0.55, lw=0)
    for start, end, stage in stage_spans:
        f_ax.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.065, lw=0)
        stage_ax.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.96, lw=0)
    for start, end, band in band_spans:
        band_ax.axvspan(start, end, color=BAND_COLORS[band], alpha=0.96, lw=0)

    physical_line = f_ax.plot(
        midpoint_min,
        frame[F_COLUMN],
        color="#718a98",
        lw=1.15,
        alpha=0.88,
        label=r"physical $f(t)$ before $g$",
    )[0]
    final_line = f_ax.plot(
        midpoint_min,
        frame["bar_f_lambda"],
        color="#15384c",
        lw=2.05,
        label=r"$\bar f(\lambda,t)=g(f(t))$",
    )[0]
    f_ax.set_ylim(-0.03, 1.04)
    f_ax.set_ylabel("Dimensionless value")
    f_ax.grid(True, alpha=0.14)
    present_stages = [stage for stage in STAGE_ORDER if stage in set(frame["predicted_stage"])]
    handles: list[Any] = [physical_line, final_line]
    handles.extend(
        Patch(
            facecolor=STAGE_COLORS[stage],
            label=f"{stage} (p={stage_weights[stage]:g})",
        )
        for stage in present_stages
    )
    f_ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncol=min(len(handles), 7),
        loc="upper left",
    )

    stage_ax.set_ylim(0.0, 1.0)
    stage_ax.set_yticks([0.5], ["GSSC stage\nand weight"], fontsize=7.3)
    stage_ax.tick_params(axis="x", labelbottom=False)
    label_long_spans(stage_ax, stage_spans, total_width)
    band_ax.set_ylim(0.0, 1.0)
    band_ax.set_yticks([0.5], ["Dominant\nspectral band"], fontsize=7.3)
    band_ax.tick_params(axis="x", labelbottom=False)
    label_long_spans(band_ax, band_spans, total_width)
    for axis in (stage_ax, band_ax):
        axis.spines[["top", "right", "bottom"]].set_visible(False)

    cumulative_ax.plot(
        end_min,
        frame["cumulative_apparent_s"] / 60.0,
        color="#963d72",
        lw=2.25,
        label=r"apparent: $\sum \bar f_i\,\Delta t_i$",
    )
    cumulative_ax.plot(
        end_min,
        frame["cumulative_weighted_dream_s"] / 60.0,
        color="#d55262",
        lw=2.20,
        label=r"stage-weighted dream: $\sum p(s_i)\bar f_i\,\Delta t_i$",
    )
    cumulative_ax.axhline(
        float(record["consensus_mean_s"]) / 60.0,
        color="#d17b2f",
        ls="--",
        lw=1.35,
        label="six-LLM dream-duration mean",
    )
    cumulative_ax.set_xlabel("Physical time from EDF start (min)")
    cumulative_ax.set_ylabel("Cumulative time (min)")
    cumulative_ax.grid(True, alpha=0.16)
    cumulative_ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    cumulative_ax.set_xlim(left=0.0, right=max(float(frame["end_s"].max()) / 60.0, 0.5))

    report_ax.axis("off")
    report_ax.text(
        0.0,
        0.995,
        report,
        ha="left",
        va="top",
        fontsize=report_font,
        linespacing=1.16,
        wrap=False,
        transform=report_ax.transAxes,
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_correlation_values(
    training: pd.DataFrame,
    test: pd.DataFrame,
    consensus: pd.DataFrame,
    stage_weights: dict[str, float],
) -> pd.DataFrame:
    metadata_columns = [
        "annotation_order",
        "dream_report_id",
        "record_uid",
        "subject_id",
        "split",
        "manual_final_stage",
        "consensus_mean_s",
        "consensus_sigma_population_s",
    ]
    train_values = training[
        ["annotation_order", "predicted_dream_s", "apparent_time_s", "protocol"]
    ].copy()
    train_values["evaluation_protocol"] = (
        "conditional nested participant-grouped OOF for g given frozen upstream f/stage; "
        "upstream GSSC stage model not cross-fit"
    )
    test_values = test[
        ["annotation_order", "predicted_dream_s", "apparent_time_s", "protocol"]
    ].copy()
    test_values["evaluation_protocol"] = (
        "held-aside test; global g locked from selected training IDs"
    )
    values = pd.concat([train_values, test_values], ignore_index=True).merge(
        consensus[metadata_columns],
        on="annotation_order",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", ""),
    )
    values["dream_time_definition"] = (
        "sum p(predicted_stage) * bar_f_lambda * duration_s"
    )
    values["stage_weights_json"] = json.dumps(stage_weights, sort_keys=True)
    values["stage_color_definition"] = "manual awakening-endpoint stage"
    return values.sort_values("annotation_order").reset_index(drop=True)


def plot_correlation(
    values: pd.DataFrame,
    prediction_column: str,
    ylabel: str,
    output: Path,
    dpi: int,
) -> None:
    target = values["consensus_mean_s"].to_numpy(dtype=float) / 60.0
    predicted = values[prediction_column].to_numpy(dtype=float) / 60.0
    upper = max(float(np.max(target)), float(np.max(predicted)), 1.0) * 1.08
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharex=True, sharey=True)
    panel_specs = [
        ("train", "Training: conditional nested OOF for $g$"),
        ("test", "Test: locked global $g$"),
    ]
    for axis, (split, title) in zip(axes, panel_specs, strict=True):
        group = values.loc[values["split"].eq(split)].copy()
        axis.plot([0.0, upper], [0.0, upper], ls="--", lw=1.1, color="#7f8c8d", zorder=1)
        for stage in STAGE_ORDER:
            stage_group = group.loc[group["manual_final_stage"].eq(stage)]
            if stage_group.empty:
                continue
            axis.scatter(
                stage_group["consensus_mean_s"] / 60.0,
                stage_group[prediction_column] / 60.0,
                s=49,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                linewidth=0.65,
                label=stage,
                zorder=3,
            )
        for row in group.itertuples(index=False):
            axis.annotate(
                str(int(row.annotation_order)),
                (
                    float(row.consensus_mean_s) / 60.0,
                    float(getattr(row, prediction_column)) / 60.0,
                ),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7.1,
                color="#27323a",
            )
        metrics = metric_summary(group, prediction_column)
        axis.text(
            0.03,
            0.97,
            (
                f"n={metrics['n']}\n$r$={metrics['pearson_r']:.3f}\n"
                f"$\\rho$={metrics['spearman_rho']:.3f}\n"
                f"RMSE={metrics['rmse_s'] / 60.0:.2f} min"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.86,
                "edgecolor": "#ccd2d5",
            },
        )
        axis.set_title(title, fontsize=10.5)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.18)
        axis.set_xlabel("Six-LLM dream-duration estimate (min)")
    axes[0].set_ylabel(ylabel)
    present = [stage for stage in STAGE_ORDER if stage in set(values["manual_final_stage"])]
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=STAGE_COLORS[stage],
                markeredgecolor="white",
                label=stage,
            )
            for stage in present
        ],
        title="Manual awakening-endpoint stage",
        loc="lower center",
        ncol=len(present),
        frameon=False,
        fontsize=8.3,
        title_fontsize=8.3,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.008,
        "IDs 10, 19, 29, and 33 excluded as too short. Training is conditional "
        "nested OOF for g; frozen upstream GSSC stages were not cross-fit.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#4b565c",
    )
    fig.subplots_adjust(bottom=0.21, wspace=0.14)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_index(
    output_dir: Path,
    rows: list[dict[str, Any]],
    stage_weights: dict[str, float],
    profile: str,
) -> None:
    weight_text = ", ".join(f"{stage}={stage_weights[stage]:g}" for stage in STAGE_ORDER)
    lines = [
        "# Stage-weighted dream-time figures",
        "",
        "Dream time is defined as",
        "",
        "`T_dream = sum p(predicted_stage) * bar_f(lambda,t) * Delta t`.",
        "",
        f"Weight profile: `{profile}` ({weight_text}). The weights are fixed inputs,",
        "not fitted on Kumral. IDs 10 and 19 are excluded from training and IDs 29",
        "and 33 from test because their recordings are too short.",
        "",
        "The stage ribbon is a frozen exploratory GSSC v0.0.9 hard prediction, not",
        "a manual epoch-wise hypnogram. Training correlation values are conditional",
        "nested participant-grouped OOF predictions for g. Test predictions use the",
        "global g locked from the remaining training records.",
        "",
        "- [Predicted dream time vs estimate](predicted_dream_time_vs_estimate.png)",
        "- [Total apparent time vs estimate](apparent_time_vs_estimate.png)",
        "- [Correlation values](combined_correlation_values.csv)",
        "- [Metrics](metrics.json)",
        "- [QA](QA.json)",
        "",
        "| Split | ID | Record | Epochs | Figure | Replay CSV |",
        "|---|---:|---|---:|---|---|",
    ]
    for row in rows:
        subdir = "training_samples" if row["split"] == "train" else "test_samples"
        lines.append(
            f"| {row['split']} | {row['annotation_order']} | `{row['record_uid']}` | "
            f"{row['epochs']} | [PNG]({subdir}/{row['stem']}.png) | "
            f"[CSV]({subdir}/{row['stem']}.csv) |"
        )
    (output_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.root / "final_plots"
    if args.output_dir.resolve() == args.root.resolve():
        raise ValueError("Output directory must not overwrite the fit directory")
    training_dir = args.output_dir / "training_samples"
    test_dir = args.output_dir / "test_samples"
    training_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    epochs, training, test, final_records, consensus, portable = load_inputs(
        args.root, args.consensus
    )
    stage_weights = {stage: float(portable["stage_weights"][stage]) for stage in STAGE_ORDER}
    profile = str(portable["stage_weight_profile"])
    prediction_by_id = {
        int(row.annotation_order): pd.Series(row._asdict())
        for row in final_records.itertuples(index=False)
    }
    summaries: list[dict[str, Any]] = []
    maximum_weight_error = 0.0
    maximum_apparent_identity_error = 0.0
    maximum_dream_identity_error = 0.0
    maximum_endpoint_error = 0.0
    for _, record in consensus.sort_values("annotation_order").iterrows():
        annotation_order = int(record["annotation_order"])
        prediction = prediction_by_id[annotation_order]
        frame = make_record_trajectory(epochs, record, prediction, stage_weights)
        split = str(record["split"])
        directory = training_dir if split == "train" else test_dir
        stem = f"ID_{annotation_order:03d}_{Path(record['eeg_filename']).stem}"
        csv_path, png_path = directory / f"{stem}.csv", directory / f"{stem}.png"
        frame.to_csv(csv_path, index=False)
        plot_record(record, frame, stage_weights, png_path, args.dpi)

        duration = frame["duration_s"].to_numpy(float)
        bar_f = frame["bar_f_lambda"].to_numpy(float)
        weight = frame["dream_weight"].to_numpy(float)
        expected_weight = frame["predicted_stage"].astype(str).map(stage_weights).to_numpy(float)
        apparent_increment = bar_f * duration
        dream_increment = weight * bar_f * duration
        weight_error = float(np.max(np.abs(weight - expected_weight)))
        apparent_error = float(
            np.max(np.abs(frame["apparent_increment_s"].to_numpy(float) - apparent_increment))
        )
        dream_error = float(
            np.max(
                np.abs(
                    frame["weighted_dream_increment_s"].to_numpy(float) - dream_increment
                )
            )
        )
        apparent = float(apparent_increment.sum())
        dream = float(dream_increment.sum())
        endpoint_error = max(
            abs(apparent - float(prediction["apparent_time_s"])),
            abs(dream - float(prediction["predicted_dream_s"])),
        )
        maximum_weight_error = max(maximum_weight_error, weight_error)
        maximum_apparent_identity_error = max(maximum_apparent_identity_error, apparent_error)
        maximum_dream_identity_error = max(maximum_dream_identity_error, dream_error)
        maximum_endpoint_error = max(maximum_endpoint_error, endpoint_error)
        summaries.append(
            {
                "annotation_order": annotation_order,
                "split": split,
                "record_uid": str(record["record_uid"]),
                "stem": stem,
                "epochs": len(frame),
                "physical_s": float(duration.sum()),
                "apparent_s": apparent,
                "weighted_dream_s": dream,
                "weighted_maximum_s": float(np.sum(weight * duration)),
                "target_s": float(record["consensus_mean_s"]),
                "csv_sha256": sha256(csv_path),
                "png_sha256": sha256(png_path),
            }
        )
        print(f"wrote {split} ID {annotation_order}: {png_path.name}", flush=True)

    correlation = build_correlation_values(training, test, consensus, stage_weights)
    correlation.to_csv(args.output_dir / "combined_correlation_values.csv", index=False)
    plot_correlation(
        correlation,
        "predicted_dream_s",
        "Stage-weighted predicted dream time (min)",
        args.output_dir / "predicted_dream_time_vs_estimate.png",
        args.dpi,
    )
    plot_correlation(
        correlation,
        "apparent_time_s",
        "Total apparent time (min)",
        args.output_dir / "apparent_time_vs_estimate.png",
        args.dpi,
    )
    metrics = {
        outcome: {
            split: metric_summary(correlation.loc[correlation["split"].eq(split)], column)
            for split in ("train", "test")
        }
        for outcome, column in (
            ("predicted_dream_time_vs_estimate", "predicted_dream_s"),
            ("apparent_time_vs_estimate", "apparent_time_s"),
        )
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_index(args.output_dir, summaries, stage_weights, profile)

    sample_png_count = len(list(training_dir.glob("*.png"))) + len(
        list(test_dir.glob("*.png"))
    )
    sample_csv_count = len(list(training_dir.glob("*.csv"))) + len(
        list(test_dir.glob("*.csv"))
    )
    bounds_ok = all(
        -1e-9
        <= row["weighted_dream_s"]
        <= row["apparent_s"] + 1e-9
        <= row["physical_s"] + 1e-9
        for row in summaries
    )
    qa = {
        "passed": bool(
            sample_png_count == len(TRAIN_IDS) + len(TEST_IDS)
            and sample_csv_count == len(TRAIN_IDS) + len(TEST_IDS)
            and len(correlation) == len(TRAIN_IDS) + len(TEST_IDS)
            and maximum_weight_error <= 1e-15
            and maximum_apparent_identity_error <= 1e-10
            and maximum_dream_identity_error <= 1e-10
            and maximum_endpoint_error <= 1e-7
            and bounds_ok
        ),
        "stage_provenance": STAGE_PROVENANCE,
        "stage_weight_profile": profile,
        "stage_weights": stage_weights,
        "training_ids": list(TRAIN_IDS),
        "heldout_ids": list(TEST_IDS),
        "excluded_too_short_ids": [10, 19, 29, 33],
        "stage_used_by_physical_f": False,
        "predicted_stage_weight_used_for_dream_integral": True,
        "predicted_stage_weight_used_to_calibrate_g": True,
        "sample_trajectory_protocol": (
            "same final global g fitted on selected training IDs for all selected records"
        ),
        "training_correlation_protocol": (
            "conditional nested participant-grouped OOF for g given frozen upstream f/stage; "
            "upstream stage model not cross-fit"
        ),
        "test_trajectory_protocol": "locked global g from selected training IDs",
        "sample_png_count": sample_png_count,
        "sample_csv_count": sample_csv_count,
        "correlation_row_count": len(correlation),
        "correlation_png_count": 2,
        "maximum_stage_weight_abs_error": maximum_weight_error,
        "maximum_apparent_increment_identity_abs_error_s": maximum_apparent_identity_error,
        "maximum_weighted_dream_increment_identity_abs_error_s": maximum_dream_identity_error,
        "maximum_record_endpoint_abs_error_s": maximum_endpoint_error,
        "all_records_satisfy_0_le_weighted_dream_le_apparent_le_physical": bounds_ok,
        "input_sha256": {
            "all_epoch_trajectories.csv": sha256(args.root / "all_epoch_trajectories.csv"),
            "training_nested_oof_predictions.csv": sha256(
                args.root / "training_nested_oof_predictions.csv"
            ),
            "heldout_test_predictions.csv": sha256(args.root / "heldout_test_predictions.csv"),
            "all_final_model_record_predictions.csv": sha256(
                args.root / "all_final_model_record_predictions.csv"
            ),
            "portable_stage_weighted_global_g.json": sha256(
                args.root / "portable_stage_weighted_global_g.json"
            ),
            "consensus": sha256(args.consensus),
        },
        "records": summaries,
    }
    write_json(args.output_dir / "QA.json", qa)
    if not qa["passed"]:
        raise RuntimeError("Stage-weighted plot QA failed; inspect QA.json")
    print(
        f"completed {len(summaries)} sample figures + 2 correlation figures; "
        f"maximum endpoint error={maximum_endpoint_error:.3g} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
