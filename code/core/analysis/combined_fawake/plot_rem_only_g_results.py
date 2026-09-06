#!/usr/bin/env python3
"""Create the final REM-gated dream-time figures and replayable CSV files.

All 37 sample trajectories use the same final global ``g`` fitted on training
IDs 6--25.  The training correlation panel separately reports conditional
nested-OOF performance for ``g`` given the frozen upstream physical ``f`` and
exploratory stage model; that upstream stage model was not cross-fit.  The test
panel uses the same globally locked ``g`` as the sample trajectories.

The sleep stages are exploratory hard predictions, not manual hypnograms.
They do not change the locked physical self-similarity ``f``.  In this REM-only
calibration, however, the hard REM label *does* define the dream-time mask and
therefore participates in fitting ``g`` on the training set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
from math import comb
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "outputs/dense_repeat_fawake_v3/rem_only_g_gssc_v2_confirmed"
DEFAULT_CONSENSUS = (
    REPO_ROOT
    / "outputs/combined_fawake_v2/kappa_fit_v1/"
    "kumral_dream_duration_6llm_consensus.csv"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "final_plots_v2"

STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "W": "#efb366",
    "N1": "#73b3e7",
    "N2": "#77b977",
    "N3": "#173f35",
    "REM": "#d55262",
}
BAND_ORDER = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]
BAND_COLORS = {
    "delta": "#273c75",
    "theta": "#487eb0",
    "alpha": "#44bd32",
    "sigma": "#e1b12c",
    "beta": "#e84118",
    "gamma": "#8c3fa4",
}
STAGE_PROVENANCE = (
    "exploratory hard prediction from pretrained GSSC v0.0.9 using C3-TP10/C4-TP9, "
    "official five-class argmax and loudest-vote; not a manual epoch-wise hypnogram"
)
TRAJECTORY_COLUMNS_TO_PRESERVE = [
    "bar_f_lambda",
    "apparent_increment_s",
    "dream_increment_s",
    "cumulative_apparent_s",
    "cumulative_dream_s",
    "g_config_key",
]


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
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
        json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bernstein_from_controls(values: np.ndarray, control_points: Iterable[float]) -> np.ndarray:
    """Evaluate an endpoint-fixed Bernstein polynomial from decoded controls."""
    x = np.asarray(values, dtype=float)
    controls = np.asarray(list(control_points), dtype=float)
    if controls.ndim != 1 or len(controls) < 2:
        raise ValueError("Bernstein control_points must be a one-dimensional sequence")
    degree = len(controls) - 1
    result = np.zeros_like(x)
    for index, control in enumerate(controls):
        result += (
            control
            * comb(degree, index)
            * np.power(x, index)
            * np.power(1.0 - x, degree - index)
        )
    return result


def run_length_spans(frame: pd.DataFrame, column: str) -> list[tuple[float, float, str]]:
    ordered = frame.sort_values(["epoch_index", "start_s"])
    if ordered.empty:
        return []
    first = ordered.iloc[0]
    start, end, value = (
        float(first["start_s"]) / 60.0,
        float(first["end_s"]) / 60.0,
        str(first[column]),
    )
    spans: list[tuple[float, float, str]] = []
    for row in ordered.iloc[1:].itertuples(index=False):
        current = str(getattr(row, column))
        row_start, row_end = float(row.start_s) / 60.0, float(row.end_s) / 60.0
        if current == value and abs(row_start - end) < 1.0e-8:
            end = row_end
        else:
            spans.append((start, end, value))
            start, end, value = row_start, row_end, current
    spans.append((start, end, value))
    return spans


def label_long_spans(
    axis: plt.Axes,
    spans: Iterable[tuple[float, float, str]],
    total_width_min: float,
) -> None:
    minimum_width = max(0.8, 0.035 * total_width_min)
    for start, end, value in spans:
        if end - start >= minimum_width:
            axis.text(
                0.5 * (start + end),
                0.5,
                value,
                color="white",
                fontsize=6.8,
                ha="center",
                va="center",
                fontweight="bold",
                clip_on=True,
            )


def wrap_report(report: str) -> tuple[str, float]:
    """Wrap without editing the text content; return a size that avoids clipping."""
    words = len(re.findall(r"\b[\w'-]+\b", report, flags=re.UNICODE))
    if words <= 110:
        fontsize, width = 9.7, 47
    elif words <= 220:
        fontsize, width = 8.5, 53
    elif words <= 340:
        fontsize, width = 7.3, 59
    elif words <= 410:
        fontsize, width = 6.5, 65
    else:
        fontsize, width = 6.0, 70
    paragraphs = [
        textwrap.fill(
            paragraph.strip(),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        for paragraph in str(report).splitlines()
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs), fontsize


def load_inputs(
    root: Path,
    consensus_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epochs = pd.read_csv(root / "all_epoch_trajectories.csv", low_memory=False)
    training = pd.read_csv(root / "training_nested_oof_predictions.csv")
    test = pd.read_csv(root / "heldout_test_predictions.csv")
    final_records = pd.read_csv(root / "all_final_model_record_predictions.csv")
    consensus = pd.read_csv(consensus_path, low_memory=False)
    consensus = consensus.loc[consensus["annotation_order"].between(6, 42)].copy()

    if sorted(consensus.loc[consensus["split"].eq("train"), "annotation_order"]) != list(
        range(6, 26)
    ):
        raise ValueError("Consensus training split must be IDs 6--25")
    if sorted(consensus.loc[consensus["split"].eq("test"), "annotation_order"]) != list(
        range(26, 43)
    ):
        raise ValueError("Consensus test split must be IDs 26--42")
    if sorted(training["annotation_order"].astype(int)) != list(range(6, 26)):
        raise ValueError("Training nested OOF table must contain IDs 6--25")
    if sorted(test["annotation_order"].astype(int)) != list(range(26, 43)):
        raise ValueError("Held-out table must contain IDs 26--42")
    if epochs["record_uid"].nunique() != 37:
        raise ValueError("Expected epoch trajectories for exactly 37 records")
    if sorted(final_records["annotation_order"].astype(int)) != list(range(6, 43)):
        raise ValueError("Final global-g record table must contain IDs 6--42")
    required_epochs = {
        "record_uid",
        "epoch_index",
        "start_s",
        "end_s",
        "duration_s",
        "f_repeat_micro_v3",
        "bar_f_lambda",
        "predicted_stage",
        "dominant_band",
        "is_REM",
    }
    missing = sorted(required_epochs - set(epochs.columns))
    if missing:
        raise ValueError(f"Epoch trajectory is missing columns: {missing}")
    return epochs, training, test, final_records, consensus


def make_record_trajectory(
    source: pd.DataFrame,
    record: pd.Series,
    prediction: pd.Series,
) -> pd.DataFrame:
    """Build a self-contained trajectory whose endpoint matches its evaluation row."""
    frame = source.loc[source["record_uid"].eq(record["record_uid"])].copy()
    frame = frame.sort_values(["epoch_index", "start_s"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No trajectory for {record['record_uid']}")
    if frame.duplicated(["record_uid", "epoch_index"]).any():
        raise ValueError(f"Duplicate epoch index for {record['record_uid']}")

    # Preserve an immutable copy of the source calculation columns.  Every
    # sample figure then replays exactly this same final global g.
    for column in TRAJECTORY_COLUMNS_TO_PRESERVE:
        if column in frame.columns:
            frame[f"source_final_global_g_{column}"] = frame[column]

    values = frame["source_final_global_g_bar_f_lambda"].to_numpy(dtype=float)
    config_key = str(prediction["config_key"])
    if str(record["split"]) == "train":
        trajectory_protocol = (
            "descriptive training trajectory; same final global g fitted on all IDs 6--25"
        )
    else:
        trajectory_protocol = "held-aside test; same final global g locked from IDs 6--25"

    durations = frame["duration_s"].to_numpy(dtype=float)
    is_rem = frame["predicted_stage"].astype(str).eq("REM").to_numpy(dtype=bool)
    if not np.array_equal(is_rem.astype(np.int8), frame["is_REM"].to_numpy(dtype=np.int8)):
        raise ValueError(f"is_REM disagrees with predicted_stage for {record['record_uid']}")
    frame["bar_f_lambda"] = np.clip(values, 0.0, 1.0)
    frame["apparent_increment_s"] = frame["bar_f_lambda"] * durations
    frame["dream_increment_s"] = frame["apparent_increment_s"] * is_rem.astype(float)
    frame["cumulative_apparent_s"] = frame["apparent_increment_s"].cumsum()
    frame["cumulative_dream_s"] = frame["dream_increment_s"].cumsum()
    frame["g_config_key"] = config_key
    frame["trajectory_protocol"] = trajectory_protocol
    frame["dream_target_s"] = float(record["consensus_mean_s"])
    frame["dream_target_sigma_population_s"] = float(
        record["consensus_sigma_population_s"]
    )
    frame["dream_report_en"] = str(record["dream_report_en"])
    frame["predicted_stage_provenance"] = STAGE_PROVENANCE
    frame["predicted_stage_is_manual_hypnogram"] = False
    frame["predicted_stage_used_by_physical_f"] = False
    frame["predicted_REM_mask_used_for_dream_integral"] = True
    frame["predicted_REM_mask_used_to_calibrate_g"] = True
    frame["dream_time_formula"] = (
        "sum I[predicted_stage == REM] * bar_f_lambda * duration_s"
    )
    frame["apparent_time_formula"] = "sum bar_f_lambda * duration_s"

    apparent = float(frame["apparent_increment_s"].sum())
    dream = float(frame["dream_increment_s"].sum())
    expected_apparent = float(prediction["apparent_time_s"])
    expected_dream = float(prediction["predicted_dream_s"])
    if abs(apparent - expected_apparent) > 1.0e-7:
        raise ValueError(
            f"Apparent-time replay mismatch for {record['record_uid']}: "
            f"{apparent} versus {expected_apparent}"
        )
    if abs(dream - expected_dream) > 1.0e-7:
        raise ValueError(
            f"Dream-time replay mismatch for {record['record_uid']}: "
            f"{dream} versus {expected_dream}"
        )
    physical = float(durations.sum())
    if not (-1.0e-9 <= dream <= apparent + 1.0e-9 <= physical + 1.0e-9):
        raise ValueError(f"Bound 0 <= dream <= apparent <= physical fails for {record['record_uid']}")
    return frame


def plot_record(record: pd.Series, frame: pd.DataFrame, output: Path, dpi: int) -> None:
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
        frame["f_repeat_micro_v3"],
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
    handles.extend(Patch(facecolor=STAGE_COLORS[stage], label=stage) for stage in present_stages)
    f_ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.8,
        ncol=min(len(handles), 7),
        loc="upper left",
    )
    # Intentionally no figure title and no top-axis title.

    stage_ax.set_ylim(0.0, 1.0)
    stage_ax.set_yticks([0.5], ["Exploratory\npredicted stage"], fontsize=7.3)
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
        frame["cumulative_dream_s"] / 60.0,
        color="#d55262",
        lw=2.20,
        label=r"REM-gated dream: $\sum I_{\rm REM}\bar f_i\,\Delta t_i$",
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

    # The right panel contains the report and nothing else, as requested.
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


def metric_summary(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    target = frame["consensus_mean_s"].to_numpy(dtype=float)
    predicted = frame[prediction_column].to_numpy(dtype=float)
    residual = predicted - target
    pearson = pearsonr(target, predicted)
    spearman = spearmanr(target, predicted)
    return {
        "n": len(frame),
        "mae_s": float(np.mean(np.abs(residual))),
        "rmse_s": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


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
        ("test", "Test: locked global $g$ from IDs 6--25"),
    ]
    for axis, (split, title) in zip(axes, panel_specs):
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
                (float(row.consensus_mean_s) / 60.0, float(getattr(row, prediction_column)) / 60.0),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7.1,
                color="#27323a",
            )
        metrics = metric_summary(group, prediction_column)
        axis.text(
            0.03,
            0.97,
            f"n={metrics['n']}\n$r$={metrics['pearson_r']:.3f}\n$\\rho$={metrics['spearman_rho']:.3f}\nRMSE={metrics['rmse_s'] / 60.0:.2f} min",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.86, "edgecolor": "#ccd2d5"},
        )
        axis.set_title(title, fontsize=10.5)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.18)
        axis.set_xlabel("Six-LLM dream-duration estimate (min)")
    axes[0].set_ylabel(ylabel)
    stage_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=STAGE_COLORS[stage],
            markeredgecolor="white",
            label=stage,
        )
        for stage in STAGE_ORDER
        if stage in set(values["manual_final_stage"])
    ]
    fig.legend(
        handles=stage_handles,
        title="Manual awakening-endpoint stage",
        loc="lower center",
        ncol=len(stage_handles),
        frameon=False,
        fontsize=8.3,
        title_fontsize=8.3,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.text(
        0.5,
        0.008,
        "Training is conditional nested OOF for g given frozen upstream f/stage; "
        "the upstream exploratory stage model was not cross-fit.",
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#4b565c",
    )
    fig.subplots_adjust(bottom=0.21, wspace=0.14)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_correlation_values(
    training: pd.DataFrame,
    test: pd.DataFrame,
    consensus: pd.DataFrame,
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
    metadata = consensus[metadata_columns].copy()
    train_values = training[
        ["annotation_order", "predicted_dream_s", "apparent_time_s", "protocol"]
    ].copy()
    train_values["evaluation_protocol"] = (
        "conditional nested participant-grouped OOF for g given frozen upstream f/stage; "
        "upstream stage model not cross-fit"
    )
    test_values = test[
        ["annotation_order", "predicted_dream_s", "apparent_time_s", "protocol"]
    ].copy()
    test_values["evaluation_protocol"] = "held-aside test; global g locked from IDs 6--25"
    values = pd.concat([train_values, test_values], ignore_index=True).merge(
        metadata,
        on="annotation_order",
        how="left",
        validate="one_to_one",
        suffixes=("_prediction", ""),
    )
    values["dream_time_definition"] = (
        "sum I[predicted_stage == REM] * bar_f_lambda * duration_s"
    )
    values["stage_color_definition"] = "manual awakening-endpoint stage"
    return values.sort_values("annotation_order").reset_index(drop=True)


def write_index(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# REM-gated dream-time figures",
        "",
        "Dream time is defined as",
        "",
        "`T_dream = sum I[predicted_stage == REM] * bar_f(lambda,t) * Delta t`.",
        "",
        "The stage ribbon is an **exploratory hard model prediction**, not a manual",
        "epoch-wise hypnogram. It does not enter the locked physical self-similarity",
        "`f(t)`, but its REM indicator now defines the dream-time mask and consequently",
        "participates in training-only calibration of `g`. A record with no predicted",
        "REM therefore has a flat zero dream-time curve.",
        "",
        "All 37 trajectory figures use the **same final global `g`**, fitted once on",
        "training IDs 6--25. Training trajectories are therefore descriptive/in-sample;",
        "test trajectories are held aside. The right-hand panel",
        "contains only the complete, unedited English dream report.",
        "The training correlation panels instead use conditional nested participant-",
        "grouped OOF predictions for `g` given frozen upstream `f`/stage. The upstream",
        "exploratory stage model was not cross-fit; these panels are therefore not an",
        "end-to-end OOF estimate of the complete EEG-to-time pipeline.",
        "",
        "- [Predicted dream time vs estimate](predicted_dream_time_vs_estimate.png)",
        "- [Total apparent time vs estimate](apparent_time_vs_estimate.png)",
        "- [Correlation values](combined_correlation_values.csv)",
        "- [Metrics](metrics.json)",
        "- [QA](QA.json)",
        "",
        "| Split | ID | Record | Epochs | Predicted REM epochs | Figure | Replay CSV |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        subdir = f"{row['split']}ing_samples" if row["split"] == "train" else "test_samples"
        lines.append(
            f"| {row['split']} | {row['annotation_order']} | `{row['record_uid']}` | "
            f"{row['epochs']} | {row['rem_epochs']} | "
            f"[PNG]({subdir}/{row['stem']}.png) | [CSV]({subdir}/{row['stem']}.csv) |"
        )
    (output_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=190)
    args = parser.parse_args(argv)
    if args.output_dir.resolve() == args.root.resolve():
        raise ValueError("Output directory must not overwrite the REM-only fit directory")
    training_dir = args.output_dir / "training_samples"
    test_dir = args.output_dir / "test_samples"
    training_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    epochs, training, test, final_records, consensus = load_inputs(args.root, args.consensus)
    prediction_by_id = {
        int(row.annotation_order): pd.Series(row._asdict())
        for row in final_records.itertuples(index=False)
    }
    summaries: list[dict[str, Any]] = []
    maximum_apparent_identity_error = 0.0
    maximum_dream_identity_error = 0.0
    maximum_endpoint_error = 0.0
    for _, record in consensus.sort_values("annotation_order").iterrows():
        annotation_order = int(record["annotation_order"])
        prediction = prediction_by_id[annotation_order]
        frame = make_record_trajectory(epochs, record, prediction)
        split = str(record["split"])
        directory = training_dir if split == "train" else test_dir
        stem = f"ID_{annotation_order:03d}_{Path(record['eeg_filename']).stem}"
        csv_path, png_path = directory / f"{stem}.csv", directory / f"{stem}.png"
        frame.to_csv(csv_path, index=False)
        plot_record(record, frame, png_path, args.dpi)

        durations = frame["duration_s"].to_numpy(dtype=float)
        apparent_identity = float(
            np.max(
                np.abs(
                    frame["apparent_increment_s"].to_numpy(dtype=float)
                    - frame["bar_f_lambda"].to_numpy(dtype=float) * durations
                )
            )
        )
        dream_identity = float(
            np.max(
                np.abs(
                    frame["dream_increment_s"].to_numpy(dtype=float)
                    - frame["bar_f_lambda"].to_numpy(dtype=float)
                    * durations
                    * frame["is_REM"].to_numpy(dtype=float)
                )
            )
        )
        apparent = float(frame["apparent_increment_s"].sum())
        dream = float(frame["dream_increment_s"].sum())
        endpoint_error = max(
            abs(apparent - float(prediction["apparent_time_s"])),
            abs(dream - float(prediction["predicted_dream_s"])),
        )
        maximum_apparent_identity_error = max(maximum_apparent_identity_error, apparent_identity)
        maximum_dream_identity_error = max(maximum_dream_identity_error, dream_identity)
        maximum_endpoint_error = max(maximum_endpoint_error, endpoint_error)
        summaries.append(
            {
                "annotation_order": annotation_order,
                "split": split,
                "record_uid": str(record["record_uid"]),
                "stem": stem,
                "epochs": len(frame),
                "rem_epochs": int(frame["is_REM"].sum()),
                "physical_s": float(durations.sum()),
                "apparent_s": apparent,
                "dream_s": dream,
                "target_s": float(record["consensus_mean_s"]),
                "csv_sha256": sha256(csv_path),
                "png_sha256": sha256(png_path),
            }
        )
        print(f"wrote {split} ID {annotation_order}: {png_path.name}", flush=True)

    correlation = build_correlation_values(training, test, consensus)
    correlation.to_csv(args.output_dir / "combined_correlation_values.csv", index=False)
    plot_correlation(
        correlation,
        "predicted_dream_s",
        "REM-gated predicted dream time (min)",
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
    write_index(args.output_dir, summaries)

    sample_png_count = len(list(training_dir.glob("*.png"))) + len(list(test_dir.glob("*.png")))
    sample_csv_count = len(list(training_dir.glob("*.csv"))) + len(list(test_dir.glob("*.csv")))
    bounds_ok = all(
        -1.0e-9 <= row["dream_s"] <= row["apparent_s"] + 1.0e-9 <= row["physical_s"] + 1.0e-9
        for row in summaries
    )
    qa = {
        "passed": bool(
            sample_png_count == 37
            and sample_csv_count == 37
            and len(correlation) == 37
            and maximum_apparent_identity_error <= 1.0e-10
            and maximum_dream_identity_error <= 1.0e-10
            and maximum_endpoint_error <= 1.0e-7
            and bounds_ok
        ),
        "stage_provenance": STAGE_PROVENANCE,
        "stage_used_by_physical_f": False,
        "predicted_REM_mask_used_for_dream_integral": True,
        "predicted_REM_mask_used_to_calibrate_g": True,
        "sample_trajectory_protocol": "same final global g fitted on training IDs 6--25 for all 37 records",
        "training_correlation_protocol": (
            "conditional nested participant-grouped OOF for g given frozen upstream f/stage; "
            "upstream stage model not cross-fit"
        ),
        "test_trajectory_protocol": "locked global g from IDs 6--25",
        "sample_png_count": sample_png_count,
        "sample_csv_count": sample_csv_count,
        "correlation_row_count": len(correlation),
        "correlation_png_count": 2,
        "maximum_apparent_increment_identity_abs_error_s": maximum_apparent_identity_error,
        "maximum_dream_increment_identity_abs_error_s": maximum_dream_identity_error,
        "maximum_record_endpoint_abs_error_s": maximum_endpoint_error,
        "all_records_satisfy_0_le_dream_le_apparent_le_physical": bounds_ok,
        "input_sha256": {
            "all_epoch_trajectories.csv": sha256(args.root / "all_epoch_trajectories.csv"),
            "training_nested_oof_predictions.csv": sha256(
                args.root / "training_nested_oof_predictions.csv"
            ),
            "heldout_test_predictions.csv": sha256(args.root / "heldout_test_predictions.csv"),
            "all_final_model_record_predictions.csv": sha256(
                args.root / "all_final_model_record_predictions.csv"
            ),
            "consensus": sha256(args.consensus),
        },
        "records": summaries,
    }
    write_json(args.output_dir / "QA.json", qa)
    if not qa["passed"]:
        raise RuntimeError("Final plot QA failed; inspect QA.json")
    print(
        f"completed 37 sample figures + 2 correlation figures; "
        f"maximum endpoint error={maximum_endpoint_error:.3g} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
