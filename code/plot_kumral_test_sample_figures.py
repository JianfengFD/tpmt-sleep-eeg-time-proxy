#!/usr/bin/env python3
"""Create the paper-candidate Kumral test-record figures and redraw tables.

The script deliberately does not copy the large source EDF files.  It reads the
locked epoch-level TpMT trajectories from the main analysis and, when needed,
streams the requested EDF member from the public Kumral ZIP archive to derive a
compact one-record-per-second summary of the unfiltered C3--TP10 EEG.  Once the
two CSV files for a record have been written, the figure can be redrawn without
opening the source EDF/ZIP again.

Default test IDs (after the prespecified short-record exclusions) are
26, 27, 28, 30, 31, 32, and 34--42.  ID 35 is
Sub-006_task_sleep-awak003.edf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import textwrap
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
CANDIDATE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
ANALYSIS_ROOT = (
    REPO_ROOT
    / "outputs/dense_repeat_fawake_v3/stage_weighted_g_literature_v1"
)
CONSENSUS_PATH = (
    REPO_ROOT
    / "outputs/combined_fawake_v2/kappa_fit_v1/"
    "kumral_dream_duration_6llm_consensus.csv"
)
KUMRAL_ZIP = REPO_ROOT / "data/Kumral_2023/Kumral et al., 2023.zip"
DATA_DIR = CANDIDATE_ROOT / "data/test_samples"
FIGURE_DIR = CANDIDATE_ROOT / "figures/test_samples"

TEST_IDS = (26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42)
STAGE_ORDER = ("W", "N1", "N2", "N3", "REM")
STAGE_COLORS = {
    "W": "#efb366",
    "N1": "#73b3e7",
    "N2": "#77b977",
    "N3": "#173f35",
    "REM": "#d55262",
}

DATASET_METADATA = {
    "dream_set": 19,
    # DREAM v9 `Data records.csv`, Set ID 19.  Its Case ID is the source
    # record's within-set identifier (the 66-record Kumral set).
    "dream_set_row_within_66": {
        26: 3,
        27: 4,
        28: 10,
        30: 9,
        31: 12,
        32: 13,
        34: 15,
        35: 18,
        36: 22,
        37: 23,
        38: 24,
        39: 29,
        40: 30,
        41: 31,
        42: 32,
    },
    "global_data_record_key_id": {
        26: 2810,
        27: 2811,
        28: 2817,
        30: 2816,
        31: 2819,
        32: 2820,
        34: 2822,
        35: 2825,
        36: 2829,
        37: 2830,
        38: 2831,
        39: 2836,
        40: 2837,
        41: 2838,
        42: 2839,
    },
    "data_doi": "10.60493/31mg4-mfq53",
    "kumral_article_doi": "10.1016/j.isci.2025.113032",
    "gssc_article_doi": "10.3389/fninf.2023.1086634",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _field(block: bytes, offset: int, width: int, count: int) -> list[str]:
    return [
        block[offset + index * width : offset + (index + 1) * width]
        .decode("latin-1")
        .strip()
        for index in range(count)
    ]


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected end of EDF stream; wanted {remaining} more bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_float(text: str) -> float:
    return float(text.strip())


def _parse_int(text: str) -> int:
    return int(float(text.strip()))


def read_bipolar_eeg_summary_from_zip(
    archive_path: Path,
    edf_filename: str,
    first_channel: str = "C3",
    reference_channel: str = "TP10",
) -> pd.DataFrame:
    """Stream an EDF member and summarize unfiltered bipolar EEG per data record.

    EDF records are normally one second in this dataset.  The exported values
    retain the raw bipolar minimum, maximum and distributional quantiles in uV;
    no temporal or frequency-domain filter is applied.
    """

    with zipfile.ZipFile(archive_path) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if Path(item.filename).name == edf_filename
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one ZIP member named {edf_filename!r}; found {len(candidates)}"
            )
        info = candidates[0]
        with archive.open(info) as handle:
            fixed = _read_exact(handle, 256)
            header_bytes = _parse_int(fixed[184:192].decode("ascii"))
            n_records = _parse_int(fixed[236:244].decode("ascii"))
            record_duration_s = _parse_float(fixed[244:252].decode("ascii"))
            n_signals = _parse_int(fixed[252:256].decode("ascii"))
            signal_header = _read_exact(handle, header_bytes - 256)

            offset = 0
            labels = _field(signal_header, offset, 16, n_signals)
            offset += 16 * n_signals
            offset += 80 * n_signals  # transducer
            physical_dimensions = _field(signal_header, offset, 8, n_signals)
            offset += 8 * n_signals
            physical_minima = np.asarray(
                [_parse_float(value) for value in _field(signal_header, offset, 8, n_signals)]
            )
            offset += 8 * n_signals
            physical_maxima = np.asarray(
                [_parse_float(value) for value in _field(signal_header, offset, 8, n_signals)]
            )
            offset += 8 * n_signals
            digital_minima = np.asarray(
                [_parse_float(value) for value in _field(signal_header, offset, 8, n_signals)]
            )
            offset += 8 * n_signals
            digital_maxima = np.asarray(
                [_parse_float(value) for value in _field(signal_header, offset, 8, n_signals)]
            )
            offset += 8 * n_signals
            offset += 80 * n_signals  # prefiltering
            samples_per_record = np.asarray(
                [_parse_int(value) for value in _field(signal_header, offset, 8, n_signals)],
                dtype=int,
            )

            try:
                first_index = labels.index(first_channel)
                reference_index = labels.index(reference_channel)
            except ValueError as error:
                raise ValueError(
                    f"Channels {first_channel}/{reference_channel} not found; labels={labels}"
                ) from error
            if physical_dimensions[first_index].lower() != physical_dimensions[
                reference_index
            ].lower():
                raise ValueError("Bipolar source channels use different physical units")
            unit = physical_dimensions[first_index] or "unknown"

            byte_offsets = np.concatenate(([0], np.cumsum(samples_per_record) * 2))
            record_bytes = int(byte_offsets[-1])
            if n_records < 0:
                n_records = (info.file_size - header_bytes) // record_bytes

            def physical(signal_index: int, raw_record: bytes) -> np.ndarray:
                values = np.frombuffer(
                    raw_record,
                    dtype="<i2",
                    count=int(samples_per_record[signal_index]),
                    offset=int(byte_offsets[signal_index]),
                ).astype(np.float64)
                digital_span = digital_maxima[signal_index] - digital_minima[signal_index]
                if digital_span <= 0:
                    raise ValueError(f"Invalid digital range for {labels[signal_index]}")
                scale = (
                    physical_maxima[signal_index] - physical_minima[signal_index]
                ) / digital_span
                return (values - digital_minima[signal_index]) * scale + physical_minima[
                    signal_index
                ]

            rows: list[dict[str, float | int | str]] = []
            for record_index in range(n_records):
                raw_record = _read_exact(handle, record_bytes)
                first = physical(first_index, raw_record)
                reference = physical(reference_index, raw_record)
                if first.size != reference.size:
                    raise ValueError("Bipolar channels have different sample counts")
                bipolar = first - reference
                q01, q05, q50, q95, q99 = np.quantile(
                    bipolar, [0.01, 0.05, 0.50, 0.95, 0.99]
                )
                start_s = record_index * record_duration_s
                end_s = start_s + record_duration_s
                rows.append(
                    {
                        "edf_record_index": record_index,
                        "start_s": start_s,
                        "end_s": end_s,
                        "midpoint_s": 0.5 * (start_s + end_s),
                        "channel": f"{first_channel}-{reference_channel}",
                        "physical_unit": unit,
                        "sample_frequency_hz": first.size / record_duration_s,
                        "samples_in_summary_bin": first.size,
                        "raw_min": float(np.min(bipolar)),
                        "raw_q01": float(q01),
                        "raw_q05": float(q05),
                        "raw_median": float(q50),
                        "raw_q95": float(q95),
                        "raw_q99": float(q99),
                        "raw_max": float(np.max(bipolar)),
                    }
                )
            extra = handle.read(1)
            if extra:
                raise ValueError("EDF stream contains unexplained bytes after the declared records")

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No EEG records decoded from {edf_filename}")
    return result


def run_length_spans(
    frame: pd.DataFrame, column: str
) -> list[tuple[float, float, str]]:
    ordered = frame.sort_values(["epoch_index", "start_s"])
    first = ordered.iloc[0]
    start = float(first["start_s"]) / 60.0
    end = float(first["end_s"]) / 60.0
    value = str(first[column])
    spans: list[tuple[float, float, str]] = []
    for row in ordered.iloc[1:].itertuples(index=False):
        current = str(getattr(row, column))
        row_start = float(row.start_s) / 60.0
        row_end = float(row.end_s) / 60.0
        if current == value and math.isclose(row_start, end, abs_tol=1.0e-8):
            end = row_end
        else:
            spans.append((start, end, value))
            start, end, value = row_start, row_end, current
    spans.append((start, end, value))
    return spans


def label_long_stage_spans(
    axis: plt.Axes,
    spans: Iterable[tuple[float, float, str]],
    total_width_min: float,
) -> None:
    minimum_width = max(0.8, 0.035 * total_width_min)
    for start, end, stage in spans:
        if end - start >= minimum_width:
            axis.text(
                0.5 * (start + end),
                0.5,
                stage,
                color="white",
                fontsize=6.8,
                ha="center",
                va="center",
                fontweight="bold",
                clip_on=True,
            )


def wrap_report(report: str) -> tuple[str, float]:
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


def make_redraw_trajectory(
    source: pd.DataFrame, record: pd.Series, prediction: pd.Series
) -> pd.DataFrame:
    frame = source.loc[source["record_uid"].eq(record["record_uid"])].copy()
    frame = frame.sort_values(["epoch_index", "start_s"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No locked trajectory for {record['record_uid']}")
    keep = [
        "record_uid",
        "dataset",
        "subject_id",
        "split",
        "epoch_index",
        "start_s",
        "end_s",
        "duration_s",
        "predicted_stage",
        "p_W",
        "p_N1",
        "p_N2",
        "p_N3",
        "p_REM",
        "f_repeat_micro_v3",
        "bar_f_lambda",
        "stage_dream_probability",
        "dream_weight",
        "apparent_increment_s",
        "weighted_dream_increment_s",
        "cumulative_apparent_s",
        "cumulative_weighted_dream_s",
        "g_config_key",
        "stage_weights_json",
    ]
    missing = sorted(set(keep) - set(frame.columns))
    if missing:
        raise ValueError(f"Locked trajectory is missing {missing}")
    frame = frame[keep].copy()
    frame["annotation_order"] = int(record["annotation_order"])
    frame["eeg_filename"] = str(record["eeg_filename"])
    frame["dream_report_en"] = str(record["dream_report_en"])
    frame["dream_duration_estimation_s"] = float(record["consensus_mean_s"])
    frame["dream_duration_sigma_population_s"] = float(
        record["consensus_sigma_population_s"]
    )
    frame["phenomenological_time_s"] = float(prediction["apparent_time_s"])
    frame["stage_gated_dream_time_s"] = float(prediction["predicted_dream_s"])
    frame["sleep_stage_model"] = "GSSC v0.0.9 official five-class argmax"
    frame["sleep_stage_model_reference"] = "Hanna and Floel (2023)"
    frame["sleep_stage_model_doi"] = DATASET_METADATA["gssc_article_doi"]
    frame["kumral_data_doi"] = DATASET_METADATA["data_doi"]
    frame["kumral_article_doi"] = DATASET_METADATA["kumral_article_doi"]
    analysis_id = int(record["annotation_order"])
    frame["dream_set"] = DATASET_METADATA["dream_set"]
    frame["dream_set_row_within_66"] = DATASET_METADATA[
        "dream_set_row_within_66"
    ][analysis_id]
    frame["global_data_record_key_id"] = DATASET_METADATA[
        "global_data_record_key_id"
    ][analysis_id]
    return frame


def plot_record(
    frame: pd.DataFrame,
    eeg: pd.DataFrame,
    png_path: Path,
    pdf_path: Path | None,
    dpi: int,
) -> None:
    midpoint_min = (frame["start_s"] + frame["end_s"]) / 120.0
    end_min = frame["end_s"] / 60.0
    stage_spans = run_length_spans(frame, "predicted_stage")
    x_max = max(float(frame["end_s"].max()), float(eeg["end_s"].max())) / 60.0
    total_width = max(x_max, 0.5)
    report, report_font = wrap_report(str(frame["dream_report_en"].iloc[0]))

    fig = plt.figure(figsize=(18.0, 10.4))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.82, 1.0], wspace=0.10)
    left = GridSpecFromSubplotSpec(
        4,
        1,
        subplot_spec=outer[0],
        height_ratios=[4.35, 0.48, 0.48, 2.05],
        hspace=0.08,
    )
    top_ax = fig.add_subplot(left[0])
    stage_ax = fig.add_subplot(left[1], sharex=top_ax)
    eeg_ax = fig.add_subplot(left[2], sharex=top_ax)
    cumulative_ax = fig.add_subplot(left[3], sharex=top_ax)
    report_ax = fig.add_subplot(outer[1])

    prefix = float(frame["start_s"].min()) / 60.0
    if prefix > 0:
        for axis in (top_ax, stage_ax, eeg_ax, cumulative_ax):
            axis.axvspan(
                0.0,
                prefix,
                facecolor="#dddddd",
                hatch="////",
                alpha=0.55,
                lw=0,
            )
    for start, end, stage in stage_spans:
        top_ax.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.055, lw=0)
        stage_ax.axvspan(start, end, color=STAGE_COLORS[stage], alpha=0.96, lw=0)

    top_ax.plot(
        midpoint_min,
        frame["bar_f_lambda"],
        color="#15384c",
        lw=2.05,
    )
    top_ax.set_ylim(-0.03, 1.04)
    top_ax.set_ylabel(r"$\bar f(t)$")
    top_ax.grid(True, alpha=0.14)
    top_ax.tick_params(axis="x", labelbottom=False)

    stage_ax.set_ylim(0.0, 1.0)
    stage_ax.set_yticks([0.5], ["Sleep stage"], fontsize=7.5)
    stage_ax.tick_params(axis="x", labelbottom=False)
    label_long_stage_spans(stage_ax, stage_spans, total_width)

    eeg_time_min = eeg["midpoint_s"].to_numpy(float) / 60.0
    raw_q01 = eeg["raw_q01"].to_numpy(float)
    raw_q05 = eeg["raw_q05"].to_numpy(float)
    raw_median = eeg["raw_median"].to_numpy(float)
    raw_q95 = eeg["raw_q95"].to_numpy(float)
    raw_q99 = eeg["raw_q99"].to_numpy(float)
    eeg_ax.fill_between(
        eeg_time_min,
        raw_q01,
        raw_q99,
        color="#8aa6b5",
        alpha=0.22,
        lw=0,
    )
    eeg_ax.fill_between(
        eeg_time_min,
        raw_q05,
        raw_q95,
        color="#476f85",
        alpha=0.66,
        lw=0,
    )
    eeg_ax.plot(eeg_time_min, raw_median, color="#173f54", lw=0.34, alpha=0.85)
    lower = float(np.nanquantile(raw_q01, 0.005))
    upper = float(np.nanquantile(raw_q99, 0.995))
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        padding = 0.04 * (upper - lower)
        eeg_ax.set_ylim(lower - padding, upper + padding)
    unit = str(eeg["physical_unit"].iloc[0]).replace("uV", r"$\mu$V")
    eeg_ax.set_ylabel(f"Raw EEG\nC3–TP10 ({unit})", fontsize=7.2)
    eeg_ax.tick_params(axis="x", labelbottom=False)
    eeg_ax.tick_params(axis="y", labelsize=6.5)

    for axis in (stage_ax, eeg_ax):
        axis.spines[["top", "right", "bottom"]].set_visible(False)

    cumulative_ax.plot(
        end_min,
        frame["cumulative_apparent_s"] / 60.0,
        color="#963d72",
        lw=2.25,
        label="Phenomenological time",
    )
    cumulative_ax.plot(
        end_min,
        frame["cumulative_weighted_dream_s"] / 60.0,
        color="#d55262",
        lw=2.20,
        label="Stage-gated dream time",
    )
    cumulative_ax.axhline(
        float(frame["dream_duration_estimation_s"].iloc[0]) / 60.0,
        color="#d17b2f",
        ls="--",
        lw=1.35,
        label="Dream-duration estimation",
    )
    cumulative_ax.set_xlabel("Physical time from EDF start (min)")
    cumulative_ax.set_ylabel("Cumulative time (min)")
    cumulative_ax.grid(True, alpha=0.16)
    cumulative_ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    cumulative_ax.set_xlim(left=0.0, right=x_max)

    report_ax.axis("off")
    report_ax.set_title("Dream Report", fontsize=12.5, loc="left", pad=10)
    report_ax.text(
        0.0,
        0.96,
        report,
        ha="left",
        va="top",
        fontsize=report_font,
        linespacing=1.16,
        wrap=False,
        transform=report_ax.transAxes,
    )
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_ids(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return TEST_IDS
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    invalid = sorted(set(values) - set(TEST_IDS))
    if invalid:
        raise ValueError(f"IDs are not in the filtered test set: {invalid}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        default="all",
        help="Comma-separated analysis IDs, or 'all' (default)",
    )
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--consensus", type=Path, default=CONSENSUS_PATH)
    parser.add_argument("--kumral-zip", type=Path, default=KUMRAL_ZIP)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--refresh-eeg",
        action="store_true",
        help="Recompute compact raw-EEG CSVs even if they already exist",
    )
    parser.add_argument(
        "--redraw-only",
        action="store_true",
        help=(
            "Redraw exclusively from the packaged trajectory/EEG CSVs; this "
            "mode needs neither the original EDF ZIP nor the upstream analysis"
        ),
    )
    parser.add_argument(
        "--no-pdf", action="store_true", help="Write PNG only (PDF is default)"
    )
    args = parser.parse_args(argv)

    selected_ids = parse_ids(args.ids)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    if args.redraw_only:
        manifest_path = CANDIDATE_ROOT / "data/test_sample_figure_manifest.csv"
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        manifest = manifest.loc[manifest["annotation_order"].isin(selected_ids)].copy()
        if sorted(manifest["annotation_order"].astype(int)) != sorted(selected_ids):
            raise ValueError("Packaged manifest does not contain exactly the requested IDs")
        for row in manifest.sort_values("annotation_order").itertuples(index=False):
            trajectory = pd.read_csv(args.data_dir / row.trajectory_csv, low_memory=False)
            eeg = pd.read_csv(args.data_dir / row.eeg_summary_csv, low_memory=False)
            png_path = args.figure_dir / row.png
            pdf_path = None if args.no_pdf else args.figure_dir / row.pdf
            plot_record(trajectory, eeg, png_path, pdf_path, args.dpi)
            print(f"redrew ID {int(row.annotation_order)}: {png_path.name}", flush=True)
        return 0

    epoch_source = pd.read_csv(
        args.analysis_root / "all_epoch_trajectories.csv", low_memory=False
    )
    final_predictions = pd.read_csv(
        args.analysis_root / "all_final_model_record_predictions.csv"
    )
    consensus = pd.read_csv(args.consensus, low_memory=False)
    consensus = consensus.loc[consensus["annotation_order"].isin(selected_ids)].copy()
    if sorted(consensus["annotation_order"].astype(int)) != sorted(selected_ids):
        raise ValueError("Consensus table does not contain exactly the requested IDs")
    prediction_by_id = {
        int(row.annotation_order): pd.Series(row._asdict())
        for row in final_predictions.itertuples(index=False)
    }

    manifest_rows: list[dict[str, object]] = []
    for _, record in consensus.sort_values("annotation_order").iterrows():
        analysis_id = int(record["annotation_order"])
        filename = str(record["eeg_filename"])
        stem = f"ID_{analysis_id:03d}_{Path(filename).stem}"
        trajectory_path = args.data_dir / f"{stem}_trajectory.csv"
        eeg_path = args.data_dir / f"{stem}_eeg.csv"
        png_path = args.figure_dir / f"{stem}.png"
        pdf_path = None if args.no_pdf else args.figure_dir / f"{stem}.pdf"

        if analysis_id not in prediction_by_id:
            raise ValueError(f"No final locked prediction for analysis ID {analysis_id}")
        trajectory = make_redraw_trajectory(
            epoch_source, record, prediction_by_id[analysis_id]
        )
        trajectory.to_csv(trajectory_path, index=False)

        if args.refresh_eeg or not eeg_path.exists():
            print(f"streaming raw EEG for ID {analysis_id}: {filename}", flush=True)
            eeg = read_bipolar_eeg_summary_from_zip(args.kumral_zip, filename)
            eeg["annotation_order"] = analysis_id
            eeg["record_uid"] = str(record["record_uid"])
            eeg["eeg_filename"] = filename
            eeg["summary_note"] = (
                "unfiltered C3-TP10 bipolar raw EEG summarized within each EDF data record"
            )
            eeg.to_csv(eeg_path, index=False)
        else:
            eeg = pd.read_csv(eeg_path, low_memory=False)

        plot_record(trajectory, eeg, png_path, pdf_path, args.dpi)
        manifest_rows.append(
            {
                "annotation_order": analysis_id,
                "record_uid": str(record["record_uid"]),
                "eeg_filename": filename,
                "dream_set": DATASET_METADATA["dream_set"],
                "dream_set_row_within_66": DATASET_METADATA[
                    "dream_set_row_within_66"
                ][analysis_id],
                "global_data_record_key_id": DATASET_METADATA[
                    "global_data_record_key_id"
                ][analysis_id],
                "trajectory_csv": trajectory_path.name,
                "eeg_summary_csv": eeg_path.name,
                "png": png_path.name,
                "pdf": "" if pdf_path is None else pdf_path.name,
                "trajectory_rows": len(trajectory),
                "eeg_summary_rows": len(eeg),
                "trajectory_sha256": sha256(trajectory_path),
                "eeg_summary_sha256": sha256(eeg_path),
                "png_sha256": sha256(png_path),
                "pdf_sha256": "" if pdf_path is None else sha256(pdf_path),
            }
        )
        print(f"wrote ID {analysis_id}: {png_path.name}", flush=True)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(CANDIDATE_ROOT / "data/test_sample_figure_manifest.csv", index=False)
    metadata_path = CANDIDATE_ROOT / "data/test_sample_figure_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema": "tpmt.paper_candidate.test_sample_figures.v1",
                "test_ids": list(selected_ids),
                "stage_model": "GSSC v0.0.9 official five-class argmax",
                "stage_model_reference": "Hanna and Floel (2023)",
                "stage_model_doi": DATASET_METADATA["gssc_article_doi"],
                "raw_eeg_display": (
                    "unfiltered C3-TP10 bipolar EEG; one EDF-record bins; "
                    "1st-99th and 5th-95th percentile envelopes plus median"
                ),
                "kumral_data_doi": DATASET_METADATA["data_doi"],
                "kumral_article_doi": DATASET_METADATA["kumral_article_doi"],
                "dream_set": DATASET_METADATA["dream_set"],
                "dream_set_row_within_66_by_analysis_id": DATASET_METADATA[
                    "dream_set_row_within_66"
                ],
                "global_data_record_key_id_by_analysis_id": DATASET_METADATA[
                    "global_data_record_key_id"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote manifest for {len(manifest)} figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
