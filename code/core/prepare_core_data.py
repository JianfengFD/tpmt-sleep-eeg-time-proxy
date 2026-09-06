#!/usr/bin/env python3
"""Build the compact GSSC input tables from the original upstream results.

This is a packaging/provenance utility. It does not refit any model and does
not modify upstream files. The destination is the package's data/core tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(
        root / "outputs/self_similarity_fbar_v1/epoch_features_physical_grid.csv",
        low_memory=False,
    )
    final = labels["is_final_labelled_epoch"].astype(str).str.casefold().isin({"true", "1"})
    labels = labels.loc[
        labels["dataset"].eq("Kumral") & final,
        [
            "record_uid", "dataset", "filename", "subject_group", "split",
            "manual_final_stage", "valid_signal_end_s", "unmodelled_prefix_s",
            "is_final_labelled_epoch",
        ],
    ].sort_values("record_uid")
    if len(labels) != 66 or labels["record_uid"].duplicated().any():
        raise ValueError("Expected 66 unique Kumral endpoint labels")
    labels.to_csv(out / "kumral_endpoint_labels_66.csv", index=False)

    benchmark = pd.read_csv(
        root / "outputs/sleep_staging_benchmark_v1/endpoint_predictions.csv",
        low_memory=False,
    )
    benchmark = benchmark.loc[
        benchmark["dataset"].eq("Kumral")
        & benchmark["model"].eq("gssc_eeg_central_consensus")
        & benchmark["status"].eq("ok")
    ].sort_values("record_uid")
    if len(benchmark) != 66 or benchmark["record_uid"].duplicated().any():
        raise ValueError("Expected 66 unique GSSC benchmark endpoints")
    benchmark.to_csv(out / "gssc_endpoint_benchmark_66.csv", index=False)

    old = pd.read_csv(
        root / "outputs/self_similarity_fbar_v1/all_epoch_trajectories.csv",
        usecols=[
            "record_uid", "dataset", "epoch_index", "start_s", "end_s",
            "duration_s", "dominant_band",
        ],
        low_memory=False,
    )
    old = old.loc[old["dataset"].eq("Kumral")].sort_values(
        ["record_uid", "epoch_index"]
    )
    if old["record_uid"].nunique() != 66:
        raise ValueError("Expected dominant-band data for 66 Kumral records")
    old.to_csv(out / "kumral_dominant_band_lookup.csv.gz", index=False, compression="gzip")
    print(f"Wrote {len(labels)} endpoints and {len(old)} epoch lookup rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
