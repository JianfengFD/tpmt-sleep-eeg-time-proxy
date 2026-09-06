#!/usr/bin/env python3
"""Verify the lightweight core data package and its scientific invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    data = root / "data/core"

    physical = pd.read_csv(data / "physical_f/all_epoch_trajectories.csv")
    stages = pd.read_csv(data / "gssc/pipeline_stage_trajectories.csv.gz")
    consensus = pd.read_csv(data / "dream_duration/kumral_dream_duration_6llm_consensus.csv")
    final = pd.read_csv(data / "global_g/all_final_model_record_predictions.csv")
    endpoint = pd.read_csv(data / "staging_inputs/kumral_endpoint_labels_66.csv")

    assert len(endpoint) == 66 and endpoint["record_uid"].nunique() == 66
    assert set(endpoint["split"]) == {"train", "test"}
    assert (endpoint["split"] == "train").sum() == 44
    assert (endpoint["split"] == "test").sum() == 22
    assert stages["record_uid"].nunique() == 66
    assert final["record_uid"].nunique() == 33
    assert consensus["record_uid"].nunique() >= 37
    expected_train = {6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25}
    expected_test = {26, 27, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42}
    actual_train = set(final.loc[final["split"].eq("train"), "annotation_order"].astype(int))
    actual_test = set(final.loc[final["split"].eq("test"), "annotation_order"].astype(int))
    assert actual_train == expected_train
    assert actual_test == expected_test

    f_column = "f_repeat_micro_v3"
    assert physical[f_column].between(0.0, 1.0).all()
    assert stages["predicted_stage"].isin(["W", "N1", "N2", "N3", "REM"]).all()
    probability_columns = [f"p_{stage}" for stage in ["W", "N1", "N2", "N3", "REM"]]
    if all(column in stages for column in probability_columns):
        total = stages[probability_columns].sum(axis=1).to_numpy(float)
        assert np.allclose(total, 1.0, atol=1e-5)

    fit = json.loads((data / "global_g/fit_results.json").read_text())
    portable = json.loads(
        (data / "global_g/portable_stage_weighted_global_g.json").read_text()
    )
    assert fit["protocol"]["schema"] == "tpmt.stage_weighted_global_g.v1"
    assert fit["selected_configuration"]["config_key"] == "bernstein_degree_5__lambda_0"
    expected_weights = {"W": 0.0, "N1": 0.849, "N2": 0.532, "N3": 0.508, "REM": 0.834}
    assert portable["stage_weights"] == expected_weights
    assert portable["training_ids"] == sorted(expected_train)
    assert portable["heldout_ids"] == sorted(expected_test)
    validation = portable["validation"]
    assert validation["endpoint_fixed"] and validation["bounded"] and validation["monotone"]
    assert abs(float(validation["g_at_0"])) < 1e-12
    assert abs(float(validation["g_at_1"]) - 1.0) < 1e-12

    trajectory = pd.read_csv(data / "global_g/all_epoch_trajectories.csv")
    assert len(trajectory) == 4625 and trajectory["record_uid"].nunique() == 33
    assert trajectory["bar_f_lambda"].between(0.0, 1.0).all()
    mapped = trajectory["predicted_stage"].map(expected_weights).to_numpy(float)
    assert np.allclose(mapped, trajectory["dream_weight"].to_numpy(float), atol=1e-12)
    recomputed = trajectory.assign(
        apparent_check=trajectory["bar_f_lambda"] * trajectory["duration_s"],
        dream_check=trajectory["bar_f_lambda"] * trajectory["duration_s"] * trajectory["dream_weight"],
    ).groupby("record_uid", as_index=False).agg(
        apparent_check=("apparent_check", "sum"), dream_check=("dream_check", "sum")
    )
    comparison = final.merge(recomputed, on="record_uid", validate="one_to_one")
    assert np.allclose(comparison["apparent_time_s"], comparison["apparent_check"], atol=1e-7)
    assert np.allclose(comparison["predicted_dream_s"], comparison["dream_check"], atol=1e-7)

    if args.verify_manifest:
        manifest = pd.read_csv(root / "FILE_MANIFEST.csv")
        for row in manifest.itertuples(index=False):
            path = root / row.relative_path
            assert path.is_file(), path
            assert path.stat().st_size == row.bytes, path
            assert sha256(path) == row.sha256, path

    print(
        "OK: 66 Kumral endpoints (44 train/22 test), "
        f"{stages['record_uid'].nunique()} staged records, "
        f"{final['record_uid'].nunique()} stage-weighted dream-report records; "
        "f, g, stage weights, and record integrals valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
