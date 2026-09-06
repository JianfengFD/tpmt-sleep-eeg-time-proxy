#!/usr/bin/env python3
"""Extract GSSC probabilities and lock a Kumral REM/non-REM gate.

Only the manually scored final 30-second epoch in each Kumral EDF is used for
evaluation.  Model inference itself runs on the complete, end-aligned record.
Candidate score aggregation, temporal smoothing, and the REM threshold are
selected using the pre-existing *training subjects only*.  Dream duration,
filename, subject identity, and recording duration are never predictors.

The official GSSC ``loudest_vote`` chooses, at every epoch, the channel
permutation whose largest log-probability is largest.  This script preserves
that rule and exports the selected normalized probability vector rather than
fabricating probabilities from the hard label.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/Kumral_2023/Kumral et al., 2023.zip"
LABELS = ROOT / "outputs/self_similarity_fbar_v1/epoch_features_physical_grid.csv"
OUTPUT = ROOT / "outputs/sleep_staging_kumral_rem_probability_v1"
BENCHMARK = ROOT / "outputs/sleep_staging_benchmark_v1/endpoint_predictions.csv"
PHYSICAL_TRAJECTORIES = ROOT / "outputs/dense_repeat_fawake_v3/rstar_micro_fit/all_epoch_trajectories.csv"
OLD_STAGE_TRAJECTORIES = ROOT / "outputs/self_similarity_fbar_v1/all_epoch_trajectories.csv"
DREAM_CONSENSUS = ROOT / "outputs/combined_fawake_v2/kappa_fit_v1/kumral_dream_duration_6llm_consensus.csv"
GSSC_PATH = ROOT / ".vendor/gssc"
STAGES = ("W", "N1", "N2", "N3", "REM")
EPOCH_SECONDS = 30.0
SIGNAL_LENGTH = 2560
RANDOM_SEED = 20260905


def endpoint_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    final = frame["is_final_labelled_epoch"].astype(str).str.casefold().isin({"true", "1"})
    columns = [
        "record_uid", "filename", "subject_group", "split",
        "manual_final_stage", "valid_signal_end_s", "unmodelled_prefix_s",
    ]
    out = frame.loc[final & frame["dataset"].eq("Kumral"), columns].copy()
    out = out.rename(columns={"manual_final_stage": "manual_stage"})
    out["manual_stage"] = out["manual_stage"].replace({"R": "REM", "S1": "N1", "S2": "N2", "S3": "N3"})
    if len(out) != 66 or out["record_uid"].duplicated().any():
        raise ValueError(f"Expected 66 unique Kumral endpoints, got {len(out)}")
    return out.sort_values("record_uid").reset_index(drop=True)


def archive_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        members = [item.filename for item in archive.infolist() if item.filename.lower().endswith(".edf")]
    return {Path(member).name: member for member in members}


def extract_member(archive_path: Path, member: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, target.open("wb") as sink:
        shutil.copyfileobj(source, sink, length=16 * 1024 * 1024)


def exact_bounds(row: pd.Series, raw: mne.io.BaseRaw) -> tuple[int, int, int]:
    sfreq = float(raw.info["sfreq"])
    end_s = min(float(row["valid_signal_end_s"]), float(raw.times[-1]) + 1.0 / sfreq)
    prefix_s = max(0.0, float(row["unmodelled_prefix_s"]))
    n_epochs = int(np.floor((end_s - prefix_s) / EPOCH_SECONDS + 1e-9))
    stop = int(round(end_s * sfreq))
    start = stop - int(round(n_epochs * EPOCH_SECONDS * sfreq))
    if n_epochs < 1 or start < 0:
        raise ValueError(f"Bad sequence bounds for {row['record_uid']}")
    return start, stop, n_epochs


def load_gssc() -> Any:
    sys.path.insert(0, str(GSSC_PATH))
    original_load = torch.load

    def trusted_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        kwargs.setdefault("map_location", "cpu")
        return original_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        from gssc.infer import EEGInfer

        model = EEGInfer(use_cuda=False, cut="back")
    finally:
        torch.load = original_load
    return model


def gssc_probabilities(model: Any, raw: mne.io.BaseRaw, row: pd.Series) -> pd.DataFrame:
    """Return genuine GSSC probabilities for both EEG permutations and consensus."""
    from gssc.utils import epo_arr_zscore, permute_sigs, prepare_inst

    needed = ["C3", "TP10", "C4", "TP9"]
    missing = [channel for channel in needed if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    start, stop, n_epochs = exact_bounds(row, raw)
    sfreq = float(raw.info["sfreq"])
    picked = raw.get_data(picks=needed, start=start, stop=stop)
    values = np.vstack((picked[0] - picked[1], picked[2] - picked[3]))
    expected = int(round(n_epochs * EPOCH_SECONDS * sfreq))
    values = np.pad(values[:, :expected], ((0, 0), (0, 1)), mode="edge")
    info = mne.create_info(["C3-M2", "C4-M1"], sfreq=sfreq, ch_types=["eeg", "eeg"])
    sequence = mne.io.RawArray(values, info, verbose="ERROR")
    sequence.filter(0.3, 30.0, verbose="ERROR")

    epochs, _ = prepare_inst(sequence, SIGNAL_LENGTH, "back")
    signals = {
        "eeg": {"chans": ["C3-M2", "C4-M1"], "drop": False, "flip": False},
        "eog": {"chans": [], "drop": True, "flip": False},
    }
    sig_combs, permutation_matrix, all_channels, _ = permute_sigs(epochs, signals)
    data = epo_arr_zscore(epochs.get_data(picks=all_channels) * 1e6)

    log_probabilities: list[np.ndarray] = []
    permutation_names: list[str] = []
    with torch.inference_mode():
        for permutation in permutation_matrix:
            sigs: dict[str, torch.Tensor] = {}
            chosen_eeg = ""
            for signal_index, (signal_name, channel_options) in enumerate(sig_combs.items()):
                selected = channel_options[permutation[signal_index]]
                if not selected:
                    continue
                channel = selected[0]
                channel_index = all_channels.index(channel)
                tensor = torch.tensor(
                    data[:, channel_index, :SIGNAL_LENGTH].reshape(-1, 1, SIGNAL_LENGTH),
                    dtype=torch.float32,
                )
                sigs[signal_name] = tensor
                if signal_name == "eeg":
                    chosen_eeg = channel
            representations = model.net(sigs, rep_output="rep_only").swapaxes(-1, 1)
            hidden = torch.zeros(10, 1, 256)
            output, _ = model.con_net(representations, hidden)
            log_probabilities.append(output[:, 0, :].cpu().numpy())
            permutation_names.append(chosen_eeg)

    logp = np.stack(log_probabilities, axis=0)
    if logp.shape != (2, n_epochs, 5):
        raise RuntimeError(f"Unexpected GSSC output shape: {logp.shape}")
    probabilities = np.exp(logp)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    # This is exactly GSSC's official loudest-vote criterion: NLL evaluated
    # against each permutation's own argmax equals -max(log probability).
    selected_permutation = np.argmax(logp.max(axis=-1), axis=0)
    consensus = probabilities[selected_permutation, np.arange(n_epochs)]
    arithmetic = probabilities.mean(axis=0)
    geometric = np.exp(logp.mean(axis=0))
    geometric /= geometric.sum(axis=-1, keepdims=True)

    result = pd.DataFrame({
        "epoch_index": np.arange(n_epochs, dtype=int),
        "start_s": np.arange(n_epochs, dtype=float) * EPOCH_SECONDS,
        "end_s": (np.arange(n_epochs, dtype=float) + 1.0) * EPOCH_SECONDS,
        "selected_permutation": np.asarray(permutation_names)[selected_permutation],
    })
    for permutation_index, permutation_name in enumerate(permutation_names):
        short = "c3_m2" if permutation_name == "C3-M2" else "c4_m1"
        for stage_index, stage in enumerate(STAGES):
            result[f"p_{stage}_{short}"] = probabilities[permutation_index, :, stage_index]
    for prefix, values_ in (("consensus", consensus), ("mean", arithmetic), ("geomean", geometric)):
        for stage_index, stage in enumerate(STAGES):
            result[f"p_{stage}_{prefix}"] = values_[:, stage_index]
        result[f"stage_{prefix}"] = np.asarray(STAGES)[np.argmax(values_, axis=1)]
    return result


def candidate_scores(epochs: pd.DataFrame) -> dict[str, np.ndarray]:
    bases = {
        "consensus": epochs["p_REM_consensus"].to_numpy(float),
        "mean": epochs["p_REM_mean"].to_numpy(float),
        "geomean": epochs["p_REM_geomean"].to_numpy(float),
        "c3_m2": epochs["p_REM_c3_m2"].to_numpy(float),
        "c4_m1": epochs["p_REM_c4_m1"].to_numpy(float),
    }
    output: dict[str, np.ndarray] = {}
    for base_name, values in bases.items():
        series = pd.Series(values)
        for window in (1, 2, 3, 5):
            # A causal trailing mean is usable at every epoch and does not
            # synthesize unavailable future samples at the labelled endpoint.
            output[f"{base_name}__trailing_mean_{window}"] = (
                series.rolling(window, min_periods=1).mean().to_numpy(float)
            )
    return output


def threshold_grid(scores: np.ndarray) -> np.ndarray:
    fixed = np.linspace(0.01, 0.99, 99)
    unique = np.unique(scores)
    midpoints = (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else unique
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], fixed, unique, midpoints)))


def choose_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = threshold_grid(scores)
    predictions = scores[None, :] >= thresholds[:, None]
    truth = y.astype(bool)[None, :]
    tp = np.sum(predictions & truth, axis=1)
    fp = np.sum(predictions & ~truth, axis=1)
    fn = np.sum(~predictions & truth, axis=1)
    tn = np.sum(~predictions & ~truth, axis=1)
    sensitivity = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
    bacc = (sensitivity + specificity) / 2
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    accuracy = (tp + tn) / y.size
    # np.lexsort uses the last key as primary; all keys are maximized.
    order = np.lexsort((thresholds, -np.abs(thresholds - 0.5), accuracy, f1, bacc))
    index = int(order[-1])
    return float(thresholds[index]), {
        "balanced_accuracy": float(bacc[index]), "f1": float(f1[index]),
        "accuracy": float(accuracy[index]), "sensitivity": float(sensitivity[index]),
        "specificity": float(specificity[index]),
    }


def train_only_selection(endpoints: pd.DataFrame) -> tuple[str, float, pd.DataFrame]:
    train = endpoints.loc[endpoints["split"].eq("train")].copy()
    y = train["is_rem"].to_numpy(bool)
    groups = train["subject_group"].to_numpy(str)
    candidates = sorted(column.removeprefix("score__") for column in train if column.startswith("score__"))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = train[f"score__{candidate}"].to_numpy(float)
        oof_pred = np.zeros(len(train), dtype=bool)
        fold_thresholds: list[float] = []
        for held_group in np.unique(groups):
            fit = groups != held_group
            held = ~fit
            threshold, _ = choose_threshold(y[fit], scores[fit])
            fold_thresholds.append(threshold)
            oof_pred[held] = scores[held] >= threshold
        tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[False, True]).ravel()
        oof_bacc = balanced_accuracy_score(y, oof_pred)
        oof_f1 = f1_score(y, oof_pred, zero_division=0)
        oof_acc = accuracy_score(y, oof_pred)
        rows.append({
            "candidate": candidate,
            "loso_balanced_accuracy": float(oof_bacc),
            "loso_f1": float(oof_f1),
            "loso_accuracy": float(oof_acc),
            "loso_sensitivity": float(tp / (tp + fn)),
            "loso_specificity": float(tn / (tn + fp)),
            "loso_average_precision_score": float(average_precision_score(y, scores)),
            "loso_roc_auc_score": float(roc_auc_score(y, scores)),
            "median_fold_threshold": float(np.median(fold_thresholds)),
        })
    results = pd.DataFrame(rows).sort_values(
        ["loso_balanced_accuracy", "loso_f1", "loso_accuracy", "loso_average_precision_score", "candidate"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    chosen = str(results.iloc[0]["candidate"])
    # Refit only the scalar threshold on every training endpoint after the
    # candidate is locked by subject-LOSO predictions.
    threshold, _ = choose_threshold(y, train[f"score__{chosen}"].to_numpy(float))
    return chosen, threshold, results


def binary_bacc(y: np.ndarray, pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    return float(((tp / (tp + fn)) + (tn / (tn + fp))) / 2)


def subject_bootstrap_stability(
    endpoints: pd.DataFrame,
    originally_chosen: str,
    repetitions: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit the entire 20-candidate search without touching test subjects.

    Each replicate samples the 13 training subjects with replacement, selects
    a candidate and threshold on the in-bag records, and compares it with the
    pre-registered official argmax gate on unsampled subjects.  OOB estimates
    are retained only when both REM and non-REM are represented.
    """
    train = endpoints.loc[endpoints["split"].eq("train")].reset_index(drop=True)
    groups = np.asarray(sorted(train["subject_group"].unique()))
    candidates = sorted(column.removeprefix("score__") for column in train if column.startswith("score__"))
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []
    for repeat in range(repetitions):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([
            train.index[train["subject_group"].eq(group)].to_numpy()
            for group in sampled_groups
        ])
        inbag = train.loc[sampled_indices].reset_index(drop=True)
        y_in = inbag["is_rem"].to_numpy(bool)
        if len(np.unique(y_in)) < 2:
            continue
        ranked: list[tuple[tuple[float, ...], str, float]] = []
        for candidate in candidates:
            scores_in = inbag[f"score__{candidate}"].to_numpy(float)
            threshold, values = choose_threshold(y_in, scores_in)
            key = (
                values["balanced_accuracy"], values["f1"], values["accuracy"],
                -abs(threshold - 0.5),
            )
            ranked.append((key, candidate, threshold))
        ranked.sort(reverse=True)
        _, selected, selected_threshold = ranked[0]
        original_threshold, _ = choose_threshold(
            y_in, inbag[f"score__{originally_chosen}"].to_numpy(float)
        )
        unsampled = sorted(set(groups) - set(sampled_groups))
        oob = train.loc[train["subject_group"].isin(unsampled)]
        row: dict[str, Any] = {
            "repeat": repeat,
            "selected_candidate": selected,
            "selected_threshold": float(selected_threshold),
            "original_candidate_threshold": float(original_threshold),
            "oob_subjects": int(len(unsampled)),
            "oob_records": int(len(oob)),
            "oob_delta_balanced_accuracy_selected_minus_official": np.nan,
            "oob_delta_balanced_accuracy_original_minus_official": np.nan,
        }
        if len(oob) and oob["is_rem"].nunique() == 2:
            y_oob = oob["is_rem"].to_numpy(bool)
            baseline = oob["stage_consensus"].eq("REM").to_numpy(bool)
            selected_pred = oob[f"score__{selected}"].to_numpy(float) >= selected_threshold
            original_pred = (
                oob[f"score__{originally_chosen}"].to_numpy(float) >= original_threshold
            )
            baseline_bacc = binary_bacc(y_oob, baseline)
            row["oob_delta_balanced_accuracy_selected_minus_official"] = (
                binary_bacc(y_oob, selected_pred) - baseline_bacc
            )
            row["oob_delta_balanced_accuracy_original_minus_official"] = (
                binary_bacc(y_oob, original_pred) - baseline_bacc
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    delta = frame["oob_delta_balanced_accuracy_selected_minus_official"].dropna()
    original_delta = frame["oob_delta_balanced_accuracy_original_minus_official"].dropna()
    thresholds = frame["original_candidate_threshold"].dropna()
    frequency = frame["selected_candidate"].value_counts(normalize=True)
    summary = {
        "repetitions_requested": repetitions,
        "repetitions_completed": int(len(frame)),
        "oob_repetitions_with_both_classes": int(len(delta)),
        "selected_candidate_frequency": {str(key): float(value) for key, value in frequency.items()},
        "original_candidate_threshold_quantiles_2p5_25_50_75_97p5": [
            float(value) for value in thresholds.quantile([0.025, 0.25, 0.5, 0.75, 0.975])
        ],
        "multi_selection_oob_delta_bacc_quantiles_2p5_50_97p5": [
            float(value) for value in delta.quantile([0.025, 0.5, 0.975])
        ] if len(delta) else [None, None, None],
        "multi_selection_probability_oob_delta_positive": float((delta > 0).mean()) if len(delta) else None,
        "original_candidate_oob_delta_bacc_quantiles_2p5_50_97p5": [
            float(value) for value in original_delta.quantile([0.025, 0.5, 0.975])
        ] if len(original_delta) else [None, None, None],
        "original_candidate_probability_oob_delta_positive": (
            float((original_delta > 0).mean()) if len(original_delta) else None
        ),
    }
    return frame, summary


def verify_against_existing_benchmark(endpoints: pd.DataFrame, path: Path) -> dict[str, Any]:
    benchmark = pd.read_csv(path, low_memory=False)
    benchmark = benchmark.loc[
        benchmark["dataset"].eq("Kumral")
        & benchmark["model"].eq("gssc_eeg_central_consensus")
        & benchmark["status"].eq("ok"),
        ["record_uid", "predicted_stage"],
    ]
    check = endpoints[["record_uid", "stage_consensus"]].merge(
        benchmark, on="record_uid", how="outer", validate="one_to_one", indicator=True
    )
    mismatch = check.loc[
        check["_merge"].ne("both") | check["stage_consensus"].ne(check["predicted_stage"])
    ]
    if len(mismatch):
        raise AssertionError(f"GSSC endpoint reproduction failed for {len(mismatch)} records")
    return {"records_checked": int(len(check)), "mismatches": 0, "exact_match": True}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def subject_cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    score: np.ndarray,
    repetitions: int = 2000,
) -> dict[str, Any]:
    """Percentile intervals that preserve repeated records within subjects."""
    truth = frame["is_rem"].to_numpy(bool)
    groups = frame["subject_group"].to_numpy(str)
    unique_groups = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(RANDOM_SEED + len(frame))
    values: dict[str, list[float]] = {
        key: [] for key in (
            "accuracy", "balanced_accuracy", "rem_sensitivity",
            "nonrem_specificity", "rem_precision", "rem_f1",
            "average_precision", "roc_auc",
        )
    }
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        take = np.concatenate([indices[group] for group in sampled])
        y = truth[take]
        p = prediction[take]
        s = score[take]
        values["accuracy"].append(float(np.mean(y == p)))
        if np.unique(y).size < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        values["balanced_accuracy"].append(float((sensitivity + specificity) / 2))
        values["rem_sensitivity"].append(float(sensitivity))
        values["nonrem_specificity"].append(float(specificity))
        values["rem_precision"].append(float(precision))
        values["rem_f1"].append(float(f1))
        values["average_precision"].append(float(average_precision_score(y, s)))
        values["roc_auc"].append(float(roc_auc_score(y, s)))
    return {
        key: {
            "percentile_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "valid_replicates": int(len(samples)),
        }
        for key, samples in values.items() if samples
    }


def metric_block(frame: pd.DataFrame, prediction: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    truth = frame["is_rem"].to_numpy(bool)
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[False, True]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=[False, True], zero_division=0
    )
    return {
        "n": int(len(frame)), "subjects": int(frame["subject_group"].nunique()),
        "rem_prevalence": float(truth.mean()),
        "confusion_truth_nonrem_rem_by_prediction_nonrem_rem": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": float(accuracy_score(truth, prediction)),
        "accuracy_wilson_95ci": wilson_interval(int((truth == prediction).sum()), len(truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "rem_sensitivity": None if sensitivity is None else float(sensitivity),
        "rem_sensitivity_wilson_95ci": wilson_interval(int(tp), int(tp + fn)),
        "nonrem_specificity": None if specificity is None else float(specificity),
        "nonrem_specificity_wilson_95ci": wilson_interval(int(tn), int(tn + fp)),
        "rem_precision": float(precision[1]), "rem_f1": float(f1[1]),
        "average_precision": float(average_precision_score(truth, score)),
        "roc_auc": float(roc_auc_score(truth, score)),
        "support_nonrem_rem": [int(support[0]), int(support[1])],
        "subject_cluster_bootstrap": subject_cluster_bootstrap_intervals(
            frame, prediction, score
        ),
    }


def record_level_qc(all_epochs: pd.DataFrame, endpoints: pd.DataFrame) -> pd.DataFrame:
    """Describe whole-record plausibility without treating it as validation.

    Thresholds below are pre-specified descriptive warnings.  They are not
    fitted to endpoint labels, held-out subjects, or dream durations.
    """
    rows: list[dict[str, Any]] = []
    probability_columns = [f"p_{stage}_consensus" for stage in STAGES]
    for record_uid, frame in all_epochs.groupby("record_uid", sort=True):
        frame = frame.sort_values("epoch_index")
        gate = frame["rem_gate"].to_numpy(bool)
        starts = np.flatnonzero(gate & np.r_[True, ~gate[:-1]])
        ends = np.flatnonzero(gate & np.r_[~gate[1:], True])
        lengths = ends - starts + 1
        probabilities = frame[probability_columns].to_numpy(float)
        entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
        normalized_entropy = entropy / np.log(len(STAGES))
        confidence = probabilities.max(axis=1)
        c3_stage = np.asarray(STAGES)[np.argmax(
            frame[[f"p_{stage}_c3_m2" for stage in STAGES]].to_numpy(float), axis=1
        )]
        c4_stage = np.asarray(STAGES)[np.argmax(
            frame[[f"p_{stage}_c4_m1" for stage in STAGES]].to_numpy(float), axis=1
        )]
        n_rem = int(gate.sum())
        isolated = int((lengths == 1).sum()) if len(lengths) else 0
        rem_fraction = float(gate.mean())
        isolated_epoch_fraction = float(isolated / n_rem) if n_rem else 0.0
        low_confidence_fraction = float((confidence < 0.50).mean())
        channel_disagreement_fraction = float((c3_stage != c4_stage).mean())
        flags = []
        if rem_fraction > 0.35:
            flags.append("high_REM_fraction_gt_0.35")
        if isolated_epoch_fraction > 0.25:
            flags.append("fragmented_REM_isolated_fraction_gt_0.25")
        if low_confidence_fraction > 0.25:
            flags.append("low_confidence_fraction_gt_0.25")
        if channel_disagreement_fraction > 0.50:
            flags.append("C3_C4_stage_disagreement_gt_0.50")
        rows.append({
            "record_uid": record_uid,
            "total_epochs": int(len(frame)),
            "total_duration_min": float(len(frame) * EPOCH_SECONDS / 60),
            "rem_epochs": n_rem,
            "rem_fraction": rem_fraction,
            "rem_bout_count": int(len(lengths)),
            "rem_bout_median_s": float(np.median(lengths) * EPOCH_SECONDS) if len(lengths) else 0.0,
            "rem_bout_longest_s": float(np.max(lengths) * EPOCH_SECONDS) if len(lengths) else 0.0,
            "isolated_single_epoch_rem_bouts": isolated,
            "isolated_single_epoch_rem_fraction_of_rem_epochs": isolated_epoch_fraction,
            "mean_normalized_entropy": float(normalized_entropy.mean()),
            "low_confidence_fraction_pmax_lt_0.50": low_confidence_fraction,
            "c3_c4_stage_disagreement_fraction": channel_disagreement_fraction,
            "diagnostic_flags": ";".join(flags),
            "diagnostic_flag_count": len(flags),
        })
    qc = pd.DataFrame(rows)
    return endpoints[["record_uid", "subject_group", "split", "manual_stage"]].merge(
        qc, on="record_uid", how="left", validate="one_to_one"
    )


def align_to_locked_physical_keys(all_epochs: pd.DataFrame, physical_path: Path) -> pd.DataFrame:
    keys = pd.read_csv(
        physical_path,
        usecols=["record_uid", "dataset", "epoch_index", "start_s", "end_s", "duration_s"],
    )
    keys = keys.loc[keys["dataset"].eq("Kumral")].copy()
    if keys.duplicated(["record_uid", "epoch_index"]).any():
        raise ValueError("Locked physical trajectories contain duplicate Kumral epoch indices")
    result = all_epochs.drop(columns=["start_s", "end_s"], errors="ignore").merge(
        keys, on=["record_uid", "epoch_index"], how="outer", validate="one_to_one", indicator=True
    )
    if not result["_merge"].eq("both").all():
        counts = result["_merge"].value_counts().to_dict()
        raise AssertionError(f"GSSC/physical epoch-key mismatch: {counts}")
    return result.drop(columns="_merge").sort_values(["record_uid", "epoch_index"]).reset_index(drop=True)


def write_pipeline_adapter(
    all_epochs: pd.DataFrame,
    output_path: Path,
    old_stage_path: Path,
    dream_consensus_path: Path,
) -> dict[str, Any]:
    join_keys = ["record_uid", "epoch_index", "start_s", "end_s", "duration_s"]
    old = pd.read_csv(old_stage_path, usecols=[*join_keys, "dataset", "dominant_band"])
    old = old.loc[old["dataset"].eq("Kumral")].drop(columns="dataset")
    if old.duplicated(join_keys).any():
        raise ValueError("Old stage trajectory contains duplicate locked keys")
    adapter = all_epochs.merge(old, on=join_keys, how="left", validate="one_to_one")
    if adapter["dominant_band"].isna().any():
        raise AssertionError("Could not carry dominant_band onto every GSSC epoch")
    adapter = adapter.rename(columns={
        "stage_rem_gated": "predicted_stage",
        **{f"p_{stage}_consensus": f"p_{stage}" for stage in STAGES},
    })
    columns = [
        *join_keys, "dataset", "predicted_stage", "dominant_band",
        *[f"p_{stage}" for stage in STAGES],
        "rem_score", "rem_gate", "official_argmax_rem_gate",
        "tuned_candidate_rem_score", "tuned_candidate_rem_gate",
    ]
    adapter[columns].to_csv(output_path, index=False, compression="gzip")

    consensus = pd.read_csv(dream_consensus_path, low_memory=False)
    used = consensus.loc[
        consensus["used_for_kappa_fit"].astype(str).str.casefold().isin({"true", "1"})
        | consensus["used_for_holdout_test"].astype(str).str.casefold().isin({"true", "1"})
    ]
    expected_records = set(used["record_uid"].astype(str))
    actual_records = set(adapter["record_uid"].astype(str))
    missing = sorted(expected_records - actual_records)
    if missing:
        raise AssertionError(f"Pipeline adapter lacks {len(missing)} g-fit records")
    physical_keys = set(map(tuple, adapter.loc[adapter["record_uid"].isin(expected_records), join_keys].to_numpy()))
    return {
        "all_kumral_records": int(adapter["record_uid"].nunique()),
        "all_kumral_epochs": int(len(adapter)),
        "g_fit_or_test_records_verified": int(len(expected_records)),
        "g_fit_or_test_epoch_keys_verified": int(len(physical_keys)),
        "missing_g_records": missing,
        "exact_locked_key_alignment": True,
    }


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    shards = args.output / "all_epoch_records"
    shards.mkdir(exist_ok=True)
    labels = endpoint_table(args.labels)
    members = archive_members(args.archive)
    model = load_gssc()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    with tempfile.TemporaryDirectory(prefix="kumral_gssc_", dir=args.temp_dir) as temp_name:
        temp_dir = Path(temp_name)
        for index, row in labels.iterrows():
            shard = shards / f"{row['record_uid'].replace(':', '__')}.csv.gz"
            if shard.exists() and not args.force:
                print(f"[{index + 1}/66] cached {row['record_uid']}", flush=True)
                continue
            member = members.get(str(row["filename"]))
            if member is None:
                raise FileNotFoundError(row["filename"])
            edf = temp_dir / str(row["filename"])
            extract_member(args.archive, member, edf)
            raw = None
            try:
                raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
                frame = gssc_probabilities(model, raw, row)
                scores = candidate_scores(frame)
                for candidate, values in scores.items():
                    frame[f"score__{candidate}"] = values
                frame.insert(0, "record_uid", row["record_uid"])
                frame.insert(1, "subject_group", row["subject_group"])
                frame.insert(2, "split", row["split"])
                frame.to_csv(shard, index=False, compression="gzip")
                print(f"[{index + 1}/66] inferred {row['record_uid']} ({len(frame)} epochs)", flush=True)
            finally:
                if raw is not None:
                    raw.close()
                edf.unlink(missing_ok=True)

    epoch_frames = [pd.read_csv(path, low_memory=False) for path in sorted(shards.glob("*.csv.gz"))]
    if len(epoch_frames) != 66:
        raise RuntimeError(f"Expected 66 probability shards, found {len(epoch_frames)}")
    all_epochs = pd.concat(epoch_frames, ignore_index=True)
    # Shards store time relative to the modelled sequence.  Replace those
    # columns with the locked physical trajectory keys, which include each
    # record's unmodelled prefix (e.g. 27, 57, ... seconds).
    all_epochs = align_to_locked_physical_keys(all_epochs, args.physical_trajectories)
    endpoint_scores = all_epochs.sort_values(["record_uid", "epoch_index"]).groupby("record_uid", as_index=False).tail(1)
    score_columns = [column for column in all_epochs if column.startswith("score__")]
    endpoints = labels.merge(
        endpoint_scores[["record_uid", "stage_consensus", *score_columns]],
        on="record_uid", how="left", validate="one_to_one",
    )
    endpoints["is_rem"] = endpoints["manual_stage"].eq("REM")

    reproduction = verify_against_existing_benchmark(endpoints, args.benchmark)
    chosen, threshold, selection = train_only_selection(endpoints)
    bootstrap, bootstrap_summary = subject_bootstrap_stability(endpoints, chosen)
    score_column = f"score__{chosen}"
    all_epochs["official_argmax_rem_gate"] = all_epochs["stage_consensus"].eq("REM")
    all_epochs["tuned_candidate_rem_score"] = all_epochs[score_column]
    all_epochs["tuned_candidate_rem_gate"] = all_epochs[score_column].ge(threshold)

    train_endpoints = endpoints.loc[endpoints["split"].eq("train")]
    train_truth = train_endpoints["is_rem"].to_numpy(bool)
    baseline_train_bacc = binary_bacc(
        train_truth, train_endpoints["stage_consensus"].eq("REM").to_numpy(bool)
    )
    selected_oof_bacc = float(selection.iloc[0]["loso_balanced_accuracy"])
    delta_quantiles = bootstrap_summary["multi_selection_oob_delta_bacc_quantiles_2p5_50_97p5"]
    threshold_quantiles = bootstrap_summary["original_candidate_threshold_quantiles_2p5_25_50_75_97p5"]
    # A data-selected gate is allowed to replace the pre-registered baseline
    # only with an OOF gain and a strictly positive 95% train-subject-bootstrap
    # lower bound.  The threshold distribution must also avoid saturation.
    stable_improvement = bool(
        selected_oof_bacc > baseline_train_bacc
        and delta_quantiles[0] is not None and delta_quantiles[0] > 0
        and threshold_quantiles[0] > 0.01 and threshold_quantiles[-1] < 0.99
    )
    production_rule = "tuned_candidate" if stable_improvement else "official_argmax"
    all_epochs["rem_score"] = (
        all_epochs[score_column] if stable_improvement else all_epochs["p_REM_consensus"]
    )
    all_epochs["rem_gate"] = (
        all_epochs["tuned_candidate_rem_gate"]
        if stable_improvement else all_epochs["official_argmax_rem_gate"]
    )
    # Preserve non-REM ordering from official consensus probabilities.
    nonrem_probs = all_epochs[[f"p_{stage}_consensus" for stage in STAGES[:-1]]].to_numpy(float)
    all_epochs["stage_rem_gated"] = np.asarray(STAGES[:-1])[np.argmax(nonrem_probs, axis=1)]
    all_epochs.loc[all_epochs["rem_gate"], "stage_rem_gated"] = "REM"
    all_epochs.to_csv(args.output / "all_epoch_probabilities.csv.gz", index=False, compression="gzip")
    adapter_validation = write_pipeline_adapter(
        all_epochs,
        args.output / "pipeline_stage_trajectories.csv.gz",
        args.old_stage_trajectories,
        args.dream_consensus,
    )

    endpoints["official_argmax_rem"] = endpoints["stage_consensus"].eq("REM")
    endpoints["tuned_candidate_rem_score"] = endpoints[score_column]
    endpoints["tuned_candidate_rem"] = endpoints[score_column].ge(threshold)
    endpoints["rem_score"] = (
        endpoints[score_column] if stable_improvement
        else endpoint_scores.set_index("record_uid").loc[endpoints["record_uid"], "p_REM_consensus"].to_numpy()
    )
    endpoints["predicted_rem"] = (
        endpoints["tuned_candidate_rem"] if stable_improvement else endpoints["official_argmax_rem"]
    )
    endpoints["predicted_binary_stage"] = np.where(endpoints["predicted_rem"], "REM", "non-REM")
    endpoints.to_csv(args.output / "endpoint_predictions.csv", index=False)
    record_level_qc(all_epochs, endpoints).to_csv(args.output / "record_level_qc.csv", index=False)
    selection.to_csv(args.output / "train_only_candidate_selection.csv", index=False)
    bootstrap.to_csv(args.output / "train_subject_bootstrap_stability.csv.gz", index=False, compression="gzip")

    official_partitions = {
        "train": endpoints.loc[endpoints["split"].eq("train")],
        "test": endpoints.loc[endpoints["split"].eq("test")],
        # Descriptive only: this intentionally pools model-selection and
        # held-out partitions and must not replace the primary test result.
        "all_66_endpoints_descriptive": endpoints,
    }
    endpoint_probability_lookup = endpoint_scores.set_index("record_uid")["p_REM_consensus"]
    metrics = {
        "chosen_score": chosen,
        "locked_threshold": float(threshold),
        "production_rule": production_rule,
        "tuned_gate_replaced_official_baseline": stable_improvement,
        "replacement_guard": {
            "selected_loso_bacc": selected_oof_bacc,
            "official_argmax_train_bacc": baseline_train_bacc,
            "bootstrap": bootstrap_summary,
        },
        "official_argmax_baseline": {
            partition: metric_block(
                part,
                part["official_argmax_rem"].to_numpy(bool),
                endpoint_probability_lookup.loc[part["record_uid"]].to_numpy(float),
            )
            for partition, part in official_partitions.items()
        },
        "experimental_tuned_candidate": {
            split: metric_block(
                part,
                part["tuned_candidate_rem"].to_numpy(bool),
                part["tuned_candidate_rem_score"].to_numpy(float),
            )
            for split, part in endpoints.groupby("split")
        },
        "selection_rule": "candidate by training-subject LOSO balanced accuracy/F1; threshold refit on all training endpoints",
        "train": metric_block(
            endpoints.loc[endpoints["split"].eq("train")],
            endpoints.loc[endpoints["split"].eq("train"), "predicted_rem"].to_numpy(bool),
            endpoints.loc[endpoints["split"].eq("train"), "rem_score"].to_numpy(float),
        ),
        "held_out_test": metric_block(
            endpoints.loc[endpoints["split"].eq("test")],
            endpoints.loc[endpoints["split"].eq("test"), "predicted_rem"].to_numpy(bool),
            endpoints.loc[endpoints["split"].eq("test"), "rem_score"].to_numpy(float),
        ),
        "official_gssc_five_class_endpoint_accuracy": {
            split: float((part["stage_consensus"] == part["manual_stage"]).mean())
            for split, part in endpoints.groupby("split")
        },
        "hard_label_reproduction": reproduction,
        "pipeline_adapter_validation": adapter_validation,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "model": "GSSC 0.0.9 official sig_net_v1.pt + gru_net_v1.pt",
        "input": "complete end-aligned sequence; C3-TP10 and C4-TP9; 0.3-30 Hz; 85.333 Hz; continuous per-channel z-score",
        "probability_semantics": "exp(log-softmax); official per-epoch loudest-vote permutation selected by largest maximum log-probability",
        "manual_ground_truth_scope": "final 30-second endpoint only (66 labels); full-record hypnograms are unavailable",
        "leakage_exclusions": ["dream duration", "dream report", "filename", "recording duration", "subject identity"],
        "stage_order": list(STAGES),
        "records": 66, "epochs": int(len(all_epochs)),
        "archive": str(args.archive), "labels": str(args.labels),
        "chosen_score": chosen, "locked_threshold": float(threshold),
        "production_rule": production_rule,
        "hard_label_reproduction": reproduction,
        "pipeline_adapter_validation": adapter_validation,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--physical-trajectories", type=Path, default=PHYSICAL_TRAJECTORIES)
    parser.add_argument("--old-stage-trajectories", type=Path, default=OLD_STAGE_TRAJECTORIES)
    parser.add_argument("--dream-consensus", type=Path, default=DREAM_CONSENSUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--temp-dir", type=Path, default=Path("/private/tmp"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
