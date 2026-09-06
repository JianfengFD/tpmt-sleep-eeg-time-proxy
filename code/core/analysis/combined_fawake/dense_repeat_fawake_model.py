#!/usr/bin/env python3
"""Physical self-similarity-first ``f_awake`` with recurrence micro-tuning.

The primary term ``F_s`` is built only from local structure-function scores
``S(t; Delta, gamma)``. The robust default fixes one long/short scale pair,
forms the two-scale persistence ratio at each of the nine predeclared gamma
thresholds,

    Q_gamma(t) = (S(t; Delta_long, gamma) + eps)
                 / (S(t; Delta_short, gamma) + eps),

and uses ``median_gamma Q_gamma``. This leaves only 20 candidate scale pairs
instead of selecting one of 288 scale/threshold grid points from a small
endpoint sample. The original single-S and single-gamma ratios remain
available through ``fs_candidate_policy='legacy_full_grid'`` for sensitivity
analysis.

In every participant-grouped CV fold, the training-only N3 and W medians of Q
(or S) are mapped smoothly to 0.05 and 0.95 by a logistic curve. There is no
hard clipping at the two anchors.

The primary short-time modifier is one signed return-prominence feature P from
``augment_adjacent_recurrence_grid.py``. Its training-fold N3/W medians are
likewise mapped to 0.05/0.95, now written as a tanh calibration, to obtain R.
The endpoint-fixed bounded fusion is

    f = F_s + alpha F_s (1 - F_s) (2 R - 1),   0 <= alpha <= 1.

Thus R can only modify the S-derived value, and its effect vanishes as F_s
approaches zero or one. The older thresholded adjacent-block repeat fraction
is retained only as the explicit ``block_fraction`` sensitivity mode.

All eligible feature definitions are enumerated below. Stage labels select a
definition and fit its N3/W anchors using participant-grouped CV. Dream
durations and dream text are never read by this module.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


CALIBRATION_EPSILON = 0.05
# Backward-compatible public name; the revised smooth mapping uses 0.05/0.95.
ENDPOINT_EPSILON = CALIBRATION_EPSILON
PERSISTENCE_RATIO_EPSILON = 0.02
RANDOM_STATE = 20260904

SELF_SIMILARITY_DELTA_GRID_S = (
    0.10,
    0.20,
    0.35,
    0.50,
    0.75,
    1.00,
    1.50,
    2.20,
    3.30,
    5.00,
    7.50,
    9.50,
)
SELF_SIMILARITY_GAMMA_GRID = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.75)
PERSISTENCE_LONG_DELTA_GRID_S = (3.30, 5.00, 7.50, 9.50)
PERSISTENCE_SHORT_DELTA_GRID_S = (0.50, 0.75, 1.00, 1.50, 2.20)

RECURRENCE_CONTEXT_HALF_WINDOW_GRID_S = (0.50, 1.00, 2.00, 5.00)
RECURRENCE_TAU_GRID_S = (0.04, 0.05, 0.06, 0.08, 0.10)

# Older adjacent-block fraction grid, sensitivity analysis only.
ADJACENT_CONTEXT_HALF_WINDOW_GRID_S = (0.50, 1.00, 2.00, 5.00)
ADJACENT_TAU_GRID_S = (0.04, 0.05, 0.06, 0.08, 0.10)
ADJACENT_ETA_GRID = (0.00, 0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.95)

ALPHA_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 9))
ALPHA_COMPLEXITY_PENALTY = 0.005
REPEAT_MODES = ("return_prominence", "block_fraction")
FS_CANDIDATE_POLICIES = ("robust_gamma_median", "legacy_full_grid", "all")
INNER_CV_REPEATS = 3
INNER_CV_REPEAT_SEED_STEP = 7919


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _s_column(delta_s: float, gamma: float) -> str:
    return f"structure_S_d{_token(delta_s)}_g{_token(gamma)}"


FIXED_SELF_SIMILARITY_COLUMNS = tuple(
    _s_column(delta, gamma)
    for delta in SELF_SIMILARITY_DELTA_GRID_S
    for gamma in SELF_SIMILARITY_GAMMA_GRID
)
FIXED_RETURN_PROMINENCE_COLUMNS = tuple(
    f"recurrence_P_h{_token(context)}_tau{_token(tau)}"
    for context in RECURRENCE_CONTEXT_HALF_WINDOW_GRID_S
    for tau in RECURRENCE_TAU_GRID_S
)
FIXED_ADJACENT_REPEAT_COLUMNS = tuple(
    f"adjacent_repeat_ctxh{_token(context)}_tau{_token(tau)}_eta{_token(eta)}"
    for context in ADJACENT_CONTEXT_HALF_WINDOW_GRID_S
    for tau in ADJACENT_TAU_GRID_S
    for eta in ADJACENT_ETA_GRID
)


def _build_fs_candidates() -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for column in FIXED_SELF_SIMILARITY_COLUMNS:
        key = f"single__{column.removeprefix('structure_S_')}"
        candidates[key] = {
            "kind": "single_S",
            "column": column,
            "formula": column,
        }
    for gamma in SELF_SIMILARITY_GAMMA_GRID:
        for long_delta in PERSISTENCE_LONG_DELTA_GRID_S:
            for short_delta in PERSISTENCE_SHORT_DELTA_GRID_S:
                long_column = _s_column(long_delta, gamma)
                short_column = _s_column(short_delta, gamma)
                key = (
                    f"persistence_ratio__long_d{_token(long_delta)}"
                    f"__short_d{_token(short_delta)}__g{_token(gamma)}"
                )
                candidates[key] = {
                    "kind": "persistence_ratio",
                    "long_column": long_column,
                    "short_column": short_column,
                    "gamma": float(gamma),
                    "long_delta_s": float(long_delta),
                    "short_delta_s": float(short_delta),
                    "ratio_epsilon": PERSISTENCE_RATIO_EPSILON,
                    "formula": f"({long_column}+0.02)/({short_column}+0.02)",
                }
    # A priori robustness family: retain a concrete (Delta_long, Delta_short)
    # pair but marginalize the nuisance threshold gamma over the complete fixed
    # extraction grid. This cuts the selected family from 288 noisy grid points
    # to 20 physical scale pairs and prevents one small fold from choosing an
    # idiosyncratic threshold. No fitted weights or non-S features are added.
    for long_delta in PERSISTENCE_LONG_DELTA_GRID_S:
        for short_delta in PERSISTENCE_SHORT_DELTA_GRID_S:
            key = (
                f"gamma_median_persistence__long_d{_token(long_delta)}"
                f"__short_d{_token(short_delta)}"
            )
            gamma_pairs = [
                {
                    "gamma": float(gamma),
                    "long_column": _s_column(long_delta, gamma),
                    "short_column": _s_column(short_delta, gamma),
                }
                for gamma in SELF_SIMILARITY_GAMMA_GRID
            ]
            candidates[key] = {
                "kind": "gamma_median_persistence_ratio",
                "long_delta_s": float(long_delta),
                "short_delta_s": float(short_delta),
                "gamma_grid": [float(gamma) for gamma in SELF_SIMILARITY_GAMMA_GRID],
                "gamma_pairs": gamma_pairs,
                "ratio_epsilon": PERSISTENCE_RATIO_EPSILON,
                "formula": (
                    "median_gamma[(S(long,gamma)+0.02)/(S(short,gamma)+0.02)]"
                ),
            }
    return candidates


FIXED_FS_CANDIDATES = _build_fs_candidates()


def _fs_candidate_allowed(spec: dict[str, Any], policy: str) -> bool:
    if policy not in FS_CANDIDATE_POLICIES:
        raise ValueError(f"fs_candidate_policy must be one of {FS_CANDIDATE_POLICIES}")
    robust = spec["kind"] == "gamma_median_persistence_ratio"
    if policy == "robust_gamma_median":
        return robust
    if policy == "legacy_full_grid":
        return not robust
    return True


def available_fs_candidates(
    frame: pd.DataFrame,
    *,
    policy: str = "robust_gamma_median",
) -> dict[str, dict[str, Any]]:
    """Return only predeclared Fs candidates supported by ``frame``."""
    result: dict[str, dict[str, Any]] = {}
    for key, spec in FIXED_FS_CANDIDATES.items():
        if not _fs_candidate_allowed(spec, policy):
            continue
        required = _candidate_required_s_columns(spec)
        if all(column in frame.columns for column in required):
            result[key] = copy.deepcopy(spec)
    if not result:
        raise ValueError("No Fs candidate from the declared fixed S grid is present")
    return result


def fs_physical_score(
    frame: pd.DataFrame, candidate: str | dict[str, Any]
) -> np.ndarray:
    """Evaluate one fixed single-S or two-scale persistence-ratio candidate."""
    spec = FIXED_FS_CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    kind = str(spec["kind"])
    if kind == "single_S":
        return frame[str(spec["column"])].to_numpy(float)
    if kind == "persistence_ratio":
        long_value = frame[str(spec["long_column"])].to_numpy(float)
        short_value = frame[str(spec["short_column"])].to_numpy(float)
        epsilon = float(spec.get("ratio_epsilon", PERSISTENCE_RATIO_EPSILON))
        return (long_value + epsilon) / (short_value + epsilon)
    if kind == "gamma_median_persistence_ratio":
        epsilon = float(spec.get("ratio_epsilon", PERSISTENCE_RATIO_EPSILON))
        ratios = [
            (
                frame[str(pair["long_column"])].to_numpy(float) + epsilon
            )
            / (frame[str(pair["short_column"])].to_numpy(float) + epsilon)
            for pair in spec["gamma_pairs"]
        ]
        return np.median(np.column_stack(ratios), axis=1)
    raise ValueError(f"Unsupported Fs candidate kind: {kind}")


def _check_epsilon(endpoint_epsilon: float) -> float:
    epsilon = float(endpoint_epsilon)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("endpoint_epsilon must lie strictly between 0 and 0.5")
    return epsilon


def smooth_logistic_calibration(
    values: np.ndarray | Iterable[float],
    n3_anchor: float,
    wake_anchor: float,
    endpoint_epsilon: float = CALIBRATION_EPSILON,
) -> np.ndarray:
    """Smoothly map the N3/W anchors to epsilon/1-epsilon.

    Values beyond the anchors continue smoothly toward zero or one and are not
    clipped to the anchor targets. The raw Fs W median must exceed N3.
    """
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Fs values must be finite")
    if not (math.isfinite(n3_anchor) and math.isfinite(wake_anchor)):
        raise ValueError("Fs anchors must be finite")
    if not wake_anchor > n3_anchor:
        raise ValueError("The raw Fs W median must exceed the N3 median")
    epsilon = _check_epsilon(endpoint_epsilon)
    midpoint = 0.5 * (float(n3_anchor) + float(wake_anchor))
    slope = 2.0 * math.log((1.0 - epsilon) / epsilon) / (
        float(wake_anchor) - float(n3_anchor)
    )
    z = slope * (raw - midpoint)
    # Numerically stable logistic without hard clipping.
    output = np.empty_like(z, dtype=float)
    positive = z >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    output[~positive] = exp_z / (1.0 + exp_z)
    return output


# Backward-compatible semantic alias used by the first implementation.
calibrate_self_similarity = smooth_logistic_calibration


def smooth_tanh_recurrence_calibration(
    values: np.ndarray | Iterable[float],
    n3_anchor: float,
    wake_anchor: float,
    endpoint_epsilon: float = CALIBRATION_EPSILON,
) -> np.ndarray:
    """Map recurrence P anchors smoothly to epsilon/1-epsilon.

    The physical direction is fixed rather than learned: higher raw return
    prominence must mean higher R. A candidate whose W median does not exceed
    its N3 median is rejected instead of being silently sign-flipped.
    """
    raw = np.asarray(values, dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("recurrence P values must be finite")
    if not (math.isfinite(n3_anchor) and math.isfinite(wake_anchor)):
        raise ValueError("recurrence anchors must be finite")
    difference = float(wake_anchor) - float(n3_anchor)
    if difference <= 1.0e-12:
        raise ValueError("The raw recurrence P W median must exceed the N3 median")
    epsilon = _check_epsilon(endpoint_epsilon)
    midpoint = 0.5 * (float(n3_anchor) + float(wake_anchor))
    target = 1.0 - 2.0 * epsilon
    scale = 2.0 * np.arctanh(target) / difference
    return 0.5 + 0.5 * np.tanh(scale * (raw - midpoint))


def dense_repeat_transform(
    f_self_similarity: np.ndarray | Iterable[float],
    adjacent_repeat: np.ndarray | Iterable[float],
    alpha: float,
) -> np.ndarray:
    """Apply the bounded endpoint-fixed recurrence correction."""
    fs = np.asarray(f_self_similarity, dtype=float)
    repeat = np.asarray(adjacent_repeat, dtype=float)
    if fs.shape != repeat.shape:
        raise ValueError("f_self_similarity and adjacent_repeat must have the same shape")
    if not math.isfinite(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and lie in [0, 1]")
    if not (np.isfinite(fs).all() and np.isfinite(repeat).all()):
        raise ValueError("F_s and R must be finite")
    tolerance = 1.0e-12
    if np.any((fs < -tolerance) | (fs > 1.0 + tolerance)):
        raise ValueError("F_s must lie in [0, 1]")
    if np.any((repeat < -tolerance) | (repeat > 1.0 + tolerance)):
        raise ValueError("R must lie in [0, 1]")
    # These clips only absorb floating-point round-off at 0/1. The fusion
    # itself is already bounded: x^2 <= f <= 2x-x^2.
    fs = np.clip(fs, 0.0, 1.0)
    repeat = np.clip(repeat, 0.0, 1.0)
    return fs + float(alpha) * fs * (1.0 - fs) * (2.0 * repeat - 1.0)


def _safe_auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, values))


def _pair_auc(
    stages: np.ndarray,
    values: np.ndarray,
    positive: str,
    negative: str,
    subset: np.ndarray | None = None,
) -> float | None:
    mask = np.isin(stages, [positive, negative])
    if subset is not None:
        mask &= subset
    return _safe_auc((stages[mask] == positive).astype(int), values[mask])


def _number(value: float | None, fallback: float = 0.5) -> float:
    return fallback if value is None or not math.isfinite(value) else float(value)


def score_stage_candidate(
    frame: pd.DataFrame, values: np.ndarray | Iterable[float]
) -> dict[str, Any]:
    """Return the predeclared, dream-blind OOF stage-selection audit."""
    required = {"manual_final_stage", "dataset"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing stage-audit columns: {missing}")
    candidate = np.asarray(values, dtype=float)
    if candidate.shape != (len(frame),):
        raise ValueError("values must contain exactly one value per frame row")
    if not np.isfinite(candidate).all():
        raise ValueError("candidate values must be finite")
    if np.any((candidate < -1.0e-12) | (candidate > 1.0 + 1.0e-12)):
        raise ValueError("candidate f values must lie in [0, 1]")
    candidate = np.clip(candidate, 0.0, 1.0)
    stages = frame["manual_final_stage"].astype(str).to_numpy()
    zhang = frame["dataset"].astype(str).str.casefold().eq("zhang").to_numpy()

    auc_zhang_w_n1 = _pair_auc(stages, candidate, "W", "N1", zhang)
    auc_zhang_w_n2 = _pair_auc(stages, candidate, "W", "N2", zhang)
    auc_zhang_w_n3 = _pair_auc(stages, candidate, "W", "N3", zhang)
    auc_zhang_w_rem = _pair_auc(stages, candidate, "W", "REM", zhang)
    known = np.isin(stages, ["W", "N1", "N2", "N3", "REM"])
    known_zhang = known & zhang
    auc_zhang_w_rest = _safe_auc(
        (stages[known_zhang] == "W").astype(int), candidate[known_zhang]
    )

    # Pooled comparisons are reported for diagnosis only. Since W occurs only
    # in Zhang, using these to select parameters would introduce dataset
    # identity as a shortcut.
    pooled_auc = {
        stage: _pair_auc(stages, candidate, "W", stage)
        for stage in ("N1", "N2", "N3", "REM")
    }
    pooled_auc_w_rest = _safe_auc(
        (stages[known] == "W").astype(int), candidate[known]
    )

    stage_medians: dict[str, float | None] = {}
    zhang_stage_medians: dict[str, float | None] = {}
    for stage in ("W", "N1", "N2", "N3", "REM"):
        group = candidate[stages == stage]
        stage_medians[stage] = float(np.median(group)) if len(group) else None
        zhang_group = candidate[(stages == stage) & zhang]
        zhang_stage_medians[stage] = (
            float(np.median(zhang_group)) if len(zhang_group) else None
        )
    w_median = stage_medians["W"]
    n1_median = stage_medians["N1"]
    n3_median = stage_medians["N3"]
    endpoint_error = None
    endpoint_score = None
    if w_median is not None and n3_median is not None:
        endpoint_error = 0.5 * (abs(1.0 - w_median) + abs(n3_median))
        endpoint_score = float(np.clip(1.0 - endpoint_error, 0.0, 1.0))
    zhang_w_n1_gap = None
    if (
        zhang_stage_medians["W"] is not None
        and zhang_stage_medians["N1"] is not None
    ):
        zhang_w_n1_gap = float(
            zhang_stage_medians["W"] - zhang_stage_medians["N1"]
        )
    nonwake = candidate[known & (stages != "W")]
    zhang_nonwake = candidate[known_zhang & (stages != "W")]
    nonwake_high = float(np.mean(nonwake > 0.90)) if len(nonwake) else None
    zhang_nonwake_high = (
        float(np.mean(zhang_nonwake > 0.90)) if len(zhang_nonwake) else None
    )

    # W-vs-N1 inside Zhang remains the dominant term. N2 and REM are explicit
    # safeguards in the revised search, alongside N3 and W-vs-rest.
    objective = (
        0.48 * _number(auc_zhang_w_n1)
        + 0.10 * _number(auc_zhang_w_n2)
        + 0.12 * _number(auc_zhang_w_n3)
        + 0.08 * _number(auc_zhang_w_rem)
        + 0.10 * _number(auc_zhang_w_rest)
        + 0.08 * _number(endpoint_score)
        + 0.04 * float(np.clip(_number(zhang_w_n1_gap, 0.0), 0.0, 1.0))
        - 0.03 * _number(zhang_nonwake_high, 1.0)
    )
    return {
        "auc_Zhang_W_vs_N1": auc_zhang_w_n1,
        "auc_Zhang_W_vs_N2": auc_zhang_w_n2,
        "auc_Zhang_W_vs_N3": auc_zhang_w_n3,
        "auc_Zhang_W_vs_REM": auc_zhang_w_rem,
        "auc_Zhang_W_vs_rest": auc_zhang_w_rest,
        # Compatibility names now deliberately mean the Zhang-only selection
        # audit. Pooled values have unambiguous names below.
        "auc_W_vs_N1": auc_zhang_w_n1,
        "auc_W_vs_N2": auc_zhang_w_n2,
        "auc_W_vs_N3": auc_zhang_w_n3,
        "auc_W_vs_REM": auc_zhang_w_rem,
        "auc_W_vs_rest": auc_zhang_w_rest,
        "pooled_auc_W_vs_N1": pooled_auc["N1"],
        "pooled_auc_W_vs_N2": pooled_auc["N2"],
        "pooled_auc_W_vs_N3": pooled_auc["N3"],
        "pooled_auc_W_vs_REM": pooled_auc["REM"],
        "pooled_auc_W_vs_rest": pooled_auc_w_rest,
        "stage_medians": stage_medians,
        "Zhang_stage_medians": zhang_stage_medians,
        "W_median": w_median,
        "N1_median": n1_median,
        "N2_median": stage_medians["N2"],
        "N3_median": n3_median,
        "REM_median": stage_medians["REM"],
        "Zhang_W_minus_N1_median": zhang_w_n1_gap,
        "W_minus_N1_median": zhang_w_n1_gap,
        "endpoint_mean_absolute_error_to_1_0": endpoint_error,
        "endpoint_score": endpoint_score,
        "nonW_fraction_above_0p90": nonwake_high,
        "Zhang_nonW_fraction_above_0p90": zhang_nonwake_high,
        "observed_min": float(np.min(candidate)),
        "observed_max": float(np.max(candidate)),
        "bounded_unit_interval": True,
        "selection_objective": float(objective),
    }


def _validate_feature_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    kind: str,
    lower: float,
    upper: float,
) -> None:
    for column in columns:
        values = frame[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{kind} column {column!r} contains non-finite values")
        if np.any((values < lower - 1.0e-12) | (values > upper + 1.0e-12)):
            raise ValueError(f"{kind} column {column!r} is outside [{lower}, {upper}]")


def _candidate_required_s_columns(spec: dict[str, Any]) -> list[str]:
    if spec["kind"] == "single_S":
        return [str(spec["column"])]
    if spec["kind"] == "persistence_ratio":
        return [str(spec["long_column"]), str(spec["short_column"])]
    if spec["kind"] == "gamma_median_persistence_ratio":
        return list(
            dict.fromkeys(
                str(pair[column])
                for pair in spec["gamma_pairs"]
                for column in ("long_column", "short_column")
            )
        )
    raise ValueError(f"Unsupported Fs candidate kind: {spec['kind']}")


def _endpoint_anchors_from_values(
    frame: pd.DataFrame,
    raw_values: np.ndarray,
    *,
    require_increasing: bool,
) -> tuple[float, float]:
    stages = frame["manual_final_stage"].astype(str).to_numpy()
    n3 = raw_values[stages == "N3"]
    wake = raw_values[stages == "W"]
    if len(n3) == 0 or len(wake) == 0:
        raise ValueError("Every calibration training set must contain N3 and W")
    n3_anchor = float(np.median(n3))
    wake_anchor = float(np.median(wake))
    if require_increasing and not wake_anchor > n3_anchor:
        raise ValueError("The W raw median must exceed the N3 median")
    if not require_increasing and abs(wake_anchor - n3_anchor) <= 1.0e-12:
        raise ValueError("The W and N3 raw medians must differ")
    return n3_anchor, wake_anchor


def _grouped_splits(
    frame: pd.DataFrame,
    requested_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    groups = frame["subject_group"].astype(str).to_numpy()
    stages = frame["manual_final_stage"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two subject_group values are required")
    required_counts = [
        len(np.unique(groups[stages == stage])) for stage in ("W", "N1", "N3")
    ]
    if min(required_counts) < 2:
        raise ValueError("Grouped CV requires W, N1, and N3 in at least two groups each")
    n_splits = min(int(requested_splits), len(unique_groups), min(required_counts))
    if n_splits < 2:
        raise ValueError("n_splits must permit at least two grouped folds")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )
    splits = [(fit, validation) for fit, validation in splitter.split(frame, stages, groups)]
    audit: list[dict[str, Any]] = []
    for fold_index, (fit, validation) in enumerate(splits, start=1):
        fit_groups = set(groups[fit])
        validation_groups = set(groups[validation])
        if fit_groups & validation_groups:
            raise RuntimeError("subject_group leakage detected")
        audit.append(
            {
                "fold": fold_index,
                "fit_groups": sorted(fit_groups),
                "validation_groups": sorted(validation_groups),
            }
        )
    return splits, {
        "kind": "StratifiedGroupKFold",
        "n_splits": n_splits,
        "shuffle": True,
        "random_state": int(random_state),
        "group": "subject_group",
        "folds": audit,
    }


def _repeated_grouped_splits(
    frame: pd.DataFrame,
    requested_splits: int,
    random_state: int,
    repeats: int,
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], dict[str, Any]]:
    """Build deterministic repeated participant-grouped inner CV splits."""
    if isinstance(repeats, bool) or int(repeats) != repeats or int(repeats) < 1:
        raise ValueError("n_cv_repeats must be a positive integer")
    split_sets: list[list[tuple[np.ndarray, np.ndarray]]] = []
    repeat_audits: list[dict[str, Any]] = []
    for repeat_index in range(int(repeats)):
        seed = int(random_state) + INNER_CV_REPEAT_SEED_STEP * repeat_index
        splits, audit = _grouped_splits(frame, requested_splits, seed)
        split_sets.append(splits)
        repeat_audits.append(
            {
                "repeat": repeat_index + 1,
                "random_state": seed,
                "folds": audit["folds"],
            }
        )
    return split_sets, {
        "kind": "Repeated StratifiedGroupKFold",
        "n_splits": len(split_sets[0]),
        "n_repeats": int(repeats),
        "random_states": [item["random_state"] for item in repeat_audits],
        "seed_step": INNER_CV_REPEAT_SEED_STEP,
        "shuffle": True,
        "group": "subject_group",
        # First repeat retained for compatibility with existing audit tables.
        "folds": repeat_audits[0]["folds"],
        "repeats": repeat_audits,
    }


def _oof_fs_candidate(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(frame), np.nan, dtype=float)
    fold_anchors: list[dict[str, Any]] = []
    for fold_index, (fit_index, validation_index) in enumerate(splits, start=1):
        fit = frame.iloc[fit_index]
        validation = frame.iloc[validation_index]
        fit_raw = fs_physical_score(fit, spec)
        n3_anchor, wake_anchor = _endpoint_anchors_from_values(
            fit, fit_raw, require_increasing=True
        )
        output[validation_index] = smooth_logistic_calibration(
            fs_physical_score(validation, spec),
            n3_anchor,
            wake_anchor,
            CALIBRATION_EPSILON,
        )
        fold_anchors.append(
            {
                "fold": fold_index,
                "n3_anchor": n3_anchor,
                "wake_anchor": wake_anchor,
                "raw_orientation": "W_greater_than_N3",
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Incomplete grouped-OOF Fs prediction")
    return output, fold_anchors


def _oof_return_prominence(
    frame: pd.DataFrame,
    column: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(frame), np.nan, dtype=float)
    fold_anchors: list[dict[str, Any]] = []
    for fold_index, (fit_index, validation_index) in enumerate(splits, start=1):
        fit = frame.iloc[fit_index]
        fit_raw = fit[column].to_numpy(float)
        n3_anchor, wake_anchor = _endpoint_anchors_from_values(
            fit, fit_raw, require_increasing=True
        )
        output[validation_index] = smooth_tanh_recurrence_calibration(
            frame.iloc[validation_index][column].to_numpy(float),
            n3_anchor,
            wake_anchor,
            CALIBRATION_EPSILON,
        )
        fold_anchors.append(
            {
                "fold": fold_index,
                "n3_anchor": n3_anchor,
                "wake_anchor": wake_anchor,
                "raw_orientation": "W_greater_than_N3",
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Incomplete grouped-OOF recurrence prediction")
    return output, fold_anchors


def _selection_key(
    metrics: dict[str, Any], index: int
) -> tuple[float, float, float, float, int]:
    return (
        float(metrics["selection_objective"]),
        -float(metrics.get("repeat_selection_objective_sd", 0.0)),
        _number(metrics["auc_Zhang_W_vs_N1"], -math.inf),
        _number(metrics["endpoint_score"], -math.inf),
        -int(index),
    )


def _repeat_columns(frame: pd.DataFrame, repeat_mode: str) -> list[str]:
    if repeat_mode not in REPEAT_MODES:
        raise ValueError(f"repeat_mode must be one of {REPEAT_MODES}")
    fixed = (
        FIXED_RETURN_PROMINENCE_COLUMNS
        if repeat_mode == "return_prominence"
        else FIXED_ADJACENT_REPEAT_COLUMNS
    )
    available = [column for column in fixed if column in frame.columns]
    if not available:
        raise ValueError(f"No fixed-grid columns are present for repeat_mode={repeat_mode!r}")
    return available


def fit_dense_repeat_model(
    final_training_epochs: pd.DataFrame,
    *,
    n_splits: int = 4,
    random_state: int = RANDOM_STATE,
    repeat_mode: str = "return_prominence",
    fs_candidate_policy: str = "robust_gamma_median",
    n_cv_repeats: int = INNER_CV_REPEATS,
) -> dict[str, Any]:
    """Fit the two-stage physical mapping using training stage labels only."""
    required = {"dataset", "subject_group", "manual_final_stage"}
    missing = sorted(required - set(final_training_epochs.columns))
    if missing:
        raise ValueError(f"Missing dense-repeat training columns: {missing}")
    frame = final_training_epochs.reset_index(drop=True).copy()
    fs_candidates = available_fs_candidates(frame, policy=fs_candidate_policy)
    s_columns = sorted(
        {column for spec in fs_candidates.values() for column in _candidate_required_s_columns(spec)}
    )
    _validate_feature_columns(frame, s_columns, "self-similarity", 0.0, 1.0)
    repeat_columns = _repeat_columns(frame, repeat_mode)
    if repeat_mode == "return_prominence":
        _validate_feature_columns(frame, repeat_columns, "return-prominence", -1.0, 1.0)
    else:
        _validate_feature_columns(frame, repeat_columns, "block-repeat", 0.0, 1.0)
    split_sets, cv_audit = _repeated_grouped_splits(
        frame, n_splits, random_state, n_cv_repeats
    )

    # Phase 1: self-similarity only. No repeat feature is visible here.
    fs_comparison: dict[str, dict[str, Any]] = {}
    fs_oof: dict[str, np.ndarray] = {}
    for candidate_index, (key, spec) in enumerate(fs_candidates.items()):
        repeat_oof: list[np.ndarray] = []
        repeat_metrics: list[dict[str, Any]] = []
        repeat_anchors: list[dict[str, Any]] = []
        valid = True
        for repeat_index, splits in enumerate(split_sets, start=1):
            try:
                oof, fold_anchors = _oof_fs_candidate(frame, spec, splits)
            except ValueError:
                valid = False
                break
            repeat_oof.append(oof)
            repeat_metrics.append(score_stage_candidate(frame, oof))
            repeat_anchors.append(
                {"repeat": repeat_index, "fold_anchors": fold_anchors}
            )
        if not valid:
            continue
        mean_oof = np.mean(np.row_stack(repeat_oof), axis=0)
        metrics = score_stage_candidate(frame, mean_oof)
        repeat_objectives = [
            float(item["selection_objective"]) for item in repeat_metrics
        ]
        mean_prediction_objective = float(metrics["selection_objective"])
        metrics["mean_OOF_prediction_objective"] = mean_prediction_objective
        metrics["repeat_selection_objectives"] = repeat_objectives
        metrics["repeat_selection_objective_mean"] = float(
            np.mean(repeat_objectives)
        )
        metrics["repeat_selection_objective_sd"] = float(
            np.std(repeat_objectives, ddof=0)
        )
        # Candidate selection is the expected objective across deterministic
        # grouped-CV repeats. SD is only a tie-break, not a tuned penalty.
        metrics["selection_objective"] = metrics[
            "repeat_selection_objective_mean"
        ]
        metrics.update(
            {
                "Fs_candidate": key,
                "Fs_candidate_spec": copy.deepcopy(spec),
                "candidate_grid_index": candidate_index,
                "fold_anchors": repeat_anchors[0]["fold_anchors"],
                "repeat_fold_anchors": repeat_anchors,
                "calibration": (
                    "smooth logistic; N3->0.05, W->0.95; deterministic "
                    f"{len(split_sets)}-repeat grouped CV"
                ),
            }
        )
        fs_comparison[key] = metrics
        fs_oof[key] = mean_oof
    if not fs_comparison:
        raise RuntimeError("No fixed Fs candidate had ordered W/N3 anchors in every fold")
    selected_fs_key = max(
        fs_comparison,
        key=lambda key: _selection_key(
            fs_comparison[key], int(fs_comparison[key]["candidate_grid_index"])
        ),
    )
    selected_fs_spec = copy.deepcopy(fs_candidates[selected_fs_key])
    selected_fs_oof = fs_oof[selected_fs_key]

    # Phase 2: one recurrence definition and alpha modify the locked OOF Fs.
    repeat_comparison: dict[str, dict[str, Any]] = {}
    best_key: str | None = None
    best_order: tuple[float, float, float, float, int] | None = None
    for repeat_index, repeat_column in enumerate(repeat_columns):
        try:
            if repeat_mode == "return_prominence":
                repeat_oof_values: list[np.ndarray] = []
                repeat_fold_anchors = []
                for cv_repeat, splits in enumerate(split_sets, start=1):
                    repeat_oof, fold_anchors = _oof_return_prominence(
                        frame, repeat_column, splits
                    )
                    repeat_oof_values.append(repeat_oof)
                    repeat_fold_anchors.append(
                        {"repeat": cv_repeat, "fold_anchors": fold_anchors}
                    )
                repeat = np.mean(np.row_stack(repeat_oof_values), axis=0)
                repeat_calibration = (
                    "fold-training N3/W tanh to 0.05/0.95; averaged over "
                    f"{len(split_sets)} deterministic grouped-CV repeats"
                )
            else:
                repeat = frame[repeat_column].to_numpy(float)
                repeat_fold_anchors = []
                repeat_calibration = "identity (legacy block-fraction sensitivity)"
        except ValueError:
            continue
        for alpha in ALPHA_GRID:
            adjusted = dense_repeat_transform(selected_fs_oof, repeat, alpha)
            metrics = score_stage_candidate(frame, adjusted)
            regularized = float(
                metrics["selection_objective"] - ALPHA_COMPLEXITY_PENALTY * alpha
            )
            correction = adjusted - selected_fs_oof
            key = f"{repeat_column}__alpha_{_token(alpha)}"
            metrics.update(
                {
                    "Fs_candidate": selected_fs_key,
                    "repeat_mode": repeat_mode,
                    "repeat_column": repeat_column,
                    # Compatibility field retained for downstream table code.
                    "adjacent_repeat_column": repeat_column,
                    "repeat_calibration": repeat_calibration,
                    "repeat_fold_anchors": repeat_fold_anchors,
                    "alpha": float(alpha),
                    "maximum_possible_absolute_correction": float(0.25 * alpha),
                    "observed_mean_absolute_correction_oof": float(
                        np.mean(np.abs(correction))
                    ),
                    "observed_maximum_absolute_correction_oof": float(
                        np.max(np.abs(correction))
                    ),
                    "alpha_complexity_penalty": float(ALPHA_COMPLEXITY_PENALTY * alpha),
                    "regularized_selection_objective": regularized,
                }
            )
            repeat_comparison[key] = metrics
            order = (
                regularized,
                _number(metrics["auc_Zhang_W_vs_N1"], -math.inf),
                _number(metrics["endpoint_score"], -math.inf),
                -float(alpha),
                -repeat_index,
            )
            if best_order is None or order > best_order:
                best_order = order
                best_key = key
    if best_key is None:
        raise RuntimeError("No fixed-grid recurrence candidate could be calibrated")
    selected_repeat = repeat_comparison[best_key]
    selected_repeat_column = str(selected_repeat["repeat_column"])

    full_fs_raw = fs_physical_score(frame, selected_fs_spec)
    n3_anchor, wake_anchor = _endpoint_anchors_from_values(
        frame, full_fs_raw, require_increasing=True
    )
    if repeat_mode == "return_prominence":
        repeat_n3_anchor, repeat_wake_anchor = _endpoint_anchors_from_values(
            frame,
            frame[selected_repeat_column].to_numpy(float),
            require_increasing=True,
        )
        repeat_calibration_payload: dict[str, Any] = {
            "kind": "smooth_tanh_N3_W",
            "n3_anchor": repeat_n3_anchor,
            "wake_anchor": repeat_wake_anchor,
            "endpoint_epsilon": CALIBRATION_EPSILON,
            "raw_orientation": "W_greater_than_N3",
        }
    else:
        repeat_calibration_payload = {"kind": "identity_block_fraction"}

    selected_s_column = (
        str(selected_fs_spec["column"])
        if selected_fs_spec["kind"] == "single_S"
        else None
    )
    return {
        "format": "dense_repeat_fawake_v2",
        "definition": (
            "Fs=logistic(fixed self-similarity construction; robust default is the "
            "median across the nine predeclared gamma values of a fixed long/short "
            "persistence ratio); "
            "f=Fs+alpha*Fs*(1-Fs)*(2R-1); primary R is tanh-calibrated return prominence"
        ),
        "Fs_candidate_policy": fs_candidate_policy,
        "selected_Fs_candidate": selected_fs_key,
        "selected_Fs_candidate_spec": selected_fs_spec,
        "selected_S_column": selected_s_column,
        "selected_repeat_mode": repeat_mode,
        "selected_repeat_column": selected_repeat_column,
        "selected_adjacent_repeat_column": selected_repeat_column,
        "alpha": float(selected_repeat["alpha"]),
        "n3_anchor": n3_anchor,
        "wake_anchor": wake_anchor,
        "endpoint_epsilon": CALIBRATION_EPSILON,
        "Fs_calibration": {"kind": "smooth_logistic_N3_W"},
        "repeat_calibration": repeat_calibration_payload,
        "selection_cv": cv_audit,
        "selection_rule": (
            "phase 1 selects one fixed physical Fs candidate by the mean objective across "
            f"{len(split_sets)} deterministic participant-grouped CV repeats; "
            "phase 2 selects one return-prominence R and alpha on that locked OOF Fs; "
            "0.005*alpha penalizes unnecessary correction; Zhang W-vs-N1 has the largest weight"
        ),
        "selected_Fs_metrics_oof": copy.deepcopy(fs_comparison[selected_fs_key]),
        "selected_S_metrics_oof": copy.deepcopy(fs_comparison[selected_fs_key]),
        "selected_dense_repeat_metrics_oof": copy.deepcopy(selected_repeat),
        "Fs_candidate_comparison": fs_comparison,
        "S_candidate_comparison": fs_comparison,
        "repeat_candidate_comparison": repeat_comparison,
        "adjacent_repeat_candidate_comparison": repeat_comparison,
        "constraints": {
            "Fs_family": (
                "default: 20 fixed long/short persistence ratios, each marginalized by "
                "the median over all nine predeclared gamma values"
            ),
            "Fs_candidate_policy": fs_candidate_policy,
            "legacy_288_grid_role": "sensitivity only",
            "inner_cv_repeats": len(split_sets),
            "inner_cv_repeat_seed_step": INNER_CV_REPEAT_SEED_STEP,
            "only_one_repeat_column": True,
            "primary_repeat_mode": "return_prominence",
            "return_prominence_direction": "higher P must imply higher R and f",
            "block_fraction_role": "sensitivity only",
            "alpha_interval": [0.0, 1.0],
            "unit_interval_proof": "x^2 <= f <= 2x-x^2 for x=Fs in [0,1]",
            "mathematical_endpoints_fixed": True,
            "hard_anchor_clipping": False,
            "dream_duration_used": False,
        },
        "label_usage": {
            "manual_final_stage": "grid selection and fold-local N3/W anchors",
            "subject_group": "participant-grouped cross-validation",
            "dataset": "within-Zhang primary W-vs-N1 audit",
            "dream_duration": "not read",
            "dream_report": "not read",
        },
    }


def _validated_fs_spec(model: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    key = str(model["selected_Fs_candidate"])
    if key not in FIXED_FS_CANDIDATES:
        raise ValueError("The selected Fs candidate is not in the declared physical grid")
    expected = FIXED_FS_CANDIDATES[key]
    supplied = model.get("selected_Fs_candidate_spec", expected)
    for field, value in expected.items():
        if supplied.get(field) != value:
            raise ValueError(f"Fs candidate specification changed fixed field {field!r}")
    return key, copy.deepcopy(expected)


def _validated_repeat_column(model: dict[str, Any]) -> tuple[str, str]:
    mode = str(model["selected_repeat_mode"])
    column = str(model["selected_repeat_column"])
    allowed = (
        FIXED_RETURN_PROMINENCE_COLUMNS
        if mode == "return_prominence"
        else FIXED_ADJACENT_REPEAT_COLUMNS
        if mode == "block_fraction"
        else ()
    )
    if column not in allowed:
        raise ValueError("The selected repeat column is not in its declared fixed grid")
    return mode, column


def _apply_components(
    frame: pd.DataFrame, model: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, fs_spec = _validated_fs_spec(model)
    repeat_mode, repeat_column = _validated_repeat_column(model)
    required = _candidate_required_s_columns(fs_spec) + [repeat_column]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing selected physical feature columns: {missing}")
    raw_fs = fs_physical_score(frame, fs_spec)
    fs = smooth_logistic_calibration(
        raw_fs,
        float(model["n3_anchor"]),
        float(model["wake_anchor"]),
        float(model.get("endpoint_epsilon", CALIBRATION_EPSILON)),
    )
    raw_repeat = frame[repeat_column].to_numpy(float)
    if repeat_mode == "return_prominence":
        calibration = model["repeat_calibration"]
        repeat = smooth_tanh_recurrence_calibration(
            raw_repeat,
            float(calibration["n3_anchor"]),
            float(calibration["wake_anchor"]),
            float(calibration.get("endpoint_epsilon", CALIBRATION_EPSILON)),
        )
    else:
        repeat = raw_repeat
    final = dense_repeat_transform(fs, repeat, float(model["alpha"]))
    return raw_fs, fs, raw_repeat, repeat, final


def predict_dense_repeat_fawake(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    """Apply a fitted or portable model and return final ``f_awake``."""
    return _apply_components(frame, model)[-1]


def apply_dense_repeat_model(frame: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Return transparent raw/calibrated Fs, recurrence R, and final f."""
    result = frame.copy()
    raw_fs, fs, raw_repeat, repeat, final = _apply_components(result, model)
    result["self_similarity_Fs_raw"] = raw_fs
    # Compatibility name: this may now be a persistence ratio, not one S.
    result["self_similarity_S_raw"] = raw_fs
    result["f_self_similarity"] = fs
    result["repeat_feature_raw"] = raw_repeat
    result["adjacent_repeat_R"] = repeat
    result["f_awake_initial"] = final
    return result


def export_dense_repeat_model(model: dict[str, Any]) -> dict[str, Any]:
    """Export the minimal JSON-safe physical mapping and audit protocol."""
    if model.get("format") != "dense_repeat_fawake_v2":
        raise ValueError("Unsupported dense-repeat model format")
    _, fs_spec = _validated_fs_spec(model)
    repeat_mode, repeat_column = _validated_repeat_column(model)
    return {
        "format": "dense_repeat_fawake_portable_v2",
        "definition": str(model["definition"]),
        "Fs_candidate_policy": str(model["Fs_candidate_policy"]),
        "selected_Fs_candidate": str(model["selected_Fs_candidate"]),
        "selected_Fs_candidate_spec": fs_spec,
        "selected_S_column": model.get("selected_S_column"),
        "selected_repeat_mode": repeat_mode,
        "selected_repeat_column": repeat_column,
        "selected_adjacent_repeat_column": repeat_column,
        "alpha": float(model["alpha"]),
        "n3_anchor": float(model["n3_anchor"]),
        "wake_anchor": float(model["wake_anchor"]),
        "endpoint_epsilon": float(model["endpoint_epsilon"]),
        "Fs_calibration": copy.deepcopy(model["Fs_calibration"]),
        "repeat_calibration": copy.deepcopy(model["repeat_calibration"]),
        "selection_rule": str(model["selection_rule"]),
        "selection_cv": copy.deepcopy(model["selection_cv"]),
        "selected_Fs_metrics_oof": copy.deepcopy(model["selected_Fs_metrics_oof"]),
        "selected_S_metrics_oof": copy.deepcopy(model["selected_S_metrics_oof"]),
        "selected_dense_repeat_metrics_oof": copy.deepcopy(
            model["selected_dense_repeat_metrics_oof"]
        ),
        "constraints": copy.deepcopy(model["constraints"]),
        "label_usage": copy.deepcopy(model["label_usage"]),
    }


def apply_dense_repeat_portable_model(
    frame: pd.DataFrame, payload: dict[str, Any]
) -> pd.DataFrame:
    if payload.get("format") != "dense_repeat_fawake_portable_v2":
        raise ValueError("Unsupported portable dense-repeat model format")
    return apply_dense_repeat_model(frame, payload)


__all__ = [
    "ADJACENT_CONTEXT_HALF_WINDOW_GRID_S",
    "ADJACENT_ETA_GRID",
    "ADJACENT_TAU_GRID_S",
    "ALPHA_GRID",
    "CALIBRATION_EPSILON",
    "ENDPOINT_EPSILON",
    "FIXED_ADJACENT_REPEAT_COLUMNS",
    "FIXED_FS_CANDIDATES",
    "FIXED_RETURN_PROMINENCE_COLUMNS",
    "FIXED_SELF_SIMILARITY_COLUMNS",
    "PERSISTENCE_LONG_DELTA_GRID_S",
    "PERSISTENCE_RATIO_EPSILON",
    "PERSISTENCE_SHORT_DELTA_GRID_S",
    "RECURRENCE_CONTEXT_HALF_WINDOW_GRID_S",
    "RECURRENCE_TAU_GRID_S",
    "REPEAT_MODES",
    "SELF_SIMILARITY_DELTA_GRID_S",
    "SELF_SIMILARITY_GAMMA_GRID",
    "apply_dense_repeat_model",
    "apply_dense_repeat_portable_model",
    "available_fs_candidates",
    "calibrate_self_similarity",
    "dense_repeat_transform",
    "export_dense_repeat_model",
    "fit_dense_repeat_model",
    "fs_physical_score",
    "predict_dense_repeat_fawake",
    "score_stage_candidate",
    "smooth_logistic_calibration",
    "smooth_tanh_recurrence_calibration",
]
