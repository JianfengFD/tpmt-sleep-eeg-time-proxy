#!/usr/bin/env python3
"""Self-similarity-first f with a fixed return-recurrence micro-correction.

The primary quantity remains the robust physical ``F_S`` selected and
calibrated by :mod:`dense_repeat_fawake_model`. The modifier is fixed before
the final nested audit:

``P* = row_quantile_0.75(P(H=0.5 s,tau=0.08 s), P(H=0.5 s,tau=0.10 s))``.

Within every training fold, the Zhang all-non-W median of P* is mapped to 0.05
and the Zhang W median to 0.95 with a smooth logistic calibration. The
physical direction is never learned or flipped: a fold is rejected if its W
median does not exceed its non-W median. The endpoint-fixed fusion is

``f = sigmoid(logit(F_S) + beta * (2 R - 1)), beta=0.25``.

The modifier can therefore shift the F_S log-odds by at most 0.25. Exact F_S
endpoints are restored explicitly, so f(0)=0 and f(1)=1. P*, its quantile,
and beta are fixed design choices rather than parameters selected by the
nested data. Dream text, dream duration, and spectral classifier outputs are
never read by this module.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from analysis.combined_fawake.dense_repeat_fawake_model import (
    INNER_CV_REPEATS,
    _oof_fs_candidate,
    _repeated_grouped_splits,
    apply_dense_repeat_model,
    export_dense_repeat_model,
    fit_dense_repeat_model,
    score_stage_candidate,
    smooth_logistic_calibration,
)


RSTAR_COLUMNS = (
    "recurrence_P_h0p5_tau0p08",
    "recurrence_P_h0p5_tau0p1",
)
RSTAR_CONTEXT_HALF_WINDOW_S = 0.50
RSTAR_TAU_GRID_S = (0.08, 0.10)
RSTAR_ROW_QUANTILE = 0.75
RSTAR_CALIBRATION_EPSILON = 0.05
RSTAR_BETA = 0.25


def validate_adjacent_repeat_v3_model(model: dict[str, Any]) -> None:
    """Reject payloads that no longer encode the locked preferred formula."""
    if model.get("format") not in {
        "adjacent_repeat_logit_shift_v3",
        "adjacent_repeat_logit_shift_portable_v3",
    }:
        raise ValueError("Unsupported adjacent-repeat-v3 model format")
    base = model.get("base_model")
    if not isinstance(base, dict) or base.get("format") not in {
        "dense_repeat_fawake_v2",
        "dense_repeat_fawake_portable_v2",
    }:
        raise ValueError("R* payload must contain a fitted or portable dense-repeat-v2 base")

    exact_checks = (
        (list(model.get("repeat_columns", [])), list(RSTAR_COLUMNS), "repeat_columns"),
        (float(model.get("context_half_window_s", math.nan)), RSTAR_CONTEXT_HALF_WINDOW_S,
         "context_half_window_s"),
        (list(model.get("tau_grid_s", [])), list(RSTAR_TAU_GRID_S), "tau_grid_s"),
        (float(model.get("row_quantile", math.nan)), RSTAR_ROW_QUANTILE, "row_quantile"),
        (str(model.get("quantile_method", "")), "NumPy linear", "quantile_method"),
        (float(model.get("beta", math.nan)), RSTAR_BETA, "beta"),
        (float(model.get("calibration_epsilon", math.nan)), RSTAR_CALIBRATION_EPSILON,
         "calibration_epsilon"),
        (float(base.get("endpoint_epsilon", math.nan)), 0.05, "base endpoint_epsilon"),
    )
    for observed, expected, name in exact_checks:
        if isinstance(expected, float):
            valid = math.isfinite(float(observed)) and abs(float(observed) - expected) <= 1.0e-12
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(f"Locked R* field {name!r} changed: {observed!r}")

    nonwake_anchor = float(model.get("nonwake_anchor", math.nan))
    wake_anchor = float(model.get("wake_anchor", math.nan))
    if not (
        math.isfinite(nonwake_anchor)
        and math.isfinite(wake_anchor)
        and wake_anchor > nonwake_anchor + 1.0e-12
    ):
        raise ValueError("R* payload requires finite anchors with W > all-non-W")


def return_prominence_rstar(frame: pd.DataFrame) -> np.ndarray:
    """Fixed NumPy-linear rowwise 0.75 quantile of the two P values."""
    missing = [column for column in RSTAR_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing R* return-prominence columns: {missing}")
    matrix = frame.loc[:, RSTAR_COLUMNS].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("R* inputs must be finite")
    if np.any((matrix < -1.0 - 1.0e-12) | (matrix > 1.0 + 1.0e-12)):
        raise ValueError("R* return-prominence inputs must lie in [-1,1]")
    return np.quantile(matrix, RSTAR_ROW_QUANTILE, axis=1, method="linear")


def rstar_wake_nonwake_anchors(
    frame: pd.DataFrame, raw_rstar: np.ndarray | Iterable[float]
) -> tuple[float, float]:
    """Training-only Zhang non-W and W medians in the physical direction."""
    values = np.asarray(raw_rstar, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("raw_rstar must contain one finite value per row")
    stage = frame["manual_final_stage"].astype(str).to_numpy()
    zhang = frame["dataset"].astype(str).str.casefold().eq("zhang").to_numpy()
    known = np.isin(stage, ["W", "N1", "N2", "N3", "REM"])
    wake = values[zhang & (stage == "W")]
    nonwake = values[zhang & known & (stage != "W")]
    if len(wake) == 0 or len(nonwake) == 0:
        raise ValueError("R* calibration requires Zhang W and Zhang non-W rows")
    nonwake_anchor = float(np.median(nonwake))
    wake_anchor = float(np.median(wake))
    if wake_anchor - nonwake_anchor <= 1.0e-12:
        raise ValueError("Physical R* direction requires Zhang W median > non-W median")
    return nonwake_anchor, wake_anchor


def calibrate_rstar(
    values: np.ndarray | Iterable[float],
    nonwake_anchor: float,
    wake_anchor: float,
) -> np.ndarray:
    """Map training non-W/W anchors smoothly to 0.05/0.95."""
    return smooth_logistic_calibration(
        values,
        n3_anchor=float(nonwake_anchor),
        wake_anchor=float(wake_anchor),
        endpoint_epsilon=RSTAR_CALIBRATION_EPSILON,
    )


def symmetric_logit_micro_correction(
    f_self_similarity: np.ndarray | Iterable[float],
    repeat: np.ndarray | Iterable[float],
    beta: float = RSTAR_BETA,
) -> np.ndarray:
    """Apply a bounded symmetric logit shift and preserve exact F_S endpoints."""
    fs = np.asarray(f_self_similarity, dtype=float)
    r = np.asarray(repeat, dtype=float)
    if fs.shape != r.shape:
        raise ValueError("F_S and R must have the same shape")
    if not (np.isfinite(fs).all() and np.isfinite(r).all()):
        raise ValueError("F_S and R must be finite")
    if np.any((fs < -1.0e-12) | (fs > 1.0 + 1.0e-12)):
        raise ValueError("F_S must lie in [0,1]")
    if np.any((r < -1.0e-12) | (r > 1.0 + 1.0e-12)):
        raise ValueError("R must lie in [0,1]")
    strength = float(beta)
    if not math.isfinite(strength) or not 0.0 <= strength <= RSTAR_BETA:
        raise ValueError(f"beta must lie in [0,{RSTAR_BETA}]")
    fs = np.clip(fs, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)
    if strength == 0.0:
        return fs.copy()
    output = fs.copy()
    interior = (fs > 0.0) & (fs < 1.0)
    # Do not clip already-valid near-endpoint F_S values to an arbitrary
    # epsilon: doing so could make a positive shift decrease an F_S extremely
    # close to one. log/log1p retains the representable input log-odds.
    z = (
        np.log(fs[interior])
        - np.log1p(-fs[interior])
        + strength * (2.0 * r[interior] - 1.0)
    )
    output[interior] = expit(z)
    return output


def _oof_rstar(
    frame: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(frame), np.nan, dtype=float)
    raw = return_prominence_rstar(frame)
    anchors: list[dict[str, Any]] = []
    for fold_index, (fit_index, validation_index) in enumerate(splits, start=1):
        fit = frame.iloc[fit_index]
        nonwake_anchor, wake_anchor = rstar_wake_nonwake_anchors(fit, raw[fit_index])
        output[validation_index] = calibrate_rstar(
            raw[validation_index], nonwake_anchor, wake_anchor
        )
        anchors.append(
            {
                "fold": fold_index,
                "nonwake_anchor": nonwake_anchor,
                "wake_anchor": wake_anchor,
                "anchor_scope": "Zhang W versus all Zhang non-W",
                "physical_direction_valid": True,
            }
        )
    if not np.isfinite(output).all():
        raise RuntimeError("Incomplete grouped-OOF R* prediction")
    return output, anchors


def fit_adjacent_repeat_v3(
    final_training_epochs: pd.DataFrame,
    *,
    n_splits: int = 4,
    random_state: int = 20260904,
    n_cv_repeats: int = INNER_CV_REPEATS,
    base_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit fold-local R* anchors around a fully refitted self-similarity model."""
    frame = final_training_epochs.reset_index(drop=True).copy()
    required = {"dataset", "subject_group", "manual_final_stage", *RSTAR_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing adjacent-repeat-v3 training columns: {missing}")
    if base_model is None:
        base_model = fit_dense_repeat_model(
            frame,
            n_splits=n_splits,
            random_state=random_state,
            n_cv_repeats=n_cv_repeats,
        )
    if base_model.get("format") != "dense_repeat_fawake_v2":
        raise ValueError("base_model must be a fitted dense_repeat_fawake_v2 model")

    # Cross-fitted development audit. P*, q=.75, and beta=.25 are fixed and
    # are not selected from these predictions.
    selected_fs_spec = copy.deepcopy(base_model["selected_Fs_candidate_spec"])
    split_sets, cv_audit = _repeated_grouped_splits(
        frame, n_splits, random_state, n_cv_repeats
    )
    fs_repeats: list[np.ndarray] = []
    r_repeats: list[np.ndarray] = []
    anchor_repeats: list[dict[str, Any]] = []
    for repeat_index, splits in enumerate(split_sets, start=1):
        fs_oof, _ = _oof_fs_candidate(frame, selected_fs_spec, splits)
        r_oof, anchors = _oof_rstar(frame, splits)
        fs_repeats.append(fs_oof)
        r_repeats.append(r_oof)
        anchor_repeats.append({"repeat": repeat_index, "fold_anchors": anchors})
    fs_oof = np.mean(np.row_stack(fs_repeats), axis=0)
    r_oof = np.mean(np.row_stack(r_repeats), axis=0)
    corrected_oof = symmetric_logit_micro_correction(fs_oof, r_oof, RSTAR_BETA)

    raw_full = return_prominence_rstar(frame)
    nonwake_anchor, wake_anchor = rstar_wake_nonwake_anchors(frame, raw_full)
    return {
        "format": "adjacent_repeat_logit_shift_v3",
        "definition": (
            "P*=row q75 of P(H=0.5,tau=0.08/0.10); R is fold-training Zhang "
            "non-W/W logistic calibration to 0.05/0.95; "
            "f=sigmoid(logit(Fs)+0.25*(2R-1))"
        ),
        "base_model": base_model,
        "repeat_columns": list(RSTAR_COLUMNS),
        "context_half_window_s": RSTAR_CONTEXT_HALF_WINDOW_S,
        "tau_grid_s": list(RSTAR_TAU_GRID_S),
        "row_quantile": RSTAR_ROW_QUANTILE,
        "quantile_method": "NumPy linear",
        "beta": RSTAR_BETA,
        "nonwake_anchor": nonwake_anchor,
        "wake_anchor": wake_anchor,
        "calibration_epsilon": RSTAR_CALIBRATION_EPSILON,
        "selection_cv": cv_audit,
        "crossfit_repeat_anchors": anchor_repeats,
        "crossfit_pure_Fs_metrics": score_stage_candidate(frame, fs_oof),
        "crossfit_v3_metrics": score_stage_candidate(frame, corrected_oof),
        "selection_rule": (
            "P*, q=0.75, logistic direction, and beta=0.25 are fixed design choices; "
            "only the underlying robust Fs definition is training-selected by the v2 "
            "repeated participant-grouped procedure"
        ),
        "constraints": {
            "maximum_absolute_logit_shift": RSTAR_BETA,
            "exact_mathematical_endpoints_fixed": True,
            "repeat_calibration_scope": "Zhang W versus all Zhang non-W",
            "repeat_physical_direction": "higher P* must imply higher R; no sign flip",
            "dream_information_used": False,
            "test_labels_used_for_fit": False,
            "beta_selected_by_nested_cv": False,
        },
    }


def apply_adjacent_repeat_v3(frame: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Expose common F_S, the old-P comparator, and the fixed R* micro-correction."""
    validate_adjacent_repeat_v3_model(model)
    result = apply_dense_repeat_model(frame, model["base_model"])
    result["return_prominence_P_old_raw"] = result["repeat_feature_raw"].to_numpy(float)
    result["return_prominence_R_old"] = result["adjacent_repeat_R"].to_numpy(float)
    result["f_old_return_prominence"] = result["f_awake_initial"].to_numpy(float)
    raw = return_prominence_rstar(result)
    repeat = calibrate_rstar(raw, float(model["nonwake_anchor"]), float(model["wake_anchor"]))
    result["return_prominence_P_star"] = raw
    result["return_prominence_R_star"] = repeat
    result["f_repeat_micro_v3"] = symmetric_logit_micro_correction(
        result["f_self_similarity"].to_numpy(float),
        repeat,
        float(model.get("beta", RSTAR_BETA)),
    )
    return result


def export_adjacent_repeat_v3(model: dict[str, Any]) -> dict[str, Any]:
    """Return minimal JSON-safe prediction parameters plus the design audit."""
    if model.get("format") != "adjacent_repeat_logit_shift_v3":
        raise ValueError("Expected a fitted adjacent_repeat_logit_shift_v3 model")
    validate_adjacent_repeat_v3_model(model)
    return {
        "format": "adjacent_repeat_logit_shift_portable_v3",
        "definition": str(model["definition"]),
        "base_model": export_dense_repeat_model(model["base_model"]),
        "repeat_columns": list(model["repeat_columns"]),
        "context_half_window_s": float(model["context_half_window_s"]),
        "tau_grid_s": [float(value) for value in model["tau_grid_s"]],
        "row_quantile": float(model["row_quantile"]),
        "quantile_method": str(model["quantile_method"]),
        "beta": float(model["beta"]),
        "nonwake_anchor": float(model["nonwake_anchor"]),
        "wake_anchor": float(model["wake_anchor"]),
        "calibration_epsilon": float(model["calibration_epsilon"]),
        "selection_rule": str(model["selection_rule"]),
        "selection_cv": copy.deepcopy(model["selection_cv"]),
        "crossfit_pure_Fs_metrics": copy.deepcopy(model["crossfit_pure_Fs_metrics"]),
        "crossfit_v3_metrics": copy.deepcopy(model["crossfit_v3_metrics"]),
        "constraints": copy.deepcopy(model["constraints"]),
    }


__all__ = [
    "RSTAR_BETA",
    "RSTAR_CALIBRATION_EPSILON",
    "RSTAR_COLUMNS",
    "RSTAR_CONTEXT_HALF_WINDOW_S",
    "RSTAR_ROW_QUANTILE",
    "RSTAR_TAU_GRID_S",
    "apply_adjacent_repeat_v3",
    "calibrate_rstar",
    "export_adjacent_repeat_v3",
    "fit_adjacent_repeat_v3",
    "return_prominence_rstar",
    "rstar_wake_nonwake_anchors",
    "symmetric_logit_micro_correction",
    "validate_adjacent_repeat_v3_model",
]
