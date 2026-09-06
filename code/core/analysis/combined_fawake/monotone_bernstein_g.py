#!/usr/bin/env python3
"""Endpoint-fixed monotone shells for the dream-duration calibration step.

This module intentionally knows nothing about sleep-stage classifiers.  It
only maps an already constructed physical ``f(t)`` in [0, 1] through one
global scalar function ``g`` and integrates ``g(f(t)) dt`` record by record.

The main flexible family is a Bernstein polynomial

    g(x) = sum(k=0..n) c_k * B_{k,n}(x),

with c_0=0, c_n=1 and ordered control points.  Ordered control points make the
curve monotone, while the fixed end points preserve the awake/deep-sleep
anchors.  Degree n has n-1 free parameters, so degrees 2--5 provide the
requested one- through four-parameter sequence and may cross the identity in
the interior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar


Array = np.ndarray


@dataclass(frozen=True)
class ShellSpec:
    """Definition of one endpoint-fixed monotone shell family."""

    key: str
    label: str
    kind: str
    parameter_count: int
    degree: int | None = None
    primary_eligible: bool = True


@dataclass(frozen=True)
class ShellConfig:
    """A family plus a training-only regularization strength."""

    spec: ShellSpec
    regularization: float

    @property
    def key(self) -> str:
        return f"{self.spec.key}__lambda_{self.regularization:g}"


@dataclass
class FittedShell:
    """A fitted global g shell."""

    spec: ShellSpec
    regularization: float
    parameters: Array
    objective: float
    data_loss: float
    identity_penalty: float

    def transform(self, values: Array | Sequence[float]) -> Array:
        return transform(self.spec, np.asarray(values, dtype=float), self.parameters)

    def decoded_parameters(self) -> dict[str, object]:
        return decode_parameters(self.spec, self.parameters)


def shell_specs() -> list[ShellSpec]:
    """Return the predeclared primary candidates and below-identity control."""

    specs = [
        ShellSpec("identity", "Identity", "identity", 0),
        ShellSpec(
            "signed_exponential_1p",
            "Signed exponential (1 parameter)",
            "signed_exponential",
            1,
        ),
    ]
    for degree in range(2, 6):
        specs.append(
            ShellSpec(
                f"bernstein_degree_{degree}",
                f"Monotone Bernstein degree {degree} ({degree - 1} parameter"
                + ("" if degree == 2 else "s")
                + ")",
                "bernstein",
                degree - 1,
                degree=degree,
            )
        )
    specs.append(
        ShellSpec(
            "below_identity_exponential_1p",
            "Below-identity exponential control (1 parameter)",
            "below_exponential",
            1,
            primary_eligible=False,
        )
    )
    return specs


def _validate_values(values: Array) -> Array:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("g input contains a non-finite value")
    tolerance = 1e-10
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("g input must lie in [0, 1]")
    return np.clip(values, 0.0, 1.0)


def signed_exponential(values: Array, kappa: float) -> Array:
    """Endpoint-fixed exponential shell; either sign of kappa is allowed."""

    x = _validate_values(values)
    if not math.isfinite(kappa):
        raise ValueError("kappa must be finite")
    if abs(kappa) < 1e-8:
        result = x.copy()
    elif kappa > 50.0:
        result = (
            np.exp(kappa * (x - 1.0))
            * (-np.expm1(-kappa * x))
            / (-np.expm1(-kappa))
        )
    else:
        result = np.expm1(kappa * x) / np.expm1(kappa)
    return np.clip(result, 0.0, 1.0)


def bernstein_basis(values: Array, degree: int) -> Array:
    """Return B_(k,n)(x) for k=0..n as columns."""

    x = _validate_values(values).reshape(-1)
    if degree < 1:
        raise ValueError("Bernstein degree must be positive")
    columns = [
        math.comb(degree, k) * np.power(x, k) * np.power(1.0 - x, degree - k)
        for k in range(degree + 1)
    ]
    return np.column_stack(columns)


def bernstein_control_points(spec: ShellSpec, parameters: Array) -> Array:
    if spec.kind != "bernstein" or spec.degree is None:
        raise ValueError("Bernstein control points requested for a non-Bernstein shell")
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if len(parameters) != spec.parameter_count:
        raise ValueError(f"{spec.key}: expected {spec.parameter_count} parameters")
    return np.concatenate(([0.0], parameters, [1.0]))


def transform(spec: ShellSpec, values: Array, parameters: Array) -> Array:
    """Apply one shell exactly on [0, 1]."""

    x = _validate_values(values)
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if len(parameters) != spec.parameter_count:
        raise ValueError(f"{spec.key}: expected {spec.parameter_count} parameters")
    if spec.kind == "identity":
        result = x.copy()
    elif spec.kind == "signed_exponential":
        result = signed_exponential(x, float(parameters[0]))
    elif spec.kind == "below_exponential":
        if parameters[0] < -1e-12:
            raise ValueError("below-identity control requires kappa >= 0")
        result = signed_exponential(x, float(parameters[0]))
    elif spec.kind == "bernstein":
        flat = x.reshape(-1)
        controls = bernstein_control_points(spec, parameters)
        result = (bernstein_basis(flat, int(spec.degree)) @ controls).reshape(x.shape)
    else:
        raise ValueError(f"Unknown shell kind: {spec.kind}")
    # Set the mathematical anchors exactly, not only to floating precision.
    result = np.asarray(result, dtype=float)
    result = np.where(x == 0.0, 0.0, result)
    result = np.where(x == 1.0, 1.0, result)
    return np.clip(result, 0.0, 1.0)


def decode_parameters(spec: ShellSpec, parameters: Array) -> dict[str, object]:
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    if spec.kind == "identity":
        return {}
    if spec.kind in {"signed_exponential", "below_exponential"}:
        return {"kappa": float(parameters[0])}
    controls = bernstein_control_points(spec, parameters)
    return {
        "degree": int(spec.degree),
        "interior_control_points": [float(value) for value in parameters],
        "control_points": [float(value) for value in controls],
    }


def validate_shell(spec: ShellSpec, parameters: Array, tolerance: float = 1e-10) -> dict[str, object]:
    grid = np.linspace(0.0, 1.0, 20001)
    curve = transform(spec, grid, np.asarray(parameters, dtype=float))
    difference = np.diff(curve)
    delta_identity = curve - grid
    return {
        "g_at_0": float(curve[0]),
        "g_at_1": float(curve[-1]),
        "minimum": float(curve.min()),
        "maximum": float(curve.max()),
        "minimum_step": float(difference.min()),
        "maximum_above_identity": float(delta_identity.max()),
        "maximum_below_identity": float((-delta_identity).max()),
        "endpoint_fixed": bool(abs(curve[0]) <= tolerance and abs(curve[-1] - 1.0) <= tolerance),
        "bounded": bool(curve.min() >= -tolerance and curve.max() <= 1.0 + tolerance),
        "monotone": bool(np.all(difference >= -tolerance)),
        "crosses_identity": bool(delta_identity.max() > tolerance and delta_identity.min() < -tolerance),
        "never_above_identity": bool(delta_identity.max() <= tolerance),
    }


def identity_deviation(spec: ShellSpec, parameters: Array) -> float:
    """Common, family-independent regularizer: integral (g(x)-x)^2 dx."""

    grid = np.linspace(0.0, 1.0, 1001)
    curve = transform(spec, grid, parameters)
    # np.trapz keeps compatibility with the NumPy shipped in the local
    # analysis environment (np.trapezoid was added later).
    return float(np.trapz(np.square(curve - grid), grid))


def apparent_seconds(
    spec: ShellSpec,
    parameters: Array,
    values: Array,
    durations: Array,
) -> float:
    values = np.asarray(values, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if values.shape != durations.shape:
        raise ValueError("f values and durations must have identical shapes")
    if np.any(~np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("epoch durations must be finite and positive")
    return float(np.sum(transform(spec, values, parameters) * durations))


def _residuals(predicted: Array, target: Array, loss_mode: str) -> Array:
    predicted = np.asarray(predicted, dtype=float)
    target = np.asarray(target, dtype=float)
    if loss_mode == "log_mse":
        return np.log1p(predicted) - np.log1p(target)
    if loss_mode == "relative_mse":
        return (predicted - target) / np.maximum(target, 1.0)
    if loss_mode == "raw_mse":
        # Scaling preserves the raw-seconds least-squares optimum and keeps
        # numerical objective magnitudes convenient for SLSQP.
        return (predicted - target) / 1000.0
    raise ValueError(f"Unknown loss mode: {loss_mode}")


def balanced_loss(
    predicted: Array,
    target: Array,
    subject_ids: Sequence[str],
    loss_mode: str = "log_mse",
) -> float:
    """Record loss with each participant receiving equal total weight."""

    subject_ids = np.asarray(subject_ids, dtype=str)
    if len(predicted) != len(target) or len(target) != len(subject_ids):
        raise ValueError("prediction, target and subject arrays must have equal lengths")
    unique, counts = np.unique(subject_ids, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_map[subject] for subject in subject_ids], dtype=float)
    weights /= weights.sum()
    residual = _residuals(predicted, target, loss_mode)
    return float(np.sum(weights * np.square(residual)))


def _initial_parameters(spec: ShellSpec) -> list[Array]:
    if spec.kind == "identity":
        return [np.empty(0, dtype=float)]
    if spec.kind == "signed_exponential":
        return [np.asarray([value], dtype=float) for value in (0.0, -5.0, 5.0, -12.0, 12.0)]
    if spec.kind == "below_exponential":
        return [np.asarray([value], dtype=float) for value in (0.0, 2.0, 8.0, 16.0)]
    degree = int(spec.degree)
    fractions = np.arange(1, degree, dtype=float) / degree
    return [
        fractions,
        np.power(fractions, 0.5),
        np.power(fractions, 2.0),
        np.power(fractions, 0.25),
        np.power(fractions, 4.0),
    ]


def _bounds_and_constraints(spec: ShellSpec) -> tuple[list[tuple[float, float]], list[dict[str, object]]]:
    if spec.kind == "identity":
        return [], []
    if spec.kind == "signed_exponential":
        return [(-50.0, 500.0)], []
    if spec.kind == "below_exponential":
        return [(0.0, 500.0)], []
    bounds = [(0.0, 1.0) for _ in range(spec.parameter_count)]
    constraints: list[dict[str, object]] = []
    for index in range(spec.parameter_count - 1):
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda theta, index=index: float(theta[index + 1] - theta[index]),
            }
        )
    return bounds, constraints


def predict_records(
    spec: ShellSpec,
    parameters: Array,
    record_uids: Sequence[str],
    arrays: Mapping[str, tuple[Array, Array]],
) -> Array:
    return np.asarray(
        [apparent_seconds(spec, parameters, *arrays[uid]) for uid in record_uids],
        dtype=float,
    )


def fit_shell(
    config: ShellConfig,
    record_uids: Sequence[str],
    arrays: Mapping[str, tuple[Array, Array]],
    targets: Mapping[str, float],
    subjects: Mapping[str, str],
    loss_mode: str = "log_mse",
) -> FittedShell:
    """Fit one global shell to the supplied training records only."""

    uids = list(record_uids)
    target = np.asarray([targets[uid] for uid in uids], dtype=float)
    subject_ids = [subjects[uid] for uid in uids]

    # Bernstein record integrals are linear in their control points.  Cache
    # the sufficient statistics once because nested CV otherwise evaluates
    # the same epoch-level basis thousands of times during optimization.
    bernstein_record_basis: Array | None = None
    bernstein_grid_basis: Array | None = None
    regularizer_grid: Array | None = None
    if config.spec.kind == "bernstein":
        degree = int(config.spec.degree)
        basis_rows = []
        for uid in uids:
            values, durations = arrays[uid]
            basis_rows.append(
                np.sum(bernstein_basis(values, degree) * durations[:, None], axis=0)
            )
        bernstein_record_basis = np.vstack(basis_rows)
        regularizer_grid = np.linspace(0.0, 1.0, 1001)
        bernstein_grid_basis = bernstein_basis(regularizer_grid, degree)

    def pieces(theta: Array) -> tuple[float, float, float]:
        if bernstein_record_basis is not None:
            controls = bernstein_control_points(config.spec, theta)
            prediction = bernstein_record_basis @ controls
            curve = bernstein_grid_basis @ controls
            penalty = float(
                np.trapz(np.square(curve - regularizer_grid), regularizer_grid)
            )
        else:
            prediction = predict_records(config.spec, theta, uids, arrays)
            penalty = identity_deviation(config.spec, theta)
        data = balanced_loss(prediction, target, subject_ids, loss_mode=loss_mode)
        total = data + float(config.regularization) * penalty
        return total, data, penalty

    if config.spec.kind == "identity":
        theta = np.empty(0, dtype=float)
        total, data, penalty = pieces(theta)
        return FittedShell(config.spec, config.regularization, theta, total, data, penalty)

    bounds, constraints = _bounds_and_constraints(config.spec)
    if config.spec.kind in {"signed_exponential", "below_exponential"}:
        lower, upper = bounds[0]
        result = minimize_scalar(
            lambda value: pieces(np.asarray([value], dtype=float))[0],
            bounds=(lower, upper),
            method="bounded",
            options={"maxiter": 1200, "xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"{config.key} optimization failed: {result.message}")
        theta = np.asarray([result.x], dtype=float)
        total, data, penalty = pieces(theta)
        validation = validate_shell(config.spec, theta)
        if not (
            validation["endpoint_fixed"]
            and validation["bounded"]
            and validation["monotone"]
        ):
            raise RuntimeError(f"{config.key} produced an invalid shell: {validation}")
        return FittedShell(config.spec, config.regularization, theta, total, data, penalty)

    best = None
    for start in _initial_parameters(config.spec):
        result = minimize(
            lambda theta: pieces(np.asarray(theta, dtype=float))[0],
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-12},
        )
        if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
            best = result
    if best is None or not best.success:
        message = "no finite result" if best is None else str(best.message)
        raise RuntimeError(f"{config.key} optimization failed: {message}")
    theta = np.asarray(best.x, dtype=float)
    total, data, penalty = pieces(theta)
    validation = validate_shell(config.spec, theta)
    if not (validation["endpoint_fixed"] and validation["bounded"] and validation["monotone"]):
        raise RuntimeError(f"{config.key} produced an invalid shell: {validation}")
    return FittedShell(config.spec, config.regularization, theta, total, data, penalty)


def candidate_configs(regularization_grid: Iterable[float]) -> list[ShellConfig]:
    grid = sorted({float(value) for value in regularization_grid})
    if not grid or min(grid) < 0.0:
        raise ValueError("regularization grid must contain non-negative values")
    configs: list[ShellConfig] = []
    for spec in shell_specs():
        if spec.kind == "identity":
            configs.append(ShellConfig(spec, 0.0))
        else:
            configs.extend(ShellConfig(spec, value) for value in grid)
    return configs


__all__ = [
    "Array",
    "FittedShell",
    "ShellConfig",
    "ShellSpec",
    "apparent_seconds",
    "balanced_loss",
    "bernstein_basis",
    "candidate_configs",
    "decode_parameters",
    "fit_shell",
    "identity_deviation",
    "predict_records",
    "shell_specs",
    "signed_exponential",
    "transform",
    "validate_shell",
]
