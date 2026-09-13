"""Numerical validation independent of runners and test frameworks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    num_outside_tolerance: int
    nan_count: int
    inf_count: int


def compare_outputs(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> ComparisonResult:
    """Require abs(target-reference) <= atol + rtol * abs(reference).

    Relative error is measured against abs(reference); at zero it is zero
    for an exact match and infinity otherwise. It does not determine passing
    independently of the combined tolerance. NaN/Inf counts include both
    arrays. Any nonfinite pair fails and contributes infinite error metrics.
    """
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be finite and nonnegative.")
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if reference.shape != target.shape or reference.size == 0:
        raise ValueError("Outputs must have the same nonempty shape.")

    finite = np.isfinite(reference) & np.isfinite(target)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        absolute = np.where(finite, np.abs(target - reference), np.inf)
        relative = np.divide(
            absolute,
            np.abs(reference),
            out=np.where(absolute == 0, 0.0, np.inf),
            where=finite & (reference != 0),
        )
        within = finite & (absolute <= atol + rtol * np.abs(reference))
    outside = int(np.count_nonzero(~within))
    return ComparisonResult(
        passed=outside == 0,
        max_absolute_error=float(absolute.max()),
        mean_absolute_error=float(absolute.mean()),
        max_relative_error=float(relative.max()),
        num_outside_tolerance=outside,
        nan_count=int(np.isnan(reference).sum() + np.isnan(target).sum()),
        inf_count=int(np.isinf(reference).sum() + np.isinf(target).sum()),
    )
