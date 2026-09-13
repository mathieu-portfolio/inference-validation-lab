import numpy as np
import pytest

from inference_validation.comparison import compare_outputs


def test_small_perturbation_passes_including_near_zero() -> None:
    reference = np.array([0.0, 1e-12, 1.0, 100.0])
    target = reference + np.array([5e-7, 5e-7, 5e-6, 5e-4])
    result = compare_outputs(reference, target, atol=1e-6, rtol=1e-5)
    assert result.passed
    assert result.num_outside_tolerance == 0
    assert result.max_relative_error == np.inf


def test_excessive_perturbation_fails_and_reports_metrics() -> None:
    result = compare_outputs(
        np.array([1.0, 2.0]), np.array([1.0, 2.5]), atol=0.1, rtol=0.01
    )
    assert not result.passed
    assert result.num_outside_tolerance == 1
    assert result.max_absolute_error == 0.5
    assert result.mean_absolute_error == 0.25
    assert result.max_relative_error == 0.25
    assert result.nan_count == result.inf_count == 0


def test_absolute_tolerance_controls_zero_reference() -> None:
    reference, target = np.array([0.0]), np.array([1e-5])
    assert not compare_outputs(reference, target, atol=1e-6, rtol=1.0).passed
    assert compare_outputs(reference, target, atol=1e-5, rtol=0.0).passed
    assert compare_outputs(reference, reference, atol=0, rtol=0).max_relative_error == 0


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_target_fails(value: float) -> None:
    result = compare_outputs(np.array([1.0, 2.0]), np.array([1.0, value]))
    assert not result.passed
    assert result.num_outside_tolerance == 1
    assert result.nan_count == int(np.isnan(value))
    assert result.inf_count == int(np.isinf(value))
    assert result.max_absolute_error == result.mean_absolute_error == np.inf
    assert result.max_relative_error == np.inf


def test_nonfinite_reference_also_fails() -> None:
    result = compare_outputs(np.array([np.inf]), np.array([np.inf]))
    assert not result.passed
    assert result.inf_count == 2


def test_shape_mismatch_cannot_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        compare_outputs(np.zeros((2, 1)), np.zeros((2,)))


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf])
def test_invalid_tolerances_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="Tolerances"):
        compare_outputs(np.zeros(1), np.zeros(1), atol=tolerance)
