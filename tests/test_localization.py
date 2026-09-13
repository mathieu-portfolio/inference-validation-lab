import numpy as np
import onnx
import pytest

from inference_validation.comparison import compare_outputs
from inference_validation.deterministic import generate_input, set_seed
from inference_validation.localization import (
    ONNXDebugRunner,
    export_debug_onnx,
    localize_divergence,
    print_localization,
    pytorch_activations,
)
from inference_validation.models import DebugMLP
from inference_validation.onnx_export import export_onnx
from inference_validation.onnx_runner import ONNXRunner
from inference_validation.pytorch_runner import PyTorchRunner


@pytest.fixture(scope="module")
def debug_pipeline(tmp_path_factory):
    set_seed(42)
    model = DebugMLP()
    inputs = generate_input(3, seed=42, input_shape=model.input_shape)
    path = export_debug_onnx(model, inputs, tmp_path_factory.mktemp("debug") / "debug.onnx")
    onnx.checker.check_model(str(path), full_check=True)
    return model, ONNXDebugRunner(path)


def _perturb_stage(target, stage):
    """Test-only fault in a captured target activation; never mutate the runner."""
    faulty = {name: value.copy() for name, value in target.items()}
    faulty[stage] += 1.0
    return faulty


@pytest.mark.parametrize("batch_size", [1, 8])
def test_all_intermediates_match(debug_pipeline, batch_size):
    model, runner = debug_pipeline
    inputs = generate_input(batch_size, seed=42)
    reference, target = pytorch_activations(model, inputs), runner.run(inputs)
    assert list(reference) == list(target) == [
        "linear_1", "relu_1", "linear_2", "relu_2", "linear_3"
    ]
    result = localize_divergence(reference, target)
    assert result.passed
    assert result.first_divergent_stage is None
    assert result.comparison is None
    np.testing.assert_array_equal(reference["linear_3"], PyTorchRunner(model).run(inputs))


def test_known_fault_reports_stage_and_metrics(debug_pipeline, capsys):
    model, runner = debug_pipeline
    inputs = generate_input(8, seed=42)
    reference = pytorch_activations(model, inputs)
    target = _perturb_stage(runner.run(inputs), "linear_2")
    result = localize_divergence(reference, target, atol=1e-6, rtol=1e-5)
    assert not result.passed
    assert result.first_divergent_stage == "linear_2"
    assert result.comparison == compare_outputs(reference["linear_2"], target["linear_2"])
    assert result.comparison.num_outside_tolerance == 8 * 32
    print_localization(result)
    report = capsys.readouterr().out
    assert "Validation failed\nFirst divergence: linear_2" in report
    assert "max abs error:" in report
    assert "mean abs error:" in report
    assert "outside tolerance: 256" in report
    assert localize_divergence(reference, target, atol=2, rtol=0).passed


def test_later_fault_cannot_hide_first_even_with_reordered_target(debug_pipeline):
    model, runner = debug_pipeline
    inputs = generate_input(2, seed=42)
    reference = pytorch_activations(model, inputs)
    target = _perturb_stage(runner.run(inputs), "linear_2")
    target = _perturb_stage(target, "linear_3")
    assert not compare_outputs(reference["linear_3"], target["linear_3"]).passed
    result = localize_divergence(reference, dict(reversed(list(target.items()))))
    assert result.first_divergent_stage == "linear_2"


def test_missing_stage_is_rejected():
    with pytest.raises(ValueError, match="stages"):
        localize_divergence({"linear_1": np.zeros(1)}, {})


def test_normal_final_output_path_is_unchanged(debug_pipeline, tmp_path):
    model, _ = debug_pipeline
    inputs = generate_input(3, seed=42)
    path = export_onnx(model, inputs, tmp_path / "normal.onnx")
    assert [output.name for output in onnx.load(str(path)).graph.output] == ["output"]
    reference = PyTorchRunner(model).run(inputs)
    target = ONNXRunner(path).run(inputs)
    assert compare_outputs(reference, target).passed
