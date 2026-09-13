import numpy as np
import pytest

from inference_validation.comparison import compare_outputs
from inference_validation.deterministic import generate_input
from inference_validation.onnx_runner import ONNXRunner
from inference_validation.pytorch_runner import PyTorchRunner


@pytest.fixture(scope="module")
def runners(model, exported_model):
    return PyTorchRunner(model), ONNXRunner(exported_model)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_runtime_matches_pytorch(runners, model, batch_size: int) -> None:
    pytorch, onnx = runners
    inputs = generate_input(
        batch_size, seed=42, input_shape=model.input_shape
    ).requires_grad_()
    reference, target = pytorch.run(inputs), onnx.run(inputs)

    assert isinstance(target, np.ndarray)
    assert target.shape == (batch_size, model.output_size)
    assert onnx.session.get_providers() == ["CPUExecutionProvider"]
    result = compare_outputs(reference, target, atol=1e-6, rtol=1e-5)
    assert result.passed, result
    assert result.num_outside_tolerance == result.nan_count == result.inf_count == 0
