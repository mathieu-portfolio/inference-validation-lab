import numpy as np
import pytest

from inference_validation.comparison import compare_outputs
from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import MLP
from inference_validation.onnx_export import export_onnx
from inference_validation.onnx_runner import ONNXRunner
from inference_validation.pytorch_runner import PyTorchRunner


@pytest.fixture(scope="module")
def runners(tmp_path_factory: pytest.TempPathFactory):
    set_seed(42)
    model = MLP()
    path = tmp_path_factory.mktemp("runtime") / "mlp.onnx"
    export_onnx(model, generate_input(3, seed=42), path)
    return PyTorchRunner(model), ONNXRunner(path)


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
def test_runtime_matches_pytorch(runners, batch_size: int) -> None:
    pytorch, onnx = runners
    inputs = generate_input(batch_size, seed=42).requires_grad_()
    reference, target = pytorch.run(inputs), onnx.run(inputs)

    assert isinstance(target, np.ndarray)
    assert target.shape == (batch_size, 4)
    assert onnx.session.get_providers() == ["CPUExecutionProvider"]
    result = compare_outputs(reference, target, atol=1e-6, rtol=1e-5)
    assert result.passed, result
    assert result.num_outside_tolerance == result.nan_count == result.inf_count == 0
