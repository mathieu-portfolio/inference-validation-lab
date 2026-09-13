from pathlib import Path

import onnx
import pytest

from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import MLP
from inference_validation.onnx_export import export_onnx, inspect_graph


@pytest.fixture(scope="module")
def exported_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    set_seed()
    path = tmp_path_factory.mktemp("onnx") / "mlp.onnx"
    return export_onnx(MLP(), generate_input(batch_size=3), path)


def test_mlp_exports_and_passes_validation(exported_model: Path) -> None:
    assert exported_model.is_file()
    model = onnx.load(str(exported_model))
    onnx.checker.check_model(model, full_check=True)
    assert len(model.graph.initializer) == 4
    assert all(not tensor.external_data for tensor in model.graph.initializer)


def test_mlp_essential_operators(exported_model: Path) -> None:
    operators = [node.op_type for node in onnx.load(str(exported_model)).graph.node]
    # Linear layers may be represented as Gemm or as MatMul followed by Add.
    relu_index = operators.index("Relu")
    for layer_ops in (operators[:relu_index], operators[relu_index + 1 :]):
        assert "Gemm" in layer_ops or ("MatMul" in layer_ops and "Add" in layer_ops)


def test_named_tensor_dimensions_and_dtypes(exported_model: Path) -> None:
    graph = onnx.load(str(exported_model)).graph
    assert len(graph.input) == len(graph.output) == 1
    for tensor, name, width in (
        (graph.input[0], "input", 16),
        (graph.output[0], "output", 4),
    ):
        assert tensor.name == name
        tensor_type = tensor.type.tensor_type
        assert tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert len(tensor_type.shape.dim) == 2
        batch, features = tensor_type.shape.dim
        assert batch.dim_param == "batch"
        assert not batch.HasField("dim_value")
        assert features.dim_value == width


def test_graph_inspection(exported_model: Path) -> None:
    report = inspect_graph(exported_model)
    assert "Inputs:\n  input: [batch, 16] FLOAT" in report
    assert "Outputs:\n  output: [batch, 4] FLOAT" in report
    operators = [node.op_type for node in onnx.load(str(exported_model)).graph.node]
    assert report.splitlines()[-1] == "Operators: " + " -> ".join(operators)


def test_export_rejects_single_sample_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch size at least 2"):
        export_onnx(MLP(), generate_input(batch_size=1), tmp_path / "mlp.onnx")
