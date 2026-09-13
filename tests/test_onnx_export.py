from pathlib import Path

import onnx
import pytest

from inference_validation.deterministic import generate_input
from inference_validation.models import CNN
from inference_validation.onnx_export import export_onnx, inspect_graph


def test_exports_and_passes_validation(exported_model: Path, model) -> None:
    assert exported_model.is_file()
    graph_model = onnx.load(str(exported_model))
    onnx.checker.check_model(graph_model, full_check=True)
    parameter_names = {name for name, _ in model.named_parameters()}
    assert parameter_names <= {tensor.name for tensor in graph_model.graph.initializer}
    assert all(not tensor.external_data for tensor in graph_model.graph.initializer)


def test_essential_operators(exported_model: Path, model_type) -> None:
    operators = [node.op_type for node in onnx.load(str(exported_model)).graph.node]
    # Linear layers may be represented as Gemm or as MatMul followed by Add.
    relu_index = operators.index("Relu")
    if model_type is CNN:
        assert "Conv" in operators[:relu_index]
        pool_index = operators.index("MaxPool")
        assert pool_index > relu_index
        assert "Flatten" in operators or "Reshape" in operators
        linear_layers = (operators[pool_index + 1 :],)
    else:
        linear_layers = (operators[:relu_index], operators[relu_index + 1 :])
    for layer_ops in linear_layers:
        assert "Gemm" in layer_ops or ("MatMul" in layer_ops and "Add" in layer_ops)


def test_named_tensor_dimensions_and_dtypes(exported_model: Path, model_type) -> None:
    graph = onnx.load(str(exported_model)).graph
    assert len(graph.input) == len(graph.output) == 1
    for tensor, name, shape in (
        (graph.input[0], "input", model_type.input_shape),
        (graph.output[0], "output", (model_type.output_size,)),
    ):
        assert tensor.name == name
        tensor_type = tensor.type.tensor_type
        assert tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert len(tensor_type.shape.dim) == len(shape) + 1
        batch, *features = tensor_type.shape.dim
        assert batch.dim_param == "batch"
        assert not batch.HasField("dim_value")
        assert tuple(dim.dim_value for dim in features) == shape


def test_graph_inspection(exported_model: Path, model_type) -> None:
    report = inspect_graph(exported_model)
    dimensions = ", ".join(str(dim) for dim in model_type.input_shape)
    assert f"Inputs:\n  input: [batch, {dimensions}] FLOAT" in report
    assert f"Outputs:\n  output: [batch, {model_type.output_size}] FLOAT" in report
    operators = [node.op_type for node in onnx.load(str(exported_model)).graph.node]
    assert report.splitlines()[-1] == "Operators: " + " -> ".join(operators)


def test_export_rejects_single_sample_batch(tmp_path: Path, model_type) -> None:
    with pytest.raises(ValueError, match="batch size at least 2"):
        export_onnx(
            model_type(),
            generate_input(batch_size=1, input_shape=model_type.input_shape),
            tmp_path / "model.onnx",
        )
