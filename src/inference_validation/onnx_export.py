"""ONNX export and readable graph inspection for single-input models."""

from pathlib import Path

import onnx
import torch
from torch import nn


def export_onnx(
    model: nn.Module, inputs: torch.Tensor, output_path: str | Path
) -> Path:
    """Export in CPU eval mode, with a dynamic batch axis, then validate.

    Use a representative batch of at least two samples so the exporter can
    trace a dynamic batch dimension. The model is moved to CPU and eval mode.
    """
    return _export_onnx(model, inputs, output_path, output_names=["output"])


def _export_onnx(
    model: nn.Module,
    inputs: torch.Tensor,
    output_path: str | Path,
    *,
    output_names: list[str],
) -> Path:
    """Shared export mechanics for normal and explicit debug outputs."""
    if inputs.ndim < 1 or inputs.shape[0] < 2:
        raise ValueError("Use a representative input with batch size at least 2.")
    output_path = Path(output_path)
    model.cpu().eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs.cpu(),),
            output_path,
            input_names=["input"],
            output_names=output_names,
            dynamo=True,
            dynamic_shapes=({0: torch.export.Dim("batch")},),
            external_data=False,
            verbose=False,
        )
    onnx.checker.check_model(str(output_path))
    return output_path


def inspect_graph(model_path: str | Path) -> str:
    """Return tensor metadata and operator types in stored graph order."""
    graph = onnx.load(str(model_path)).graph
    lines = []
    for label, tensors in (("Inputs", graph.input), ("Outputs", graph.output)):
        lines.append(f"{label}:")
        for tensor in tensors:
            tensor_type = tensor.type.tensor_type
            dimensions = [
                str(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
                for dim in tensor_type.shape.dim
            ]
            shape = ", ".join(dimensions) if tensor_type.HasField("shape") else "?"
            dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
            lines.append(f"  {tensor.name}: [{shape}] {dtype}")
    lines.append("Operators: " + " -> ".join(node.op_type for node in graph.node))
    return "\n".join(lines)
