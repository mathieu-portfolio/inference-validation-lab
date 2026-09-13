"""Explicit debug exports and first-divergence localization."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from inference_validation.comparison import ComparisonResult, compare_outputs
from inference_validation.models import DebugMLP
from inference_validation.onnx_export import _export_onnx


class _ActivationOutputs(nn.Module):
    def __init__(self, model: DebugMLP) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.model.activations(inputs).values())


def export_debug_onnx(model: DebugMLP, inputs: torch.Tensor, output_path: str | Path) -> Path:
    """Expose named stage outputs without changing normal model/export behavior."""
    return _export_onnx(
        _ActivationOutputs(model), inputs, output_path,
        output_names=[name for name, _ in model.layers.named_children()],
    )


def pytorch_activations(model: DebugMLP, inputs: torch.Tensor) -> dict[str, np.ndarray]:
    model.cpu().eval()
    with torch.no_grad():
        return {name: value.numpy() for name, value in model.activations(inputs.cpu()).items()}


class ONNXDebugRunner:
    """Run an explicitly exported multi-output debug graph on CPU."""

    def __init__(self, model_path: str | Path) -> None:
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected one debug model input.")
        self.input_name = inputs[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run(self, inputs: torch.Tensor) -> dict[str, np.ndarray]:
        outputs = self.session.run(
            self.output_names, {self.input_name: inputs.detach().cpu().numpy()}
        )
        return dict(zip(self.output_names, outputs))


@dataclass(frozen=True)
class LocalizationResult:
    passed: bool
    first_divergent_stage: str | None
    comparison: ComparisonResult | None


def localize_divergence(
    reference: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> LocalizationResult:
    """Compare in reference insertion (execution) order, regardless of target order.

    Both mappings must contain the same nonempty set of named stages.
    Successful results have no failing stage or comparison metrics.
    """
    if not reference or reference.keys() != target.keys():
        raise ValueError("Reference and target must contain the same nonempty set of stages.")
    for name, expected in reference.items():
        comparison = compare_outputs(expected, target[name], atol=atol, rtol=rtol)
        if not comparison.passed:
            return LocalizationResult(False, name, comparison)
    return LocalizationResult(True, None, None)


def print_localization(result: LocalizationResult) -> None:
    """Print a compact localization result for interactive debugging or scripts."""
    if result.passed:
        print("Validation passed")
        return
    metrics = result.comparison
    print(
        f"Validation failed\nFirst divergence: {result.first_divergent_stage}\n"
        f"max abs error: {metrics.max_absolute_error:.6g}\n"
        f"mean abs error: {metrics.mean_absolute_error:.6g}\n"
        f"outside tolerance: {metrics.num_outside_tolerance}"
    )
