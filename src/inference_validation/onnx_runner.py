"""CPU execution of single-input, single-output ONNX models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


class ONNXRunner:
    def __init__(self, model_path: str | Path) -> None:
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected a single-input, single-output ONNX model.")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def run(self, inputs: torch.Tensor) -> np.ndarray:
        return self.session.run(
            [self.output_name], {self.input_name: inputs.detach().cpu().numpy()}
        )[0]
