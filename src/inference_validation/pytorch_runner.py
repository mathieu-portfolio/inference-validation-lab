"""CPU inference using PyTorch."""

import numpy as np
import torch
from torch import nn


class PyTorchRunner:
    def __init__(self, model: nn.Module) -> None:
        self.model = model.cpu()

    def run(self, inputs: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.cpu()).numpy()
