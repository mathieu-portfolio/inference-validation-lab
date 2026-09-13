"""Models shared by inference runners."""

import torch
from torch import nn


class MLP(nn.Module):
    """A 16 -> 32 -> 4 multilayer perceptron."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(16, 32, device="cpu"),
            nn.ReLU(),
            nn.Linear(32, 4, device="cpu"),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
