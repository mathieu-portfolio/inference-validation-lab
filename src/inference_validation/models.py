"""Models shared by inference runners."""

import torch
from torch import nn


class MLP(nn.Module):
    """A 16 -> 32 -> 4 multilayer perceptron."""

    input_shape = (16,)
    output_size = 4

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_shape[0], 32, device="cpu"),
            nn.ReLU(),
            nn.Linear(32, self.output_size, device="cpu"),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CNN(nn.Module):
    """A compact classifier for single-channel 8x8 images."""

    input_shape = (1, 8, 8)
    output_size = 4

    def __init__(self) -> None:
        super().__init__()
        channels, height, width = self.input_shape
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=3, padding=1, device="cpu"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(start_dim=1),
            nn.Linear(4 * (height // 2) * (width // 2), self.output_size, device="cpu"),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
