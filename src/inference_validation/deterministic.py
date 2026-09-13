"""Seeded CPU model initialization and input generation."""

import torch

from inference_validation.models import MLP


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch before constructing a model."""
    torch.manual_seed(seed)


def generate_input(
    batch_size: int, seed: int = 0, *, input_shape: tuple[int, ...] = MLP.input_shape
) -> torch.Tensor:
    """Generate seeded CPU inputs; input_shape excludes the batch dimension."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, *input_shape, generator=generator, device="cpu")
