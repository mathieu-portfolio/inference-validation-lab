"""Seeded CPU model initialization and input generation."""

import torch


def set_seed(seed: int = 0) -> None:
    """Seed PyTorch before constructing a model."""
    torch.manual_seed(seed)


def generate_input(batch_size: int, seed: int = 0) -> torch.Tensor:
    """Generate MLP inputs without changing the global random state."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch_size, 16, generator=generator, device="cpu")
