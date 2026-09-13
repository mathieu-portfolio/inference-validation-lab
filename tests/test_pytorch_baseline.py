import numpy as np
import torch

from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import MLP
from inference_validation.pytorch_runner import PyTorchRunner


def test_mlp_output_shape() -> None:
    set_seed()
    model = MLP()
    inputs = generate_input(batch_size=3)

    assert model(inputs).shape == (3, 4)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_repeated_inference_is_identical() -> None:
    set_seed()
    model = MLP()
    runner = PyTorchRunner(model)
    inputs = generate_input(batch_size=3).requires_grad_()
    grad_modes = []
    hook = model.register_forward_pre_hook(
        lambda module, args: grad_modes.append(torch.is_grad_enabled())
    )
    try:
        first = runner.run(inputs)
        second = runner.run(inputs)
    finally:
        hook.remove()

    assert isinstance(first, np.ndarray)
    assert first.shape == (3, 4)
    np.testing.assert_array_equal(first, second)
    assert not model.training
    assert grad_modes == [False, False]


def test_same_seed_reproduces_model_and_input() -> None:
    set_seed(42)
    first_model = MLP()
    first_input = generate_input(batch_size=3, seed=42)
    first = PyTorchRunner(first_model).run(first_input)

    # Disturb global RNG state before reconstructing the baseline.
    torch.randn(100)
    set_seed(42)
    second_model = MLP()
    second_input = generate_input(batch_size=3, seed=42)
    second = PyTorchRunner(second_model).run(second_input)

    np.testing.assert_array_equal(first, second)
