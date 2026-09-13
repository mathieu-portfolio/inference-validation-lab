import pytest

from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import CNN, MLP
from inference_validation.onnx_export import export_onnx


@pytest.fixture(scope="session", params=[MLP, CNN], ids=["mlp", "cnn"])
def model_type(request):
    return request.param


@pytest.fixture(scope="session")
def model(model_type):
    set_seed(42)
    return model_type()


@pytest.fixture(scope="session")
def exported_model(tmp_path_factory, model):
    path = tmp_path_factory.mktemp(type(model).__name__) / "model.onnx"
    inputs = generate_input(3, seed=42, input_shape=model.input_shape)
    return export_onnx(model, inputs, path)
