"""Run with: python scripts/benchmark.py (after installing the package)."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from inference_validation.benchmark import benchmark
from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import CNN, MLP
from inference_validation.onnx_export import export_onnx
from inference_validation.onnx_runner import ONNXRunner
from inference_validation.pytorch_runner import PyTorchRunner


def main() -> None:
    warmup_iterations, measured_iterations = 5, 30
    results = []
    with TemporaryDirectory(prefix="inference-benchmark-") as directory:
        for model_type in (MLP, CNN):
            set_seed(42)
            model = model_type()
            example = generate_input(3, seed=42, input_shape=model.input_shape)
            path = export_onnx(model, example, Path(directory) / f"{model_type.__name__}.onnx")
            runners = (("PyTorch", PyTorchRunner(model)), ("ONNX Runtime", ONNXRunner(path)))
            for batch_size in (1, 8, 32):
                inputs = generate_input(batch_size, seed=42, input_shape=model.input_shape)
                for backend, runner in runners:
                    result = benchmark(
                        partial(runner.run, inputs),
                        batch_size=batch_size,
                        warmup_iterations=warmup_iterations,
                        measured_iterations=measured_iterations,
                    )
                    results.append((model_type.__name__, backend, result))

    print(f"CPU inference | warm-up: {warmup_iterations} | measured: {measured_iterations}")
    print("Latency in ms, including runner input/output conversion; backend default thread settings.")
    print(f"{'Model':<5} {'Backend':<12} {'Batch':>5} {'Median':>10} {'Mean':>10} "
          f"{'Min':>10} {'Max':>10} {'Stddev':>10} {'Samples/s':>12}")
    for name, backend, result in results:
        print(
            f"{name:<5} {backend:<12} {result.batch_size:>5} "
            f"{result.median_latency_seconds * 1000:>10.4f} "
            f"{result.mean_latency_seconds * 1000:>10.4f} "
            f"{result.min_latency_seconds * 1000:>10.4f} "
            f"{result.max_latency_seconds * 1000:>10.4f} "
            f"{result.stddev_latency_seconds * 1000:>10.4f} "
            f"{result.throughput_samples_per_second:>12.1f}"
        )


if __name__ == "__main__":
    main()
