"""Run with: python scripts/benchmark.py (after installing the package)."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse

from inference_validation.benchmark import benchmark
from inference_validation.deterministic import generate_input, set_seed
from inference_validation.models import CNN, MLP
from inference_validation.onnx_export import export_onnx
from inference_validation.onnx_runner import ONNXRunner
from inference_validation.pytorch_runner import PyTorchRunner


def run_benchmarks(warmup_iterations: int = 5, measured_iterations: int = 30):
    """Collect one run for the table and optional plots using the same timings."""
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
    return results


def print_table(results, warmup_iterations: int, measured_iterations: int) -> None:
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


def save_plots(results, output_directory: Path, warmup_iterations: int, measured_iterations: int) -> None:
    """Render actual measurements after all timed inference has finished."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    series = [
        ("MLP", "PyTorch", "#0072B2", "o", "-"),
        ("MLP", "ONNX Runtime", "#0072B2", "s", "--"),
        ("CNN", "PyTorch", "#D55E00", "o", "-"),
        ("CNN", "ONNX Runtime", "#D55E00", "s", "--"),
    ]
    plots = [
        ("median_latency.png", "Median inference latency", "Latency (ms)",
         lambda result: result.median_latency_seconds * 1000),
        ("throughput.png", "Inference throughput", "Throughput (samples/second)",
         lambda result: result.throughput_samples_per_second),
    ]
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        for filename, title, ylabel, metric in plots:
            fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
            for model, backend, color, marker, linestyle in series:
                values = sorted(
                    (result for name, engine, result in results if (name, engine) == (model, backend)),
                    key=lambda result: result.batch_size,
                )
                ax.plot(
                    [result.batch_size for result in values], [metric(result) for result in values],
                    label=f"{model} / {backend}", color=color, marker=marker,
                    linestyle=linestyle, linewidth=2, markersize=7,
                )
            ax.set(title=title, xlabel="Batch size", ylabel=ylabel, xticks=[1, 8, 32])
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, frameon=False)
            fig.suptitle(
                f"Example local CPU run | {warmup_iterations} warm-up + {measured_iterations} measured calls",
                fontsize=12,
            )
            fig.savefig(output_directory / filename, dpi=160)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true", help="Also regenerate docs/images benchmark plots.")
    args = parser.parse_args()
    warmup_iterations, measured_iterations = 5, 30
    results = run_benchmarks(warmup_iterations, measured_iterations)
    print_table(results, warmup_iterations, measured_iterations)
    if args.plots:
        output_directory = Path(__file__).resolve().parents[1] / "docs" / "images"
        save_plots(results, output_directory, warmup_iterations, measured_iterations)
        print(f"Plots saved to {output_directory}")


if __name__ == "__main__":
    main()
