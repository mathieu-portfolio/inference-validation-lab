"""Wall-clock measurements of synchronous inference callables."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median, pstdev
from time import perf_counter_ns


@dataclass(frozen=True)
class BenchmarkResult:
    median_latency_seconds: float
    mean_latency_seconds: float
    min_latency_seconds: float
    max_latency_seconds: float
    stddev_latency_seconds: float
    throughput_samples_per_second: float
    batch_size: int
    measured_iterations: int


def benchmark(
    inference: Callable[[], object],
    *,
    batch_size: int,
    warmup_iterations: int = 10,
    measured_iterations: int = 100,
) -> BenchmarkResult:
    """Time completed calls, excluding warm-up and caller-owned setup.

    Latencies are seconds; standard deviation is the population statistic.
    Throughput is total measured samples divided by total measured call time.
    The callable must finish inference before returning (as CPU runners do).
    """
    if batch_size < 1 or measured_iterations < 1 or warmup_iterations < 0:
        raise ValueError("Batch size and measured iterations must be positive; warm-up nonnegative.")

    for _ in range(warmup_iterations):
        inference()

    latencies = []
    for _ in range(measured_iterations):
        start = perf_counter_ns()
        output = inference()
        elapsed = perf_counter_ns() - start
        latencies.append(elapsed / 1_000_000_000)
        del output

    total = sum(latencies)
    return BenchmarkResult(
        median_latency_seconds=median(latencies),
        mean_latency_seconds=mean(latencies),
        min_latency_seconds=min(latencies),
        max_latency_seconds=max(latencies),
        stddev_latency_seconds=pstdev(latencies),
        throughput_samples_per_second=(
            batch_size * measured_iterations / total if total else float("inf")
        ),
        batch_size=batch_size,
        measured_iterations=measured_iterations,
    )
