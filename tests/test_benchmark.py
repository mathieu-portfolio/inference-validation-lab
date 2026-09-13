import math

import pytest

from inference_validation.benchmark import benchmark


def test_warmup_is_separate_from_measured_calls(monkeypatch) -> None:
    events = []
    ticks = iter([0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])

    def clock():
        events.append("clock")
        return next(ticks)

    monkeypatch.setattr("inference_validation.benchmark.perf_counter_ns", clock)
    result = benchmark(
        lambda: events.append("inference"),
        batch_size=8,
        warmup_iterations=2,
        measured_iterations=3,
    )

    assert events == ["inference"] * 2 + ["clock", "inference", "clock"] * 3
    assert result.measured_iterations == 3
    assert result.batch_size == 8
    assert result.throughput_samples_per_second == pytest.approx(8000)


@pytest.mark.parametrize(
    "durations_ms, expected",
    [
        ([1, 3, 2, 4], (2.5, 2.5, 1, 4, math.sqrt(1.25), 3200)),
        ([2], (2, 2, 2, 2, 0, 4000)),
    ],
)
def test_statistics_from_controlled_timings(monkeypatch, durations_ms, expected) -> None:
    ticks = iter(
        tick
        for index, duration in enumerate(durations_ms)
        for tick in (index * 10_000_000, index * 10_000_000 + duration * 1_000_000)
    )
    monkeypatch.setattr("inference_validation.benchmark.perf_counter_ns", lambda: next(ticks))
    result = benchmark(
        lambda: None, batch_size=8, warmup_iterations=0, measured_iterations=len(durations_ms)
    )

    median, mean, minimum, maximum, stddev, throughput = expected
    assert result.median_latency_seconds == pytest.approx(median / 1000)
    assert result.mean_latency_seconds == pytest.approx(mean / 1000)
    assert result.min_latency_seconds == pytest.approx(minimum / 1000)
    assert result.max_latency_seconds == pytest.approx(maximum / 1000)
    assert result.stddev_latency_seconds == pytest.approx(stddev / 1000)
    assert result.throughput_samples_per_second == pytest.approx(throughput)
