"""Tests for the benchmark runner."""

from __future__ import annotations

from benchmark.runner import MachineInfo, TimingResult, run_benchmark, validate_answers


class TestRunner:
    def test_run_benchmark_returns_timing(self) -> None:
        counter = 0

        def dummy() -> int:
            nonlocal counter
            counter += 1
            return counter

        result = run_benchmark(
            "test_op", "test_engine", "test_ds", dummy, 100, rounds=3, warmup=1
        )
        assert isinstance(result, TimingResult)
        assert result.median_seconds > 0
        assert len(result.times_seconds) == 3
        assert result.operation == "test_op"
        assert result.engine == "test_engine"

    def test_machine_info_populated(self) -> None:
        info = MachineInfo()
        assert info.system != ""
        assert info.cpu_count > 0
        assert info.ram_gb > 0

    def test_validate_answers_same_length(self) -> None:
        assert validate_answers({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_validate_answers_different_length(self) -> None:
        assert not validate_answers({"a": [1, 2], "b": [4, 5, 6]})
