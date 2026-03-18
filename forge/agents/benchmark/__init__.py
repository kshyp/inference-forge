"""Benchmark Runner Agent (Agent 1).

Runs throughput benchmarks using GuideLLM at fixed rate (512 RPS).
"""

from .agent import BenchmarkAgent

__all__ = ["BenchmarkAgent"]
