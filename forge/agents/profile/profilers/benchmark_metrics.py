"""Benchmark Metrics Collector - Pass-through for benchmark data.

This profiler simply extracts metrics from the benchmark result
that's already embedded in the profiling context.
"""

from typing import Any, Dict, List

from .base import BaseProfiler, ProfilingContext, RawProfilerOutput


class BenchmarkMetricsCollector(BaseProfiler):
    """Collects benchmark metrics from the benchmark result.
    
    This is a pass-through profiler - the metrics are already available
    in the profiling context from the benchmark agent. We just extract
    and format them.
    """
    
    DATA_TYPE = "benchmark_metrics"
    DEPENDENCIES = []  # No dependencies - data already in context
    ESTIMATED_DURATION_SECONDS = 1
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Return benchmark metrics from context.
        
        The benchmark metrics are already in the context from Agent 1.
        We just package them for extraction.
        """
        # Get saturation rate from context
        saturation_rate = context.saturation_rate
        
        metadata = {
            "saturation_rate": saturation_rate,
            "model_name": context.model_name,
        }
        
        return RawProfilerOutput(
            report_path=None,
            stdout="",
            stderr="",
            metadata=metadata
        )
    
    async def extract(
        self,
        raw_output: RawProfilerOutput,
        extractors: List[str]
    ) -> Dict[str, Any]:
        """Extract benchmark metrics.
        
        Available extractors:
        - ttft_p50, ttft_p95, ttft_p99: Time to first token percentiles
        - tpot_p50, tpot_p95, tpot_p99: Time per output token percentiles
        - throughput_rps: Requests per second
        - output_tokens_per_sec: Output token throughput
        - total_tokens_per_sec: Total token throughput
        - duration_seconds: Benchmark duration
        - total_requests: Total requests processed
        - failed_requests: Number of failed requests
        """
        results = {}
        
        # Extract from context metadata
        metadata = raw_output.metadata or {}
        
        for extractor in extractors:
            if extractor in metadata:
                results[extractor] = metadata[extractor]
            elif extractor == "saturation_rate_rps":
                results[extractor] = metadata.get("saturation_rate")
            else:
                # For benchmark metrics, we need the actual benchmark result
                # This would be passed through context in real implementation
                results[extractor] = None
        
        return results
    
    def is_available(self) -> bool:
        """Always available - just reads from context."""
        return True
