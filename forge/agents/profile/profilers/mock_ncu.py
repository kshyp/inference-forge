"""Mock NCU Profiler - Returns synthetic data for testing.

This profiler simulates NVIDIA Compute Profiler output without requiring:
- Real GPU
- NCU installation
- Root permissions

Useful for:
- Development and testing
- CI/CD pipelines
- Documentation examples
"""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseProfiler,
    ProfilingContext,
    RawProfilerOutput,
)


class MockNCUProfiler(BaseProfiler):
    """Mock NCU profiler that returns synthetic GPU metrics.
    
    Generates realistic-looking performance data for testing the profiler
    pipeline without needing actual GPU hardware or NCU installation.
    
    The synthetic data is deterministic given a seed (experiment_id)
    to ensure reproducible tests.
    """
    
    DATA_TYPE = "ncu_report"
    DEPENDENCIES = ["vllm_running"]  # Pretend we need vLLM
    ESTIMATED_DURATION_SECONDS = 10  # Fast for testing
    
    # Available extractors and their typical ranges
    EXTRACTOR_RANGES = {
        # Memory metrics
        "memory_bandwidth_utilization_percent": (60.0, 95.0),
        "memory_throughput_gbps": (500.0, 900.0),
        "dram_bytes_read_gb": (10.0, 100.0),
        "dram_bytes_written_gb": (5.0, 50.0),
        
        # Compute metrics
        "compute_utilization_percent": (30.0, 85.0),
        "sm_utilization_percent": (40.0, 90.0),
        "tensor_utilization_percent": (20.0, 70.0),
        
        # Kernel metrics
        "kernel_execution_time_ms": (5.0, 50.0),
        "top_kernel_time_percent": (30.0, 60.0),
        "kernel_count": (50, 200),
        
        # GPU metrics
        "gpu_active_percent": (70.0, 98.0),
        "warp_cstall_percent": (5.0, 25.0),
        "occupancy_percent": (60.0, 95.0),
        
        # Specific kernel names (these return time percentages)
        "flash_attention_time_percent": (20.0, 40.0),
        "gemm_time_percent": (25.0, 50.0),
        "layernorm_time_percent": (5.0, 15.0),
        "softmax_time_percent": (3.0, 10.0),
    }
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize mock profiler.
        
        Args:
            seed: Random seed for reproducible results. If None, uses
                  experiment_id from context when available.
        """
        self._seed = seed
        self._rng = random.Random(seed)
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Generate mock NCU report.
        
        Creates a synthetic report file and returns mock output.
        
        Args:
            context: Profiling context
            
        Returns:
            RawProfilerOutput with mock data
        """
        # Use experiment_id as seed if no explicit seed set
        seed = self._seed
        if seed is None and context.experiment_id:
            try:
                # Use hash of experiment_id as seed
                seed = hash(context.experiment_id) % (2**32)
            except Exception:
                seed = 42
        
        self._rng = random.Random(seed)
        
        # Create output directory
        output_dir = context.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate fake report path
        report_path = output_dir / f"{context.experiment_id}_mock.ncu-rep"
        
        # Write a placeholder file
        report_content = self._generate_mock_report_content(context)
        report_path.write_text(report_content)
        
        return RawProfilerOutput(
            report_path=report_path,
            stdout="Mock NCU profiling completed successfully",
            stderr="",
            is_mock=True,
            metadata={
                "seed": seed,
                "duration_seconds": self.ESTIMATED_DURATION_SECONDS,
                "mock": True,
            }
        )
    
    def _generate_mock_report_content(self, context: ProfilingContext) -> str:
        """Generate mock NCU report content.
        
        Args:
            context: Profiling context
            
        Returns:
            Mock report content as string
        """
        lines = [
            "# Mock NCU Report",
            f"# Generated for experiment: {context.experiment_id}",
            f"# Model: {context.model_name}",
            f"# Duration: {context.duration_seconds}s",
            "",
            "## Metrics Summary",
            "",
        ]
        
        # Add some random metrics
        for metric, (min_val, max_val) in list(self.EXTRACTOR_RANGES.items())[:10]:
            value = self._rng.uniform(min_val, max_val)
            lines.append(f"{metric}: {value:.2f}")
        
        lines.extend([
            "",
            "## Top Kernels",
            "",
            "flash_attention_fwd: 35.2%",
            "cutlass_gemm: 28.5%",
            "layernorm_fwd: 8.3%",
            "softmax_fwd: 5.1%",
            "",
            "## Note",
            "This is synthetic data for testing purposes.",
        ])
        
        return "\n".join(lines)
    
    async def extract(
        self,
        raw_output: RawProfilerOutput,
        extractors: List[str]
    ) -> Dict[str, Any]:
        """Extract synthetic metrics.
        
        Args:
            raw_output: Output from run() (contains seed in metadata)
            extractors: List of metrics to extract
            
        Returns:
            Dict with synthetic metric values
        """
        results = {}
        
        # Reinitialize RNG from metadata if available
        metadata = raw_output.metadata or {}
        seed = metadata.get("seed")
        if seed is not None:
            self._rng = random.Random(seed)
        
        for extractor in extractors:
            value = self._generate_metric(extractor)
            results[extractor] = value
        
        return results
    
    def _generate_metric(self, extractor: str) -> Any:
        """Generate a synthetic metric value.
        
        Args:
            extractor: Name of the metric
            
        Returns:
            Synthetic value (float, int, or None if unknown)
        """
        if extractor in self.EXTRACTOR_RANGES:
            min_val, max_val = self.EXTRACTOR_RANGES[extractor]
            
            # Generate value in range
            value = self._rng.uniform(min_val, max_val)
            
            # Return int for count metrics, float for others
            if extractor.endswith("_count"):
                return int(value)
            return round(value, 2)
        
        # Unknown extractor - return random float
        return round(self._rng.uniform(0.0, 100.0), 2)
    
    def is_available(self) -> bool:
        """Mock profiler is always available.
        
        Returns:
            Always True
        """
        return True
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible results.
        
        Args:
            seed: Random seed
        """
        self._seed = seed
        self._rng = random.Random(seed)
