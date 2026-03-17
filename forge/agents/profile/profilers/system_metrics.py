"""System Metrics Collector - Always-available GPU/system metrics.

Collects baseline metrics from nvidia-smi that don't require root privileges.
This is the "poor man's profiler" - always works, minimal overhead.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseProfiler, ProfilingContext, ProfilerResult


class SystemMetricsCollector(BaseProfiler):
    """
    Collect system metrics using nvidia-smi.
    
    Always available (doesn't require root like NCU/NSys).
    Provides baseline GPU utilization, memory, temperature.
    
    Metrics collected:
    - gpu_utilization_percent
    - memory_used_mb / memory_total_mb
    - temperature_celsius
    - power_draw_watts
    """
    
    PROFILER_NAME = "system_metrics"
    DATA_TYPE = "system_metrics_report"
    
    def is_available(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def run(self, context: ProfilingContext) -> ProfilerResult:
        """Run nvidia-smi and collect metrics."""
        
        # Query GPU metrics
        query_params = [
            "timestamp",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "power.draw",
            "clocks.current.sm",
            "pcie.link.gen.current",
            "pcie.link.width.current",
        ]
        
        cmd = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(query_params),
            "--format=csv,noheader,nounits"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ProfilerResult(
                    success=False,
                    data_type=self.DATA_TYPE,
                    profiler_name=self.PROFILER_NAME,
                    error=f"nvidia-smi failed: {result.stderr}"
                )
            
            # Parse output
            lines = result.stdout.strip().split("\n")
            metrics_by_gpu = []
            
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 10:
                    metrics_by_gpu.append({
                        "timestamp": parts[0],
                        "gpu_utilization_percent": float(parts[1]) if parts[1] else 0.0,
                        "memory_utilization_percent": float(parts[2]) if parts[2] else 0.0,
                        "memory_used_mb": float(parts[3]) if parts[3] else 0.0,
                        "memory_total_mb": float(parts[4]) if parts[4] else 0.0,
                        "temperature_celsius": float(parts[5]) if parts[5] else 0.0,
                        "power_draw_watts": float(parts[6]) if parts[6] else 0.0,
                        "sm_clock_mhz": float(parts[7]) if parts[7] else 0.0,
                        "pcie_gen": int(parts[8]) if parts[8] else 0,
                        "pcie_width": int(parts[9]) if parts[9] else 0,
                    })
            
            # Aggregate across GPUs (for single GPU, just use [0])
            aggregated = self._aggregate_metrics(metrics_by_gpu)
            
            # Save to file
            output_file = context.output_dir / "system_metrics.json"
            with open(output_file, "w") as f:
                json.dump({
                    "per_gpu": metrics_by_gpu,
                    "aggregated": aggregated,
                    "collection_timestamp": result.stdout.strip().split("\n")[0].split(",")[0] if lines else ""
                }, f, indent=2)
            
            return ProfilerResult(
                success=True,
                data_type=self.DATA_TYPE,
                profiler_name=self.PROFILER_NAME,
                output_path=output_file,
                metrics=aggregated
            )
            
        except Exception as e:
            return ProfilerResult(
                success=False,
                data_type=self.DATA_TYPE,
                profiler_name=self.PROFILER_NAME,
                error=str(e)
            )
    
    def _aggregate_metrics(self, per_gpu: list) -> Dict[str, Any]:
        """Aggregate metrics across multiple GPUs."""
        if not per_gpu:
            return {}
        
        # For single GPU, just return first GPU metrics
        if len(per_gpu) == 1:
            return per_gpu[0]
        
        # For multi-GPU, average/average appropriately
        return {
            "gpu_utilization_percent": sum(g["gpu_utilization_percent"] for g in per_gpu) / len(per_gpu),
            "memory_utilization_percent": sum(g["memory_utilization_percent"] for g in per_gpu) / len(per_gpu),
            "memory_used_mb": sum(g["memory_used_mb"] for g in per_gpu),
            "memory_total_mb": sum(g["memory_total_mb"] for g in per_gpu),
            "temperature_celsius": max(g["temperature_celsius"] for g in per_gpu),  # Max temp
            "power_draw_watts": sum(g["power_draw_watts"] for g in per_gpu),
            "num_gpus": len(per_gpu),
        }
    
    async def extract(self, raw_output: Path) -> Dict[str, Any]:
        """Extract metrics from saved file."""
        with open(raw_output) as f:
            data = json.load(f)
        return data.get("aggregated", {})


class GPUMetricsCollector(BaseProfiler):
    """
    Collect GPU metrics during benchmark run (time-series).
    
    Polls nvidia-smi at intervals during the benchmark to capture
    utilization over time, not just at a single point.
    """
    
    PROFILER_NAME = "gpu_metrics_time_series"
    DATA_TYPE = "gpu_metrics_time_series"
    
    def is_available(self) -> bool:
        return SystemMetricsCollector().is_available()
    
    async def run(self, context: ProfilingContext) -> ProfilerResult:
        """
        Run during benchmark and collect time-series metrics.
        
        Note: This is designed to run concurrently with the benchmark.
        The benchmark runner should start this, then the main benchmark,
        then stop this when benchmark completes.
        """
        # This is a placeholder - actual implementation would need
        # to be integrated with the benchmark runner for continuous collection
        
        # For now, just collect single-point metrics
        collector = SystemMetricsCollector()
        return await collector.run(context)
