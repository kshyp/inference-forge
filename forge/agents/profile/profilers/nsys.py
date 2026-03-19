"""NVIDIA Systems Profiler (nsys) for GPU timeline and kernel analysis.

This profiler extracts metrics from nsys report files. The actual nsys data
collection is handled by ProfilerAgent which wraps vLLM with nsys launch
and runs nsys start/stop during the benchmark.
"""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseProfiler,
    ExtractorError,
    ProfilerNotAvailableError,
    ProfilingContext,
    RawProfilerOutput,
)


class NSysProfiler(BaseProfiler):
    """NVIDIA Systems Profiler for GPU timeline analysis.
    
    Extracts metrics from nsys report files (.nsys-rep). The actual collection
    is done by ProfilerAgent via _run_nsys_profile_during_benchmark().
    
    This profiler reads the generated report file and extracts:
    - GPU utilization
    - Kernel execution times
    - Memory operations
    - Timeline gaps (idle periods)
    """
    
    DATA_TYPE = "nsys_report"
    DEPENDENCIES = ["vllm_running"]
    ESTIMATED_DURATION_SECONDS = 10  # Just reading file, not running nsys
    
    def is_available(self) -> bool:
        """Check if nsys is available on this system."""
        return shutil.which("nsys") is not None
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Find and return the nsys report file path.
        
        The actual nsys profiling is done by ProfilerAgent during benchmark.
        This method just locates the report file.
        """
        # Look for nsys report files in the output directory
        # The agent creates: {exp_dir}/nsys_profile_steady_state.nsys-rep
        output_dir = context.output_dir
        
        # Possible report file names
        possible_files = [
            output_dir / "nsys_profile_steady_state.nsys-rep",
            output_dir / "nsys_profile.nsys-rep",
        ]
        
        # Also check for any .nsys-rep files in the directory
        if output_dir.exists():
            nsys_files = list(output_dir.glob("*.nsys-rep"))
            possible_files.extend(nsys_files)
        
        # Find the first existing report file
        report_path = None
        for path in possible_files:
            if path.exists():
                report_path = path
                break
        
        if report_path is None:
            # No report file found - nsys may not have been run
            return RawProfilerOutput(
                report_path=None,
                stdout="",
                stderr="No nsys report file found",
                metadata={"error": "nsys_report_not_found"},
            )
        
        # Get file stats
        file_size_mb = report_path.stat().st_size / (1024 * 1024)
        
        return RawProfilerOutput(
            report_path=report_path,
            stdout=f"Found nsys report: {report_path}",
            stderr="",
            metadata={
                "file_size_mb": file_size_mb,
                "report_path": str(report_path),
            },
        )
    
    async def extract(
        self,
        raw_output: RawProfilerOutput,
        extractors: List[str]
    ) -> Dict[str, Any]:
        """Extract metrics from nsys report file.
        
        Uses nsys stats CLI to extract metrics from the report file.
        Falls back to file metadata if nsys stats fails.
        """
        if not raw_output.report_path or not raw_output.report_path.exists():
            # Return default values if no report
            return self._get_default_extractors(extractors)
        
        report_path = raw_output.report_path
        result: Dict[str, Any] = {}
        
        # Try to run nsys stats to get detailed metrics
        try:
            stats_output = await self._run_nsys_stats(report_path)
        except Exception as e:
            stats_output = {"error": str(e)}
        
        # Extract requested metrics
        for extractor in extractors:
            try:
                value = self._extract_metric(extractor, report_path, stats_output, raw_output.metadata)
                result[extractor] = value
            except Exception as e:
                result[extractor] = None
                raise ExtractorError(f"Failed to extract {extractor}: {e}")
        
        # Always include report path
        result["report_path"] = str(report_path)
        
        return result
    
    async def _run_nsys_stats(self, report_path: Path) -> Dict[str, Any]:
        """Run nsys stats to extract metrics from report file."""
        cmd = [
            "nsys", "stats",
            "--report", "cuda_gpu_trace,cuda_api_sum,gpu_sum",
            "--format", "json",
            str(report_path),
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=30
            )
            
            if proc.returncode != 0:
                return {"error": stderr.decode().strip()}
            
            # Parse JSON output
            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return {"raw_output": stdout.decode()}
                
        except asyncio.TimeoutError:
            return {"error": "nsys stats timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_metric(
        self,
        extractor: str,
        report_path: Path,
        stats_output: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Any:
        """Extract a specific metric from nsys data."""
        
        # File metadata metrics
        if extractor == "report_path":
            return str(report_path)
        
        if extractor == "file_size_mb":
            return metadata.get("file_size_mb", 0)
        
        if extractor == "has_nsys_data":
            return report_path.exists()
        
        # GPU utilization (from stats if available)
        if extractor == "gpu_utilization_percent":
            return self._parse_gpu_utilization(stats_output)
        
        if extractor == "kernel_count":
            return self._parse_kernel_count(stats_output)
        
        if extractor == "total_kernel_time_ms":
            return self._parse_total_kernel_time(stats_output)
        
        # Memory metrics
        if extractor == "memory_bandwidth_gbps":
            return self._parse_memory_bandwidth(stats_output)
        
        # Timeline analysis
        if extractor == "idle_time_percent":
            return self._parse_idle_time(stats_output)
        
        # Default: return None for unknown extractors
        return None
    
    def _parse_gpu_utilization(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse GPU utilization from nsys stats output."""
        # Try to extract from gpu_sum report
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    # Average utilization across GPUs
                    utilizations = [r.get("utilization", 0) for r in rows if "utilization" in r]
                    if utilizations:
                        return sum(utilizations) / len(utilizations)
        return None
    
    def _parse_kernel_count(self, stats_output: Dict[str, Any]) -> Optional[int]:
        """Parse total kernel count from nsys stats output."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                return len(rows)
        return None
    
    def _parse_total_kernel_time(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse total kernel execution time from nsys stats output."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                total_time = 0
                for row in rows:
                    duration = row.get("duration", 0)
                    if isinstance(duration, (int, float)):
                        total_time += duration
                return total_time / 1_000_000  # Convert to ms
        return None
    
    def _parse_memory_bandwidth(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse memory bandwidth from nsys stats output."""
        # This is a simplified extraction - real implementation would parse more details
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    # Look for memory bandwidth metric
                    for row in rows:
                        if "memory_bandwidth" in row:
                            return row["memory_bandwidth"]
        return None
    
    def _parse_idle_time(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse idle time percentage from nsys stats output."""
        # Simplified - would need timeline analysis for accurate idle time
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    utilizations = [r.get("utilization", 100) for r in rows if "utilization" in r]
                    if utilizations:
                        avg_util = sum(utilizations) / len(utilizations)
                        return 100 - avg_util
        return None
    
    def _get_default_extractors(self, extractors: List[str]) -> Dict[str, Any]:
        """Return default values when no report is available."""
        defaults = {
            "report_path": None,
            "file_size_mb": 0,
            "has_nsys_data": False,
            "gpu_utilization_percent": None,
            "kernel_count": None,
            "total_kernel_time_ms": None,
            "memory_bandwidth_gbps": None,
            "idle_time_percent": None,
        }
        return {k: defaults.get(k) for k in extractors if k in defaults}
