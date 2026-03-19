"""NSYS Report Analyzer - Extracts and analyzes NVIDIA Systems profiler data.

This module provides:
1. Extraction of structured metrics from nsys-rep files using `nsys stats`
2. LLM-powered analysis of the timeline and bottlenecks
3. Visual timeline representation for SME consumption
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from forge.llm.base import Prompt
from forge.llm.pool import IntelligencePool, get_global_intelligence_pool

logger = logging.getLogger(__name__)


@dataclass
class NSYSMetrics:
    """Structured metrics extracted from nsys report."""
    # Basic file info
    report_path: str = ""
    file_size_mb: float = 0.0
    
    # GPU utilization
    gpu_utilization_percent: Optional[float] = None
    gpu_idle_time_percent: Optional[float] = None
    
    # Kernel metrics
    kernel_count: Optional[int] = None
    total_kernel_time_ms: Optional[float] = None
    avg_kernel_time_ms: Optional[float] = None
    
    # Memory metrics
    memory_bandwidth_gbps: Optional[float] = None
    hbm_utilization_percent: Optional[float] = None
    
    # Timeline analysis
    total_duration_ms: Optional[float] = None
    idle_gaps_ms: List[float] = field(default_factory=list)
    prefill_phase_duration_ms: Optional[float] = None
    decode_phase_duration_ms: Optional[float] = None
    
    # CUDA API metrics
    cuda_launch_count: Optional[int] = None
    avg_cuda_launch_latency_us: Optional[float] = None
    
    # Raw data for LLM analysis
    raw_stats_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NSYSAnalysis:
    """Complete NSYS analysis including metrics and LLM insights."""
    metrics: NSYSMetrics
    llm_summary: str = ""
    key_observations: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timeline_visualization: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SME consumption."""
        return {
            "report_path": self.metrics.report_path,
            "file_size_mb": self.metrics.file_size_mb,
            "metrics": {
                "gpu_utilization_percent": self.metrics.gpu_utilization_percent,
                "gpu_idle_time_percent": self.metrics.gpu_idle_time_percent,
                "kernel_count": self.metrics.kernel_count,
                "total_kernel_time_ms": self.metrics.total_kernel_time_ms,
                "avg_kernel_time_ms": self.metrics.avg_kernel_time_ms,
                "memory_bandwidth_gbps": self.metrics.memory_bandwidth_gbps,
                "hbm_utilization_percent": self.metrics.hbm_utilization_percent,
                "total_duration_ms": self.metrics.total_duration_ms,
                "idle_gaps_ms": self.metrics.idle_gaps_ms,
                "prefill_phase_duration_ms": self.metrics.prefill_phase_duration_ms,
                "decode_phase_duration_ms": self.metrics.decode_phase_duration_ms,
                "cuda_launch_count": self.metrics.cuda_launch_count,
                "avg_cuda_launch_latency_us": self.metrics.avg_cuda_launch_latency_us,
            },
            "llm_summary": self.llm_summary,
            "key_observations": self.key_observations,
            "bottlenecks": self.bottlenecks,
            "recommendations": self.recommendations,
            "timeline_visualization": self.timeline_visualization,
        }


class NSYSAnalyzer:
    """Analyzes NSYS profiler reports.
    
    Usage:
        analyzer = NSYSAnalyzer()
        analysis = await analyzer.analyze("/path/to/report.nsys-rep")
        
        # Pass to SMEs
        profiling_data = {
            "nsys_report": analysis.to_dict()
        }
    """
    
    SYSTEM_PROMPT_FOR_ANALYSIS = """You are an expert GPU performance analyst specializing in vLLM inference optimization.

Your task: Analyze NVIDIA Systems (nsys) profiler data and identify performance bottlenecks.

Key concepts to look for:
- GPU utilization: % of time GPU is busy (low = wasted capacity)
- Idle gaps: Periods where GPU is not working (scheduling inefficiency)
- Kernel patterns: FlashAttention, GEMM, memory copy operations
- Prefill vs Decode: Different phases have different compute characteristics
- Memory bandwidth: HBM utilization (high = memory bound)
- CUDA launch overhead: Time spent launching kernels

Focus on actionable insights for vLLM configuration tuning."""

    def __init__(self, intelligence_pool: Optional[IntelligencePool] = None):
        """Initialize NSYS analyzer.
        
        Args:
            intelligence_pool: Optional custom pool. Uses global pool if not provided.
        """
        self.pool = intelligence_pool or get_global_intelligence_pool()
        self._nsys_available = shutil.which("nsys") is not None
    
    async def analyze(self, report_path: str, enable_llm_analysis: bool = True) -> NSYSAnalysis:
        """Analyze an NSYS report file.
        
        Args:
            report_path: Path to the .nsys-rep file
            enable_llm_analysis: Whether to run LLM analysis (requires API keys)
            
        Returns:
            NSYSAnalysis with metrics and insights
        """
        path = Path(report_path)
        if not path.exists():
            logger.warning(f"NSYS report not found: {report_path}")
            return self._create_empty_analysis(report_path)
        
        # Step 1: Extract structured metrics using nsys stats
        logger.info(f"Extracting metrics from {report_path}")
        metrics = await self._extract_metrics(path)
        
        # Step 2: Generate visual timeline
        timeline_viz = self._generate_timeline_visualization(metrics)
        
        # Step 3: Run LLM analysis if enabled
        llm_summary = ""
        observations = []
        bottlenecks = []
        recommendations = []
        
        if enable_llm_analysis and self.pool.sources:
            logger.info("Running LLM analysis of NSYS data...")
            llm_summary, observations, bottlenecks, recommendations = await self._run_llm_analysis(metrics)
        else:
            # Generate rule-based analysis as fallback
            observations, bottlenecks, recommendations = self._rule_based_analysis(metrics)
            llm_summary = "LLM analysis disabled. Using rule-based analysis."
        
        return NSYSAnalysis(
            metrics=metrics,
            llm_summary=llm_summary,
            key_observations=observations,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            timeline_visualization=timeline_viz,
        )
    
    async def _extract_metrics(self, report_path: Path) -> NSYSMetrics:
        """Extract metrics from nsys report using nsys stats."""
        metrics = NSYSMetrics(
            report_path=str(report_path),
            file_size_mb=report_path.stat().st_size / (1024 * 1024),
        )
        
        if not self._nsys_available:
            logger.warning("nsys CLI not available, cannot extract detailed metrics")
            return metrics
        
        # Run nsys stats with multiple report types
        try:
            stats_output = await self._run_nsys_stats(report_path)
            metrics.raw_stats_output = stats_output
            
            # Parse GPU utilization from gpu_sum report
            metrics.gpu_utilization_percent = self._parse_gpu_utilization(stats_output)
            if metrics.gpu_utilization_percent is not None:
                metrics.gpu_idle_time_percent = 100.0 - metrics.gpu_utilization_percent
            
            # Parse kernel metrics
            metrics.kernel_count = self._parse_kernel_count(stats_output)
            metrics.total_kernel_time_ms = self._parse_total_kernel_time(stats_output)
            if metrics.kernel_count and metrics.total_kernel_time_ms and metrics.kernel_count > 0:
                metrics.avg_kernel_time_ms = metrics.total_kernel_time_ms / metrics.kernel_count
            
            # Parse memory metrics
            metrics.memory_bandwidth_gbps = self._parse_memory_bandwidth(stats_output)
            metrics.hbm_utilization_percent = self._parse_hbm_utilization(stats_output)
            
            # Parse timeline and idle gaps
            metrics.total_duration_ms = self._parse_total_duration(stats_output)
            metrics.idle_gaps_ms = self._parse_idle_gaps(stats_output)
            
            # Parse CUDA API metrics
            metrics.cuda_launch_count = self._parse_cuda_launch_count(stats_output)
            metrics.avg_cuda_launch_latency_us = self._parse_cuda_launch_latency(stats_output)
            
        except Exception as e:
            logger.error(f"Failed to extract metrics: {e}")
        
        return metrics
    
    async def _run_nsys_stats(self, report_path: Path) -> Dict[str, Any]:
        """Run nsys stats CLI to extract metrics."""
        cmd = [
            "nsys", "stats",
            "--report", "cuda_gpu_trace,cuda_api_sum,gpu_sum,nvtx_sum",
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
                timeout=60  # nsys stats can take time for large reports
            )
            
            if proc.returncode != 0:
                logger.warning(f"nsys stats failed: {stderr.decode().strip()}")
                return {"error": stderr.decode().strip()}
            
            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return {"raw_output": stdout.decode(), "error": "JSON parse failed"}
                
        except asyncio.TimeoutError:
            logger.warning("nsys stats timed out")
            return {"error": "timeout"}
        except Exception as e:
            logger.warning(f"nsys stats error: {e}")
            return {"error": str(e)}
    
    def _parse_gpu_utilization(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse GPU utilization from nsys stats output."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    utilizations = [r.get("utilization", 0) for r in rows if "utilization" in r]
                    if utilizations:
                        return sum(utilizations) / len(utilizations)
        return None
    
    def _parse_kernel_count(self, stats_output: Dict[str, Any]) -> Optional[int]:
        """Parse total kernel count."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                return len(rows)
        return None
    
    def _parse_total_kernel_time(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse total kernel execution time in ms."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                total_time = 0
                for row in rows:
                    duration = row.get("duration", 0)
                    if isinstance(duration, (int, float)):
                        total_time += duration
                return total_time / 1_000_000  # Convert ns to ms
        return None
    
    def _parse_memory_bandwidth(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse memory bandwidth in GB/s."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    for row in rows:
                        if "memory_bandwidth" in row:
                            return row["memory_bandwidth"]
        return None
    
    def _parse_hbm_utilization(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse HBM (memory) utilization percentage."""
        # HBM utilization is often in gpu_sum report as memory_utilization
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "gpu_sum":
                rows = report.get("data", [])
                if rows:
                    for row in rows:
                        if "memory_utilization" in row:
                            return row["memory_utilization"]
        return None
    
    def _parse_total_duration(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse total profile duration in ms."""
        # Try to get from cuda_api_sum or calculate from kernel timestamps
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                if rows:
                    timestamps = []
                    for row in rows:
                        start = row.get("start", 0)
                        end = row.get("end", 0)
                        if start and end:
                            timestamps.extend([start, end])
                    if timestamps:
                        duration_ns = max(timestamps) - min(timestamps)
                        return duration_ns / 1_000_000  # Convert to ms
        return None
    
    def _parse_idle_gaps(self, stats_output: Dict[str, Any]) -> List[float]:
        """Parse idle gaps between kernel executions."""
        idle_gaps = []
        reports = stats_output.get("reports", [])
        
        for report in reports:
            if report.get("name") == "cuda_gpu_trace":
                rows = report.get("data", [])
                if len(rows) < 2:
                    return idle_gaps
                
                # Sort by start time
                sorted_rows = sorted(rows, key=lambda r: r.get("start", 0))
                
                # Find gaps between consecutive kernels
                for i in range(1, len(sorted_rows)):
                    prev_end = sorted_rows[i-1].get("end", 0)
                    curr_start = sorted_rows[i].get("start", 0)
                    
                    if prev_end and curr_start:
                        gap_ns = curr_start - prev_end
                        gap_ms = gap_ns / 1_000_000
                        if gap_ms > 1.0:  # Only gaps > 1ms
                            idle_gaps.append(gap_ms)
                
                break
        
        return idle_gaps
    
    def _parse_cuda_launch_count(self, stats_output: Dict[str, Any]) -> Optional[int]:
        """Parse number of CUDA API calls."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_api_sum":
                rows = report.get("data", [])
                # Count launch operations
                count = 0
                for row in rows:
                    name = row.get("name", "").lower()
                    if "launch" in name or "memcpy" in name:
                        count += row.get("count", 0)
                return count if count > 0 else None
        return None
    
    def _parse_cuda_launch_latency(self, stats_output: Dict[str, Any]) -> Optional[float]:
        """Parse average CUDA launch latency in microseconds."""
        reports = stats_output.get("reports", [])
        for report in reports:
            if report.get("name") == "cuda_api_sum":
                rows = report.get("data", [])
                for row in rows:
                    name = row.get("name", "").lower()
                    if "launch" in name:
                        avg_time = row.get("avg", 0)
                        # avg is in ns, convert to us
                        return avg_time / 1000.0
        return None
    
    def _generate_timeline_visualization(self, metrics: NSYSMetrics) -> str:
        """Generate a text-based timeline visualization."""
        if not metrics.total_duration_ms:
            return "No timeline data available"
        
        lines = [
            "NSYS Timeline Analysis",
            "=" * 60,
            "",
        ]
        
        # GPU Utilization bar
        if metrics.gpu_utilization_percent is not None:
            util = metrics.gpu_utilization_percent
            bar_len = 50
            filled = int(bar_len * util / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"GPU Utilization: {util:.1f}%")
            lines.append(f"[{bar}]")
            lines.append("")
        
        # Idle gaps summary
        if metrics.idle_gaps_ms:
            lines.append(f"Idle Gaps Detected: {len(metrics.idle_gaps_ms)}")
            large_gaps = [g for g in metrics.idle_gaps_ms if g > 5.0]
            if large_gaps:
                lines.append(f"  Large gaps (>5ms): {len(large_gaps)}")
                lines.append(f"  Max gap: {max(metrics.idle_gaps_ms):.1f}ms")
            lines.append("")
        
        # Kernel summary
        if metrics.kernel_count:
            lines.append(f"Total Kernels: {metrics.kernel_count}")
            lines.append(f"Total Kernel Time: {metrics.total_kernel_time_ms:.1f}ms")
            if metrics.avg_kernel_time_ms:
                lines.append(f"Avg Kernel Time: {metrics.avg_kernel_time_ms:.3f}ms")
            lines.append("")
        
        # Memory
        if metrics.memory_bandwidth_gbps:
            lines.append(f"Memory Bandwidth: {metrics.memory_bandwidth_gbps:.1f} GB/s")
        if metrics.hbm_utilization_percent:
            lines.append(f"HBM Utilization: {metrics.hbm_utilization_percent:.1f}%")
        
        return "\n".join(lines)
    
    async def _run_llm_analysis(self, metrics: NSYSMetrics) -> Tuple[str, List[str], List[str], List[str]]:
        """Run LLM analysis on NSYS metrics.
        
        Returns:
            Tuple of (summary, observations, bottlenecks, recommendations)
        """
        # Build prompt
        prompt = self._build_analysis_prompt(metrics)
        
        # Call LLM (use available sources, preferring higher quality)
        try:
            results = await self.pool.call_all(prompt, deterministic=True)
            
            # Find first successful result, preferring higher quality sources
            successful_results = [r for r in results if r.success and r.response]
            if not successful_results:
                raise RuntimeError("All LLM calls failed")
            
            # Sort by quality tier and pick the best
            best_result = max(
                successful_results,
                key=lambda r: r.source.capabilities.get("quality_tier", 0.5)
            )
            
            # Parse response
            content = best_result.response.content
            analysis = self._parse_llm_response(content)
            
            return analysis
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using rule-based fallback")
            observations, bottlenecks, recommendations = self._rule_based_analysis(metrics)
            return "Rule-based analysis (LLM failed)", observations, bottlenecks, recommendations
    
    def _build_analysis_prompt(self, metrics: NSYSMetrics) -> Prompt:
        """Build prompt for LLM analysis."""
        
        # Build metrics summary
        gpu_util_str = f"{metrics.gpu_utilization_percent:.1f}%" if metrics.gpu_utilization_percent is not None else "N/A"
        idle_time_str = f"{metrics.gpu_idle_time_percent:.1f}%" if metrics.gpu_idle_time_percent is not None else "N/A"
        kernel_count_str = str(metrics.kernel_count) if metrics.kernel_count is not None else "N/A"
        total_time_str = f"{metrics.total_kernel_time_ms:.1f}ms" if metrics.total_kernel_time_ms is not None else "N/A"
        avg_time_str = f"{metrics.avg_kernel_time_ms:.3f}ms" if metrics.avg_kernel_time_ms is not None else "N/A"
        mem_bw_str = f"{metrics.memory_bandwidth_gbps:.1f}GB/s" if metrics.memory_bandwidth_gbps is not None else "N/A"
        hbm_str = f"{metrics.hbm_utilization_percent:.1f}%" if metrics.hbm_utilization_percent is not None else "N/A"
        
        metrics_text = f"""NSYS Metrics Summary:
- GPU Utilization: {gpu_util_str}
- Idle Time: {idle_time_str}
- Total Kernels: {kernel_count_str}
- Total Kernel Time: {total_time_str}
- Avg Kernel Time: {avg_time_str}
- Memory Bandwidth: {mem_bw_str}
- HBM Utilization: {hbm_str}
- Idle Gaps >1ms: {len(metrics.idle_gaps_ms)}
"""
        
        if metrics.idle_gaps_ms:
            large_gaps = [g for g in metrics.idle_gaps_ms if g > 5.0]
            if large_gaps:
                metrics_text += f"- Large Idle Gaps (>5ms): {len(large_gaps)}, max={max(large_gaps):.1f}ms\n"
        
        user_content = f"""Analyze this NSYS profiler data for vLLM inference:

{metrics_text}

Provide your analysis in this JSON format:
{{
  "summary": "2-3 sentence overall assessment",
  "key_observations": ["observation 1", "observation 2", ...],
  "bottlenecks": ["bottleneck 1", "bottleneck 2", ...],
  "recommendations": ["config suggestion 1", "config suggestion 2", ...]
}}

Focus on actionable vLLM configuration changes. Recommend specific parameters to tune."""
        
        return Prompt(
            system=self.SYSTEM_PROMPT_FOR_ANALYSIS,
            user=user_content
        )
    
    def _parse_llm_response(self, content: str) -> Tuple[str, List[str], List[str], List[str]]:
        """Parse LLM response into structured analysis."""
        try:
            # Extract JSON
            data = self._extract_json(content)
            
            summary = data.get("summary", "")
            observations = data.get("key_observations", [])
            bottlenecks = data.get("bottlenecks", [])
            recommendations = data.get("recommendations", [])
            
            return summary, observations, bottlenecks, recommendations
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return content, [], [], []
    
    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from response."""
        content = content.strip()
        
        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try markdown code block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                return json.loads(content[start:end].strip())
        
        # Try generic code block
        if "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                return json.loads(content[start:end].strip())
        
        # Find JSON object bounds
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end+1])
        
        raise ValueError("Could not extract JSON")
    
    def _rule_based_analysis(self, metrics: NSYSMetrics) -> Tuple[List[str], List[str], List[str]]:
        """Generate analysis using rules when LLM is unavailable."""
        observations = []
        bottlenecks = []
        recommendations = []
        
        # GPU utilization analysis
        if metrics.gpu_utilization_percent is not None:
            if metrics.gpu_utilization_percent < 50:
                observations.append(f"Low GPU utilization: {metrics.gpu_utilization_percent:.1f}%")
                bottlenecks.append("GPU underutilization - batch size may be too small")
                recommendations.append("Increase max_num_seqs and max_num_batched_tokens")
            elif metrics.gpu_utilization_percent > 90:
                observations.append(f"High GPU utilization: {metrics.gpu_utilization_percent:.1f}%")
                bottlenecks.append("GPU saturated - may be compute or memory bound")
                recommendations.append("Check memory bandwidth utilization to determine if memory bound")
            else:
                observations.append(f"Good GPU utilization: {metrics.gpu_utilization_percent:.1f}%")
        
        # Idle gaps analysis
        if metrics.idle_gaps_ms:
            large_gaps = [g for g in metrics.idle_gaps_ms if g > 10.0]
            if large_gaps:
                observations.append(f"Found {len(large_gaps)} large idle gaps >10ms")
                bottlenecks.append("Significant scheduling delays between kernels")
                recommendations.append("Enable chunked prefill to reduce scheduling overhead")
                recommendations.append("Increase num_scheduler_steps to batch more work")
        
        # Memory bandwidth analysis
        if metrics.memory_bandwidth_gbps:
            # A100 has ~2 TB/s = 2000 GB/s theoretical
            # If using >70% of theoretical, likely memory bound
            observations.append(f"Memory bandwidth: {metrics.memory_bandwidth_gbps:.1f} GB/s")
        
        # HBM utilization
        if metrics.hbm_utilization_percent and metrics.hbm_utilization_percent > 80:
            bottlenecks.append("High HBM utilization - memory bandwidth bound")
            recommendations.append("Consider quantization (FP8/INT8) to reduce memory traffic")
            recommendations.append("Enable prefix caching to reduce redundant memory access")
        
        return observations, bottlenecks, recommendations
    
    def _create_empty_analysis(self, report_path: str) -> NSYSAnalysis:
        """Create empty analysis when report is not available."""
        metrics = NSYSMetrics(report_path=report_path)
        return NSYSAnalysis(
            metrics=metrics,
            llm_summary="NSYS report not available",
            key_observations=["No NSYS data collected"],
            bottlenecks=[],
            recommendations=[],
            timeline_visualization="No data available",
        )


# Convenience function
async def analyze_nsys_report(report_path: str, enable_llm: bool = True) -> Dict[str, Any]:
    """Convenience function to analyze an NSYS report.
    
    Args:
        report_path: Path to .nsys-rep file
        enable_llm: Whether to run LLM analysis
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = NSYSAnalyzer()
    analysis = await analyzer.analyze(report_path, enable_llm)
    return analysis.to_dict()
