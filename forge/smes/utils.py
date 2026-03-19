"""Utility functions for SME implementations."""

from typing import Dict, Any, List, Optional


def format_nsys_for_prompt(nsys_data: Dict[str, Any], include_timeline: bool = True) -> str:
    """Format NSYS analysis data for LLM prompts.
    
    Args:
        nsys_data: NSYS analysis dictionary from NSYSAnalyzer
        include_timeline: Whether to include the visual timeline
        
    Returns:
        Formatted string for inclusion in prompts
    """
    if not nsys_data:
        return "## NSYS Timeline Analysis\n\nNo NSYS data available.\n"
    
    # Check if we have enhanced analysis or just a path
    if "metrics" not in nsys_data:
        # Legacy format - just a path
        report_path = nsys_data.get("report_path", nsys_data.get("path", "N/A"))
        return f"## NSYS Timeline Analysis\n\nReport path: {report_path}\n"
    
    # Enhanced analysis available
    metrics = nsys_data.get("metrics", {})
    llm_summary = nsys_data.get("llm_summary", "")
    bottlenecks = nsys_data.get("bottlenecks", [])
    recommendations = nsys_data.get("recommendations", [])
    observations = nsys_data.get("key_observations", [])
    timeline_viz = nsys_data.get("timeline_visualization", "")
    
    section = "## NSYS GPU Timeline Analysis\n\n"
    
    if llm_summary:
        section += f"**Summary**: {llm_summary}\n\n"
    
    # Key metrics table
    section += "**Key Metrics**:\n"
    
    gpu_util = metrics.get("gpu_utilization_percent")
    if gpu_util is not None:
        status = "✓ Good" if gpu_util >= 70 else "⚠ Low" if gpu_util < 50 else "○ Moderate"
        section += f"- GPU Utilization: {gpu_util:.1f}% {status}\n"
    
    idle_time = metrics.get("gpu_idle_time_percent")
    if idle_time is not None:
        status = "✓ Good" if idle_time < 15 else "⚠ High" if idle_time > 30 else "○ Moderate"
        section += f"- GPU Idle Time: {idle_time:.1f}% {status}\n"
    
    kernel_count = metrics.get("kernel_count")
    if kernel_count is not None:
        section += f"- Total Kernels: {kernel_count:,}\n"
    
    total_kernel_time = metrics.get("total_kernel_time_ms")
    if total_kernel_time is not None:
        section += f"- Total Kernel Time: {total_kernel_time:.1f}ms\n"
    
    avg_kernel_time = metrics.get("avg_kernel_time_ms")
    if avg_kernel_time is not None:
        section += f"- Avg Kernel Time: {avg_kernel_time:.3f}ms\n"
    
    memory_bw = metrics.get("memory_bandwidth_gbps")
    if memory_bw is not None:
        section += f"- Memory Bandwidth: {memory_bw:.1f} GB/s\n"
    
    hbm_util = metrics.get("hbm_utilization_percent")
    if hbm_util is not None:
        status = "⚠ Memory Bound" if hbm_util > 80 else "✓ OK"
        section += f"- HBM Utilization: {hbm_util:.1f}% {status}\n"
    
    idle_gaps = metrics.get("idle_gaps_ms", [])
    if idle_gaps:
        large_gaps = [g for g in idle_gaps if g > 5.0]
        critical_gaps = [g for g in idle_gaps if g > 10.0]
        if critical_gaps:
            section += f"- Idle Gaps: {len(idle_gaps)} total, {len(critical_gaps)} critical (>10ms)\n"
        elif large_gaps:
            section += f"- Idle Gaps: {len(idle_gaps)} total, {len(large_gaps)} large (>5ms)\n"
        else:
            section += f"- Idle Gaps: {len(idle_gaps)} (all <5ms)\n"
        if large_gaps:
            section += f"  - Max gap: {max(idle_gaps):.1f}ms\n"
    
    section += "\n"
    
    # Observations
    if observations:
        section += "**Key Observations**:\n"
        for obs in observations[:5]:  # Limit to top 5
            section += f"- {obs}\n"
        section += "\n"
    
    # Bottlenecks
    if bottlenecks:
        section += "**Detected Bottlenecks**:\n"
        for b in bottlenecks[:4]:  # Limit to top 4
            section += f"- {b}\n"
        section += "\n"
    
    # Recommendations
    if recommendations:
        section += "**NSYS Analysis Recommendations**:\n"
        for r in recommendations[:4]:  # Limit to top 4
            section += f"- {r}\n"
        section += "\n"
    
    # Timeline visualization
    if include_timeline and timeline_viz:
        section += f"**Timeline Visualization**:\n```\n{timeline_viz}\n```\n"
    
    return section


def format_metrics_summary(metrics: Dict[str, Any], title: str = "Metrics") -> str:
    """Format a metrics dictionary for prompts.
    
    Args:
        metrics: Dictionary of metrics
        title: Section title
        
    Returns:
        Formatted string
    """
    if not metrics:
        return f"## {title}\n\nNo data available.\n"
    
    section = f"## {title}\n\n"
    
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            section += f"- {key}: {value:.2f}\n"
        elif isinstance(value, int):
            section += f"- {key}: {value:,}\n"
        elif isinstance(value, str):
            section += f"- {key}: {value}\n"
        elif isinstance(value, bool):
            section += f"- {key}: {'Yes' if value else 'No'}\n"
        # Skip complex nested structures
    
    return section + "\n"


def extract_idle_gap_insights(idle_gaps_ms: List[float]) -> Dict[str, Any]:
    """Analyze idle gaps and extract insights.
    
    Args:
        idle_gaps_ms: List of idle gap durations in milliseconds
        
    Returns:
        Dictionary with insights
    """
    if not idle_gaps_ms:
        return {
            "has_gaps": False,
            "severity": "none",
            "insights": ["No significant idle gaps detected"]
        }
    
    insights = []
    severity = "low"
    
    total_gaps = len(idle_gaps_ms)
    large_gaps = [g for g in idle_gaps_ms if g > 5.0]
    critical_gaps = [g for g in idle_gaps_ms if g > 10.0]
    
    if critical_gaps:
        severity = "critical"
        insights.append(f"Found {len(critical_gaps)} critical idle gaps >10ms (max: {max(critical_gaps):.1f}ms)")
        insights.append("Large scheduling delays suggest batch size or scheduler issues")
    elif large_gaps:
        severity = "moderate"
        insights.append(f"Found {len(large_gaps)} idle gaps >5ms (max: {max(large_gaps):.1f}ms)")
    
    avg_gap = sum(idle_gaps_ms) / len(idle_gaps_ms)
    if avg_gap > 3.0:
        insights.append(f"Average idle gap is {avg_gap:.1f}ms - consider increasing batch size")
    
    return {
        "has_gaps": True,
        "total_gaps": total_gaps,
        "large_gaps": len(large_gaps),
        "critical_gaps": len(critical_gaps),
        "max_gap_ms": max(idle_gaps_ms),
        "avg_gap_ms": avg_gap,
        "severity": severity,
        "insights": insights
    }


def infer_bottleneck_from_nsys(nsys_data: Dict[str, Any]) -> Optional[str]:
    """Infer primary bottleneck from NSYS analysis.
    
    Args:
        nsys_data: NSYS analysis dictionary
        
    Returns:
        Bottleneck type string or None
    """
    if not nsys_data or "metrics" not in nsys_data:
        return None
    
    metrics = nsys_data.get("metrics", {})
    
    gpu_util = metrics.get("gpu_utilization_percent")
    idle_time = metrics.get("gpu_idle_time_percent")
    idle_gaps = metrics.get("idle_gaps_ms", [])
    hbm_util = metrics.get("hbm_utilization_percent")
    
    # Check for memory bound
    if hbm_util and hbm_util > 80:
        return "memory_bandwidth_bound"
    
    # Check for scheduling issues
    if idle_gaps:
        critical_gaps = [g for g in idle_gaps if g > 10.0]
        if critical_gaps:
            return "scheduling_overhead"
    
    # Check for low utilization
    if gpu_util is not None:
        if gpu_util < 50:
            return "low_gpu_utilization"
    
    if idle_time is not None:
        if idle_time > 30:
            return "scheduling_inefficiency"
    
    return None
