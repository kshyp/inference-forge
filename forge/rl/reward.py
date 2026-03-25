"""Reward calculation for vLLM configuration optimization.

The reward function is designed to:
1. Not degrade latency (p50) by more than 5% vs baseline
2. Maximize throughput improvement vs baseline
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import math


@dataclass
class BaselineMetrics:
    """Baseline metrics from initial benchmark run."""
    throughput: float          # requests/sec
    latency_p50: float         # ms
    latency_p99: float         # ms
    ttft_p50: float           # ms (time to first token)
    tpot_p50: float           # ms (time per output token)
    
    @classmethod
    def from_benchmark_result(cls, result: Dict[str, Any]) -> "BaselineMetrics":
        """Extract baseline metrics from benchmark result."""
        metrics = result.get("benchmark_result", {}).get("metrics", {})
        
        def get_percentile(metric: str, p: str) -> float:
            """Safely extract percentile value."""
            val = metrics.get(metric, {}).get(p, 0)
            # GuideLLM sometimes returns in seconds, convert to ms
            return val * 1000 if val and val < 10 else val
        
        return cls(
            throughput=metrics.get("requests_per_sec", 0),
            latency_p50=get_percentile("itl_ms", "p50"),
            latency_p99=get_percentile("itl_ms", "p99"),
            ttft_p50=get_percentile("ttft_ms", "p50"),
            tpot_p50=get_percentile("tpot_ms", "p50"),
        )


class RewardCalculator:
    """
    Calculate reward based on throughput improvement with latency constraint.
    
    Reward Function:
    - If latency_p50 > max_latency_threshold: reward = 0 (constraint violated)
    - Else: reward = throughput_improvement_ratio (unbounded or capped)
    
    Latency constraint can be specified two ways:
    1. latency_tolerance: relative to baseline (e.g., 0.05 = 5% degradation allowed)
    2. max_latency_ms: absolute maximum in milliseconds (overrides tolerance)
    
    This encourages:
    1. Finding configs that don't exceed latency threshold
    2. Maximizing throughput within that constraint
    """
    
    def __init__(
        self,
        baseline: BaselineMetrics,
        latency_tolerance: float = 0.05,  # 5% degradation allowed
        max_reward: float = 2.0,          # Cap reward at 200% improvement
        max_latency_ms: Optional[float] = None,  # Absolute max latency (ms)
    ):
        """
        Initialize reward calculator.
        
        Args:
            baseline: Baseline metrics from initial run
            latency_tolerance: Maximum allowed latency degradation (as ratio)
            max_reward: Maximum reward value (prevents extreme outliers)
            max_latency_ms: Absolute maximum p50 latency in ms (overrides tolerance)
        """
        self.baseline = baseline
        self.latency_tolerance = latency_tolerance
        self.max_reward = max_reward
        
        # Use absolute max_latency_ms if provided, otherwise calculate from tolerance
        if max_latency_ms is not None:
            self.max_latency_threshold = max_latency_ms
            self.use_absolute_threshold = True
        else:
            self.max_latency_threshold = baseline.latency_p50 * (1 + latency_tolerance)
            self.use_absolute_threshold = False
    
    def calculate(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate reward from benchmark metrics.
        
        Args:
            metrics: Dictionary with keys:
                - throughput: requests/sec
                - latency_p50: p50 latency in ms
                - latency_p99: p99 latency in ms (optional)
                - failed_requests: count of failures (optional)
        
        Returns:
            Reward value (0 if latency constraint violated, else throughput ratio)
        """
        # Extract values
        throughput = metrics.get("throughput", 0)
        latency_p50 = metrics.get("latency_p50", float('inf'))
        failed_requests = metrics.get("failed_requests", 0)
        
        # Hard constraint: no failures
        if failed_requests > 0:
            return 0.0
        
        # Hard constraint: latency degradation
        if latency_p50 > self.max_latency_threshold:
            return 0.0
        
        # Calculate throughput improvement ratio
        if self.baseline.throughput <= 0:
            # Avoid division by zero
            return 0.0
        
        improvement_ratio = throughput / self.baseline.throughput
        
        # Cap at max_reward to prevent extreme values
        reward = min(improvement_ratio, self.max_reward)
        
        return reward
    
    def calculate_with_explanation(self, metrics: Dict[str, Any]) -> Tuple[float, str]:
        """
        Calculate reward with human-readable explanation.
        
        Returns:
            Tuple of (reward, explanation_string)
        """
        # Extract values
        throughput = metrics.get("throughput", 0)
        latency_p50 = metrics.get("latency_p50", float('inf'))
        failed_requests = metrics.get("failed_requests", 0)
        
        # Check constraints
        if failed_requests > 0:
            return 0.0, f"FAILED: {failed_requests} failed requests"
        
        if latency_p50 > self.max_latency_threshold:
            latency_delta_pct = ((latency_p50 / self.baseline.latency_p50) - 1) * 100
            if self.use_absolute_threshold:
                return (
                    0.0,
                    f"REJECTED: Latency p50 {latency_p50:.1f}ms "
                    f"(exceeds max {self.max_latency_threshold:.1f}ms, "
                    f"baseline was {self.baseline.latency_p50:.1f}ms)"
                )
            else:
                return (
                    0.0,
                    f"REJECTED: Latency p50 {latency_p50:.1f}ms "
                    f"({latency_delta_pct:+.1f}% vs baseline {self.baseline.latency_p50:.1f}ms, "
                    f"limit {self.latency_tolerance*100:.0f}%)"
                )
        
        # Calculate improvement
        if self.baseline.throughput <= 0:
            return 0.0, "ERROR: Invalid baseline throughput"
        
        improvement_ratio = throughput / self.baseline.throughput
        improvement_pct = (improvement_ratio - 1) * 100
        latency_delta_pct = ((latency_p50 / self.baseline.latency_p50) - 1) * 100
        
        reward = min(improvement_ratio, self.max_reward)
        
        if self.use_absolute_threshold:
            explanation = (
                f"ACCEPTED: Throughput {throughput:.1f} req/s "
                f"({improvement_pct:+.1f}% vs baseline {self.baseline.throughput:.1f}), "
                f"Latency p50 {latency_p50:.1f}ms ({latency_delta_pct:+.1f}%, max {self.max_latency_threshold:.1f}ms), "
                f"Reward={reward:.3f}"
            )
        else:
            explanation = (
                f"ACCEPTED: Throughput {throughput:.1f} req/s "
                f"({improvement_pct:+.1f}% vs baseline {self.baseline.throughput:.1f}), "
                f"Latency p50 {latency_p50:.1f}ms ({latency_delta_pct:+.1f}%), "
                f"Reward={reward:.3f}"
            )
        
        return reward, explanation
    
    def get_constraint_status(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed constraint status for debugging.
        
        Returns:
            Dictionary with constraint check results
        """
        latency_p50 = metrics.get("latency_p50", float('inf'))
        failed_requests = metrics.get("failed_requests", 0)
        throughput = metrics.get("throughput", 0)
        
        latency_degradation = latency_p50 / self.baseline.latency_p50 - 1
        throughput_improvement = throughput / self.baseline.throughput - 1
        
        status = {
            "latency_constraint_met": latency_p50 <= self.max_latency_threshold,
            "latency_degradation_pct": latency_degradation * 100,
            "no_failures": failed_requests == 0,
            "failed_requests": failed_requests,
            "throughput_improvement_pct": throughput_improvement * 100,
            "baseline_throughput": self.baseline.throughput,
            "current_throughput": throughput,
            "baseline_latency_p50": self.baseline.latency_p50,
            "current_latency_p50": latency_p50,
            "use_absolute_threshold": self.use_absolute_threshold,
            "max_latency_threshold_ms": self.max_latency_threshold,
        }
        
        if not self.use_absolute_threshold:
            status["latency_threshold_pct"] = self.latency_tolerance * 100
        
        return status


class LatencyConstrainedReward:
    """
    Alternative reward function with smoother penalty.
    
    Instead of hard 0/1 cutoff at 5%, uses a sigmoid penalty:
    - Near full reward when latency < 5% degradation
    - Gradual penalty as latency approaches 10% degradation
    - Zero reward when latency > 10% degradation
    """
    
    def __init__(
        self,
        baseline: BaselineMetrics,
        soft_limit: float = 0.05,   # 5% - start penalizing
        hard_limit: float = 0.10,   # 10% - zero reward
        max_reward: float = 2.0,
    ):
        self.baseline = baseline
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.max_reward = max_reward
    
    def _latency_penalty(self, latency: float) -> float:
        """Calculate penalty factor based on latency degradation."""
        degradation = (latency / self.baseline.latency_p50) - 1
        
        if degradation <= self.soft_limit:
            return 1.0  # No penalty
        elif degradation >= self.hard_limit:
            return 0.0  # Full penalty
        else:
            # Smooth sigmoid transition
            x = (degradation - self.soft_limit) / (self.hard_limit - self.soft_limit)
            return 1.0 / (1.0 + math.exp(10 * (x - 0.5)))
    
    def calculate(self, metrics: Dict[str, Any]) -> float:
        """Calculate reward with smooth latency penalty."""
        throughput = metrics.get("throughput", 0)
        latency_p50 = metrics.get("latency_p50", float('inf'))
        failed_requests = metrics.get("failed_requests", 0)
        
        if failed_requests > 0:
            return 0.0
        
        if self.baseline.throughput <= 0:
            return 0.0
        
        # Base reward from throughput improvement
        improvement_ratio = throughput / self.baseline.throughput
        
        # Apply latency penalty
        latency_penalty = self._latency_penalty(latency_p50)
        
        reward = improvement_ratio * latency_penalty
        return min(reward, self.max_reward)
