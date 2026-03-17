"""Event dataclasses for Inference Forge.

Defines all data structures passed between agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    COMPLETED = "completed"


class Platform(Enum):
    """Hardware platforms."""
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    CPU = "cpu"
    TPU = "tpu"
    UNKNOWN = "unknown"


@dataclass
class MetricsDistribution:
    """Distribution of a metric (p50, p95, p99)."""
    p50: float
    p95: float
    p99: float
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class Task:
    """Base task structure."""
    id: UUID = field(default_factory=uuid4)
    type: str = ""  # "benchmark", "profile", "analyze"
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary for health checks."""
        return {
            "id": str(self.id),
            "type": self.type,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HardwareInfo:
    """Hardware configuration information."""
    gpu_count: int
    gpu_type: str  # "NVIDIA A100", "NVIDIA T4", etc.
    gpu_memory_gb: float
    driver_version: str
    cuda_version: Optional[str] = None
    platform: Platform = Platform.UNKNOWN


@dataclass
class SaturationPoint:
    """GuideLLM saturation point result."""
    rps: float  # Requests per second at saturation
    latency_ms: float  # Latency at saturation point
    error_rate: float  # Error rate at saturation
    confidence: str  # "high", "medium", "low"


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmarking."""
    ttft_ms: MetricsDistribution  # Time to first token
    tpot_ms: MetricsDistribution  # Time per output token
    itl_ms: MetricsDistribution   # Inter-token latency
    output_tokens_per_sec: float
    total_tokens_per_sec: float
    requests_per_sec: float
    
    # Additional context
    duration_seconds: float
    total_requests: int
    failed_requests: int


@dataclass
class BenchmarkResult:
    """Output from Agent 1 (Benchmark Runner)."""
    experiment_id: UUID
    config_flags: Dict[str, Any]
    
    # GuideLLM results
    saturation_point: SaturationPoint
    metrics: BenchmarkMetrics
    
    # Paths to raw outputs
    guidellm_report_path: Path
    vllm_logs_path: Path
    
    # Context for downstream agents
    hardware_info: HardwareInfo
    vllm_command: str  # Full command used to start server
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProfilingResult:
    """Output from Agent 2 (Profile Executor)."""
    experiment_id: UUID
    platform: Platform
    
    # Paths to profiler outputs
    reports: Dict[str, Path]  # {"nsys": Path, "ncu": Path, ...}
    
    # Extracted key metrics
    gpu_utilization_percent: Optional[MetricsDistribution] = None
    memory_bandwidth_gbps: Optional[MetricsDistribution] = None
    kernel_execution_time_ms: Optional[Dict[str, float]] = None
    
    # Reference to benchmark for context
    benchmark_result: Optional[BenchmarkResult] = None
    
    # Metadata
    profilers_attempted: List[str] = field(default_factory=list)
    profilers_succeeded: List[str] = field(default_factory=list)
    profilers_failed: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExpertOpinion:
    """Output from a single SME consultation."""
    expert_id: str
    expert_name: str
    
    # Did this expert find relevant issues?
    relevant: bool
    confidence: float  # 0-1
    
    # Analysis
    bottleneck_diagnosis: str = ""  # "You are memory bandwidth bound"
    recommendation: str = ""  # Conceptual recommendation
    expected_impact: str = ""  # "20-30% throughput increase"
    risks: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    
    # Raw LLM response for debugging
    raw_response: Optional[str] = None


@dataclass
class RankedExperiment:
    """A single experiment recommendation."""
    priority: int  # 1 = try first
    
    # What to change
    config_patch: Dict[str, Any]  # Changes from baseline
    
    # Why this change
    hypothesis: str  # "Memory bandwidth bound, try larger batches"
    expected_improvement: str  # "+25% throughput"
    success_criteria: str  # "throughput > 100 AND ttft < 100ms"
    
    # Source
    source_experts: List[str] = field(default_factory=list)
    confidence: float = 0.5  # 0-1
    
    # Optional fields with defaults
    config_removals: List[str] = field(default_factory=list)  # Flags to remove
    abort_conditions: List[str] = field(default_factory=list)  # ["oom", "ttft > 500ms"]


@dataclass
class ExperimentPlan:
    """Output from Agent 3 (Coordinator) - goes back to Agent 1."""
    iteration: int
    parent_experiment_id: UUID
    
    # Ranked experiments to try
    experiments: List[RankedExperiment]
    
    # Stop conditions for this iteration
    max_iterations: int = 10
    convergence_threshold: float = 0.05  # 5% improvement
    no_improvement_limit: int = 3
    
    # Coordinator reasoning (for audit/debug)
    signals_detected: List[str] = field(default_factory=list)
    smes_consulted: List[str] = field(default_factory=list)
    synthesis_reasoning: str = ""
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentStatus:
    """Health check response."""
    agent_id: str
    state: AgentState
    current_task: Optional[Dict[str, Any]] = None
    progress_percent: float = 0.0
    uptime_seconds: float = 0.0
    last_checkpoint: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class Checkpoint:
    """Saved agent state for crash recovery."""
    agent_id: str
    task_id: UUID
    data: Dict[str, Any]  # Agent-specific checkpoint data
    created_at: datetime = field(default_factory=datetime.now)
