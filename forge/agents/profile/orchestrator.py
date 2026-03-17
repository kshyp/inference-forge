"""Profiler Orchestrator - Decides which profilers to run and in what order.

The orchestrator maps data requirements from the SME Registry to actual
profiler implementations, handles dependencies, and sequences execution.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type

from forge.core.events import Platform

from .profilers.base import BaseProfiler, UnsupportedDataTypeError
from .profilers.benchmark_metrics import BenchmarkMetricsCollector
from .profilers.vllm_logs import VLLMLogCollector
from .profilers.mock_ncu import MockNCUProfiler


@dataclass
class ExecutionStep:
    """A single profiler execution step."""
    data_type: str
    profiler_id: str
    profiler_class: Type[BaseProfiler]
    required: bool
    extractors: List[str]
    dependencies: List[str]
    
    def __repr__(self) -> str:
        req_mark = "✅" if self.required else "☑️"
        return f"{req_mark} {self.profiler_id} → {self.data_type}"


@dataclass
class ExecutionPlan:
    """Complete execution plan for a profiling session."""
    steps: List[ExecutionStep] = field(default_factory=list)
    platform: str = ""
    
    def get_required_steps(self) -> List[ExecutionStep]:
        """Get only required steps."""
        return [s for s in self.steps if s.required]
    
    def get_optional_steps(self) -> List[ExecutionStep]:
        """Get only optional steps."""
        return [s for s in self.steps if not s.required]
    
    @property
    def total_steps(self) -> int:
        return len(self.steps)
    
    @property
    def required_count(self) -> int:
        return len(self.get_required_steps())


@dataclass
class ProfilerCheckpoint:
    """Checkpoint data for profiler execution state."""
    task_id: str
    completed_steps: List[str]  # data_types already collected
    data: Dict[str, Any]  # Partial results
    failed_steps: List[str]  # Optional steps that failed
    skipped_steps: List[str]  # Steps skipped (e.g., unavailable)
    timestamp: str = ""


class ProfilerOrchestrator:
    """Orchestrates profiler execution based on SME requirements.
    
    Key responsibilities:
    1. Map data_type requirements to profiler implementations
    2. Handle platform-specific profiler selection
    3. Sort steps by dependencies (topological order)
    4. Skip already-completed steps (checkpoint resume)
    
    Usage:
        orchestrator = ProfilerOrchestrator()
        plan = orchestrator.create_plan(
            requirements={"ncu_report": {"required": True, "extractors": [...]}},
            platform=Platform.NVIDIA_CUDA,
            checkpoint=None
        )
        
        for step in plan.steps:
            profiler = step.profiler_class()
            result = await profiler.run(context)
    """
    
    # Registry of available profilers by platform
    # Maps platform_type -> {data_type -> profiler_class}
    PROFILERS: Dict[str, Dict[str, Type[BaseProfiler]]] = {
        Platform.NVIDIA_CUDA.value: {
            "vllm_logs": VLLMLogCollector,
            "ncu_report": MockNCUProfiler,  # Use mock for testing
            "benchmark_metrics": BenchmarkMetricsCollector,
            # "ncu_report": NCUProfiler,     # Real NCU (deferred)
            # "nsys_report": NSysProfiler,   # NSys (deferred)
        },
        Platform.AMD_ROCM.value: {
            "vllm_logs": VLLMLogCollector,
            "benchmark_metrics": BenchmarkMetricsCollector,
            # "rocprof_report": RocprofProfiler,  # Deferred
        },
        Platform.CPU.value: {
            "vllm_logs": VLLMLogCollector,
            "benchmark_metrics": BenchmarkMetricsCollector,
            # "perf_report": PerfProfiler,  # Deferred
        },
        Platform.UNKNOWN.value: {
            "vllm_logs": VLLMLogCollector,
            "benchmark_metrics": BenchmarkMetricsCollector,
        },
    }
    
    def __init__(self):
        self.completed_steps: Set[str] = set()
        self.failed_steps: Set[str] = set()
        self.skipped_steps: Set[str] = set()
    
    def create_plan(
        self,
        requirements: Dict[str, Dict[str, Any]],
        platform: Platform,
        checkpoint: Optional[ProfilerCheckpoint] = None
    ) -> ExecutionPlan:
        """Create execution plan based on requirements and platform.
        
        Args:
            requirements: From SMERegistry.get_profiler_requirements()
                         Format: {"ncu_report": {"required": True, "extractors": [...]}}
            platform: Target platform (NVIDIA_CUDA, AMD_ROCM, etc.)
            checkpoint: Optional checkpoint to resume from
            
        Returns:
            ExecutionPlan with sorted steps
            
        Raises:
            UnsupportedDataTypeError: If a required data_type has no profiler
                                     for this platform
        """
        steps = []
        platform_type = platform.value if isinstance(platform, Platform) else platform
        platform_profilers = self.PROFILERS.get(platform_type, self.PROFILERS[Platform.UNKNOWN.value])
        
        # Load checkpoint state
        completed = set(checkpoint.completed_steps) if checkpoint else set()
        
        for data_type, req_info in requirements.items():
            # Skip if already completed
            if data_type in completed:
                continue
            
            profiler_class = platform_profilers.get(data_type)
            
            if not profiler_class:
                required = req_info.get("required", False)
                if required:
                    raise UnsupportedDataTypeError(
                        f"No profiler available for required data type '{data_type}' "
                        f"on platform '{platform_type}'"
                    )
                # Skip optional if no profiler
                continue
            
            steps.append(ExecutionStep(
                data_type=data_type,
                profiler_id=profiler_class.__name__,
                profiler_class=profiler_class,
                required=req_info.get("required", False),
                extractors=req_info.get("extractors", []),
                dependencies=profiler_class.DEPENDENCIES
            ))
        
        # Sort steps: required first, then by dependencies
        steps = self._sort_steps(steps)
        
        return ExecutionPlan(
            steps=steps,
            platform=platform_type
        )
    
    def _sort_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """Sort steps by priority and dependencies.
        
        Order:
        1. Required steps before optional
        2. Steps with no dependencies before steps with dependencies
        3. By estimated duration (shorter first for quick wins)
        
        Args:
            steps: Unsorted steps
            
        Returns:
            Sorted steps
        """
        def sort_key(step: ExecutionStep) -> tuple:
            # Required first (False < True, so negate)
            required_priority = 0 if step.required else 1
            
            # Fewer dependencies first
            dep_count = len(step.dependencies)
            
            # Get estimated duration
            duration = step.profiler_class.ESTIMATED_DURATION_SECONDS
            
            return (required_priority, dep_count, duration)
        
        return sorted(steps, key=sort_key)
    
    def mark_completed(self, data_type: str) -> None:
        """Mark a step as completed.
        
        Args:
            data_type: The data type that was collected
        """
        self.completed_steps.add(data_type)
    
    def mark_failed(self, data_type: str) -> None:
        """Mark a step as failed.
        
        Args:
            data_type: The data type that failed
        """
        self.failed_steps.add(data_type)
    
    def mark_skipped(self, data_type: str) -> None:
        """Mark a step as skipped.
        
        Args:
            data_type: The data type that was skipped
        """
        self.skipped_steps.add(data_type)
    
    def create_checkpoint(self, task_id: str) -> ProfilerCheckpoint:
        """Create checkpoint from current state.
        
        Args:
            task_id: Task identifier
            
        Returns:
            ProfilerCheckpoint with current state
        """
        from datetime import datetime
        
        return ProfilerCheckpoint(
            task_id=task_id,
            completed_steps=list(self.completed_steps),
            data={},  # Populated by agent
            failed_steps=list(self.failed_steps),
            skipped_steps=list(self.skipped_steps),
            timestamp=datetime.now().isoformat()
        )
    
    def get_available_data_types(self, platform: Platform) -> List[str]:
        """Get list of data types available for a platform.
        
        Args:
            platform: Target platform
            
        Returns:
            List of available data type strings
        """
        platform_type = platform.value if isinstance(platform, Platform) else platform
        platform_profilers = self.PROFILERS.get(platform_type, {})
        return list(platform_profilers.keys())
    
    def is_data_type_available(self, data_type: str, platform: Platform) -> bool:
        """Check if a data type is available on a platform.
        
        Args:
            data_type: Data type to check
            platform: Target platform
            
        Returns:
            True if a profiler is available
        """
        platform_type = platform.value if isinstance(platform, Platform) else platform
        platform_profilers = self.PROFILERS.get(platform_type, {})
        return data_type in platform_profilers
    
    def print_plan(self, plan: ExecutionPlan) -> None:
        """Print execution plan for debugging.
        
        Args:
            plan: Execution plan to print
        """
        print("\n" + "=" * 60)
        print("PROFILER EXECUTION PLAN")
        print(f"Platform: {plan.platform}")
        print("=" * 60)
        
        if not plan.steps:
            print("No steps required (all completed or no requirements)")
        else:
            for i, step in enumerate(plan.steps, 1):
                req_mark = "✅" if step.required else "☑️"
                deps = f" (deps: {', '.join(step.dependencies)})" if step.dependencies else ""
                print(f"{i}. {req_mark} {step.profiler_id:20} → {step.data_type}{deps}")
                if step.extractors:
                    print(f"   Extractors: {', '.join(step.extractors[:3])}", end="")
                    if len(step.extractors) > 3:
                        print(f" (+{len(step.extractors) - 3} more)", end="")
                    print()
        
        print(f"\nTotal: {plan.total_steps} steps ({plan.required_count} required)")
        print("=" * 60 + "\n")
