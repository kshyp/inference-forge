# Profiler Agent (Agent 2) Design

**Status:** Interface Design Complete  
**Last Updated:** 2026-03-15

---

## Overview

The Profiler Agent collects performance data required by registered SMEs. It receives a consolidated list of data requirements from the SME Registry and runs the necessary profilers to satisfy them.

**Key Challenge:** Profiling tools (NCU, NSys) are complex, require specific hardware, and produce varied output formats. The Profiler Agent must abstract this complexity behind a clean interface.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROFILER AGENT (Agent 2)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │   Agent 2    │────▶│  Profiler    │────▶│ Data Source  │            │
│  │   Core       │     │  Orchestrator│     │  Interfaces  │            │
│  │              │     │              │     │              │            │
│  │ - Receive    │     │ - Determine  │     │ - NCU        │            │
│  │   task       │     │   which      │     │ - NSys       │            │
│  │ - Checkpoint │     │   profilers  │     │ - vLLM Logs  │            │
│  │ - Handle     │     │   to run     │     │ - ROCm       │            │
│  │   errors     │     │ - Sequence   │     │ - CPU        │            │
│  │ - Return     │     │   execution  │     │              │            │
│  │   results    │     │ - Aggregate  │     │              │            │
│  │              │     │   results    │     │              │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│         │                    │                    │                     │
│         │              ┌─────┴─────┐             │                     │
│         │              ▼           ▼             │                     │
│         │         ┌────────┐   ┌────────┐       │                     │
│         │         │  NCU   │   │  NSys  │       │                     │
│         │         │Profiler│   │Profiler│       │                     │
│         │         └────────┘   └────────┘       │                     │
│         │                                       │                     │
│         └───────────────────────────────────────┘                     │
│                         Extractors                                    │
│                    (Parse outputs → structured data)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Interfaces

### 1. ProfilerAgent (Agent 2 Entry Point)

```python
class ProfilerAgent(BaseAgent):
    """
    Agent 2: Collects profiling data required by SMEs.
    """
    
    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.orchestrator = ProfilerOrchestrator()
        self.checkpoint_data: Optional[ProfilerCheckpoint] = None
    
    async def run_task(self, task: ProfileTask) -> ProfilingResult:
        """
        Main entry point. Called by orchestrator after Agent 1 completes.
        
        Args:
            task: Contains benchmark_result + data_requirements from SME Registry
            
        Returns:
            ProfilingResult with collected data keyed by data_type
        """
        # 1. Load checkpoint if resuming
        if task.resume_from_checkpoint:
            self.checkpoint_data = await self.load_checkpoint(task.task_id)
        
        # 2. Determine which profilers to run
        execution_plan = self.orchestrator.create_plan(
            requirements=task.data_requirements,
            platform=task.platform_info,
            checkpoint=self.checkpoint_data
        )
        
        # 3. Run profilers (with checkpointing after each)
        collected_data: Dict[str, Any] = {}
        
        for step in execution_plan.steps:
            if step.data_type in collected_data:
                continue  # Already collected (from checkpoint)
            
            try:
                result = await self._run_profiler_step(step, task)
                collected_data[step.data_type] = result
                
                # Checkpoint after each profiler
                await self.checkpoint({
                    "completed_steps": list(collected_data.keys()),
                    "data": collected_data
                })
                
            except ProfilerError as e:
                if step.required:
                    raise  # Required profiler failed - abort
                else:
                    self.logger.warning(f"Optional profiler {step.profiler_id} failed: {e}")
                    collected_data[step.data_type] = None  # Mark as unavailable
        
        # 4. Verify we have all required data
        self._verify_completeness(task.data_requirements, collected_data)
        
        # 5. Return structured result
        return ProfilingResult(
            experiment_id=task.experiment_id,
            platform=task.platform_info,
            reports=collected_data,
            benchmark_result=task.benchmark_result
        )
```

### 2. ProfilerOrchestrator

```python
class ProfilerOrchestrator:
    """
    Decides which profilers to run and in what order.
    """
    
    # Registry of available profilers by platform
    PROFILERS: Dict[str, Dict[str, Type[BaseProfiler]]] = {
        "nvidia_cuda": {
            "ncu_report": NCUProfiler,
            "nsys_report": NSysProfiler,
            "vllm_logs": VLLMLogCollector,
        },
        "amd_rocm": {
            "rocprof_report": RocprofProfiler,
            "vllm_logs": VLLMLogCollector,
        },
        "cpu": {
            "perf_report": PerfProfiler,
            "vllm_logs": VLLMLogCollector,
        }
    }
    
    def create_plan(self, 
                    requirements: Dict[str, DataRequirement],
                    platform: PlatformInfo,
                    checkpoint: Optional[ProfilerCheckpoint]) -> ExecutionPlan:
        """
        Create execution plan based on requirements and platform.
        
        Strategy:
        - Run independent profilers in parallel where possible
        - Run dependent profilers sequentially
        - Skip profilers already completed (from checkpoint)
        - Skip optional profilers if dependencies missing
        """
        steps = []
        platform_profilers = self.PROFILERS.get(platform.type, {})
        
        for data_type, req in requirements.items():
            profiler_class = platform_profilers.get(data_type)
            
            if not profiler_class:
                if req.required:
                    raise UnsupportedDataTypeError(f"No profiler for {data_type} on {platform.type}")
                continue  # Skip optional if no profiler available
            
            # Check if already completed
            if checkpoint and data_type in checkpoint.completed_steps:
                continue
            
            steps.append(ExecutionStep(
                data_type=data_type,
                profiler_id=profiler_class.__name__,
                required=req.required,
                extractors=req.extractors,
                dependencies=profiler_class.DEPENDENCIES  # e.g., nsys needs vLLM running
            ))
        
        # Sort: required first, then by dependencies
        steps = self._topological_sort(steps)
        
        return ExecutionPlan(steps=steps)
```

### 3. BaseProfiler Interface

```python
class BaseProfiler(ABC):
    """
    Abstract base for all profilers (NCU, NSys, etc.).
    """
    
    # Dependencies that must be running before this profiler
    DEPENDENCIES: List[str] = []
    
    # Estimated runtime (for progress reporting)
    ESTIMATED_DURATION_SECONDS: int = 60
    
    @abstractmethod
    async def run(self, 
                  context: ProfilingContext) -> RawProfilerOutput:
        """
        Execute the profiler.
        
        Args:
            context: Contains vLLM PID, duration, config, etc.
            
        Returns:
            Raw output (file paths, stdout, etc.)
        """
        pass
    
    @abstractmethod
    async def extract(self,
                      raw_output: RawProfilerOutput,
                      extractors: List[str]) -> Dict[str, Any]:
        """
        Parse raw output and extract requested metrics.
        
        Args:
            raw_output: Output from run()
            extractors: List of metrics to extract
            
        Returns:
            Dict mapping extractor name to value
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this profiler can run on this system."""
        pass


@dataclass
class ProfilingContext:
    """Context passed to profilers during execution."""
    vllm_pid: Optional[int]
    vllm_log_path: Optional[Path]
    duration_seconds: int
    warmup_seconds: int
    output_dir: Path
    config: Dict[str, Any]  # vLLM config flags
```

### 4. Concrete Profiler Examples

```python
class NCUProfiler(BaseProfiler):
    """
    NVIDIA Compute Profiler (ncu).
    
    Complexity: HIGH
    - Requires root or specific permissions
    - Can only profile one process at a time
    - Significant overhead
    - Output format varies by ncu version
    """
    
    DEPENDENCIES = ["vllm_running"]  # NCU needs a running process
    ESTIMATED_DURATION_SECONDS = 120
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """
        Run NCU on the vLLM process.
        
        Command example:
        ncu --target-processes all \
            --profile-from-start off \
            --launch-count 100 \
            --export /output/profile.ncu-rep \
            --metrics dram__bytes_read,gpu__time_duration \
            python -c "trigger_workload()"
        """
        output_path = context.output_dir / "profile.ncu-rep"
        
        cmd = [
            "ncu",
            "--target-processes", "all",
            "--profile-from-start", "off",
            "--launch-count", str(self._calculate_launch_count(context)),
            "--export", str(output_path),
            "--metrics", self._select_metrics(context),
            # ... trigger workload
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise ProfilerError(f"NCU failed: {stderr.decode()}")
        
        return RawProfilerOutput(
            report_path=output_path,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    async def extract(self,
                      raw_output: RawProfilerOutput,
                      extractors: List[str]) -> Dict[str, Any]:
        """
        Parse NCU report and extract metrics.
        
        Challenge: NCU output format varies by version.
        Options:
        1. ncu --print-summary (text parsing)
        2. ncu --page raw (CSV-like)
        3. ncu-rep file (SQLite, requires ncu-ui)
        
        We use option 2 for reliability.
        """
        # Run ncu again to get CSV output
        cmd = [
            "ncu",
            "--import", str(raw_output.report_path),
            "--page", "raw",
            "--csv"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await process.communicate()
        
        # Parse CSV and extract requested metrics
        results = {}
        csv_data = stdout.decode()
        
        for extractor in extractors:
            value = self._extract_metric(csv_data, extractor)
            results[extractor] = value
        
        return results
    
    def is_available(self) -> bool:
        """Check if ncu binary exists."""
        return shutil.which("ncu") is not None


class NSysProfiler(BaseProfiler):
    """
    NVIDIA Systems Profiler (nsys).
    
    Complexity: MEDIUM
    - Easier to use than NCU
    - Timeline view very useful
    - Less detailed than NCU for kernel-level
    """
    
    DEPENDENCIES = []
    ESTIMATED_DURATION_SECONDS = 60
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Run nsys profile."""
        output_path = context.output_dir / "profile.nsys-rep"
        
        cmd = [
            "nsys", "profile",
            "--duration", str(context.duration_seconds),
            "--output", str(output_path),
            "--trace", "cuda,nvtx,osrt",
            # Attach to vLLM or run new instance
        ]
        
        # ... similar to NCU
        
        return RawProfilerOutput(report_path=output_path)
    
    async def extract(self, raw_output, extractors):
        """
        Extract metrics from nsys report.
        
        Uses nsys stats or nsys-rep parsing.
        """
        # nsys stats --report cuda_sum <file>
        pass


class VLLMLogCollector(BaseProfiler):
    """
    Collects and parses vLLM logs.
    
    Complexity: LOW
    - No special permissions needed
    - Parse text logs or use structured logging
    - Can collect from running instance
    """
    
    DEPENDENCIES = []
    ESTIMATED_DURATION_SECONDS = 5
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Read vLLM log file or fetch from API."""
        if context.vllm_log_path:
            log_content = await self._read_log_file(context.vllm_log_path)
        else:
            # Fetch from vLLM API or shared memory
            log_content = await self._fetch_logs_from_api(context)
        
        return RawProfilerOutput(
            log_content=log_content,
            report_path=context.vllm_log_path
        )
    
    async def extract(self, raw_output, extractors) -> Dict[str, Any]:
        """
        Parse log content and extract metrics.
        
        Example extractors:
        - "max_num_seqs" → parse "max_num_seqs=64" from startup
        - "scheduler_queue_depth" → parse "[sched] 5 requests waiting"
        - "kv_cache_usage_percent" → parse from periodic stats
        """
        results = {}
        
        for extractor in extractors:
            pattern = self.LOG_PATTERNS.get(extractor)
            if pattern:
                match = re.search(pattern, raw_output.log_content)
                results[extractor] = match.group(1) if match else None
        
        return results
```

---

## Error Handling Strategy

```python
class ProfilerError(Exception):
    """Base profiler error."""
    pass

class ProfilerNotAvailableError(ProfilerError):
    """Profiler binary not found or permissions issue."""
    pass

class ProfilerTimeoutError(ProfilerError):
    """Profiler took too long."""
    pass

class ExtractorError(ProfilerError):
    """Failed to extract specific metric."""
    pass

class DataQualityError(ProfilerError):
    """Data collected but seems invalid (e.g., all zeros)."""
    pass


# Handling in Agent 2
async def _run_profiler_step(self, step, task):
    profiler = self._get_profiler(step.profiler_id)
    
    # Check availability
    if not profiler.is_available():
        if step.required:
            raise ProfilerNotAvailableError(f"{step.profiler_id} not available but required")
        else:
            self.logger.warning(f"{step.profiler_id} not available, skipping optional")
            return None
    
    try:
        # Run with timeout
        raw_output = await asyncio.wait_for(
            profiler.run(self.context),
            timeout=profiler.ESTIMATED_DURATION_SECONDS * 2
        )
        
        # Extract metrics
        extracted = await profiler.extract(raw_output, step.extractors)
        
        # Validate data quality
        self._validate_data_quality(extracted, step)
        
        return extracted
        
    except asyncio.TimeoutError:
        raise ProfilerTimeoutError(f"{step.profiler_id} timed out")
    except Exception as e:
        raise ProfilerError(f"{step.profiler_id} failed: {e}") from e
```

---

## Checkpointing Strategy

```python
@dataclass
class ProfilerCheckpoint:
    """Checkpoint data for Profiler Agent."""
    task_id: str
    completed_steps: List[str]  # data_types already collected
    data: Dict[str, Any]        # Partial results
    failed_steps: List[str]     # Optional steps that failed
    timestamp: datetime


# Resume workflow
async def resume_from_checkpoint(self, checkpoint: ProfilerCheckpoint):
    """Resume interrupted profiling."""
    self.logger.info(f"Resuming from checkpoint, already have: {checkpoint.completed_steps}")
    
    # Create plan skipping completed steps
    plan = self.orchestrator.create_plan(
        requirements=self.task.data_requirements,
        platform=self.task.platform_info,
        checkpoint=checkpoint
    )
    
    # Continue with remaining steps
    for step in plan.steps:
        # ... run as normal
```

---

## Mock/Test Implementations

```python
class MockNCUProfiler(NCUProfiler):
    """Mock NCU for testing without GPU."""
    
    async def run(self, context):
        # Return fake data instead of running ncu
        return RawProfilerOutput(
            report_path=Path("/fake/profile.ncu-rep"),
            is_mock=True
        )
    
    async def extract(self, raw_output, extractors):
        # Return synthetic metrics
        import random
        return {
            "memory_bandwidth_utilization_percent": random.uniform(60, 95),
            "compute_utilization_percent": random.uniform(40, 80),
            # ... other metrics
        }


# Usage in tests
ProfilerOrchestrator.PROFILERS["nvidia_cuda"]["ncu_report"] = MockNCUProfiler
```

---

## Open Questions

| Question | Status | Notes |
|----------|--------|-------|
| NCU metric selection | Open | How to map extractors to NCU `--metrics` flags? |
| NSys vs NCU overlap | Open | Run both or choose based on requirements? |
| vLLM log streaming | Open | Tail log file or use structured logging? |
| Permission handling | Open | NCU needs root - document or workaround? |
| Real-time profiling | Open | Support continuous profiling or only one-shot? |

---

## Next Steps

1. **Implement VLLMLogCollector** (lowest complexity)
2. **Implement MockNCUProfiler** for testing
3. **Create profiler configuration** (metric mappings)
4. **Implement error handling** and retry logic
5. **Add checkpoint/restore** tests
6. **Document NCU setup** (permissions, installation)

**Deferred:**
- Real NCUProfiler (complex, hardware-dependent)
- Real NSysProfiler (less critical than NCU)
- ROCm/CPU profilers
