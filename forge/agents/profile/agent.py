"""Agent 2: Profile Executor.

Collects profiling data required by registered SMEs.
Runs at the saturation point identified by the Benchmark Agent.

Key difference from BenchmarkAgent:
- ProfilerAgent starts vLLM with DEBUG logging (higher overhead)
- Runs steady-state benchmark AT the saturation point
- Attaches profilers (NCU, NSys) during the run
- Collects detailed logs for SME analysis
"""

import asyncio
import json
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from forge.agents.base import BaseAgent
from forge.core.events import (
    BenchmarkMetrics,
    BenchmarkResult,
    HardwareInfo,
    MetricsDistribution,
    Platform,
    ProfilingResult,
    SaturationPoint,
    Task,
)
from forge.core.state import StateStore
from forge.smes.base import DataRequirement

from .orchestrator import ExecutionStep, ProfilerCheckpoint, ProfilerOrchestrator
from .profilers.base import (
    BaseProfiler,
    DataQualityError,
    ProfilingContext,
    ProfilerError,
    ProfilerNotAvailableError,
    ProfilerResult,
    ProfilerTimeoutError,
)
from .verifier import ReportVerifier


@dataclass
class ProfileTask:
    """Task payload for ProfilerAgent."""
    experiment_id: str
    benchmark_result: BenchmarkResult  # Contains saturation point from Agent 1
    data_requirements: Dict[str, Dict[str, Any]]  # From SMERegistry
    platform_info: Platform
    output_dir: Path
    # GuideLLM config for steady-state run
    guidellm_config: Dict[str, Any] = field(default_factory=dict)
    vllm_config: Dict[str, Any] = field(default_factory=dict)
    resume_from_checkpoint: bool = True


class ProfilerAgent(BaseAgent):
    """Agent 2: Collects profiling data required by SMEs.
    
    Workflow:
    1. Receive task with saturation point from BenchmarkAgent
    2. Start vLLM with DEBUG logging enabled
    3. Run steady-state benchmark AT saturation rate
    4. Collect debug logs + run profilers (NCU/NSys)
    5. Return ProfilingResult to Coordinator
    
    Key differences from BenchmarkAgent:
    - Uses DEBUG logging (not INFO)
    - Runs AT saturation point (not sweep)
    - Attaches hardware profilers (NCU/NSys)
    - Higher overhead acceptable (we're profiling, not benchmarking)
    
    Example:
        agent = ProfilerAgent(state_store, health_port=8082)
        task = Task(
            type="profile",
            payload={
                "experiment_id": "uuid",
                "benchmark_result": {...},  # From Agent 1
                "data_requirements": {"ncu_report": {"required": True, ...}},
                "platform_info": "nvidia_cuda",
                "output_dir": "./data/profiles",
                "saturation_rate": 12.5,  # From Agent 1
            }
        )
        result = await agent.run_single_task(task)
    """
    
    def __init__(
        self,
        state_store: StateStore,
        health_port: int = 8082,
        data_dir: str = "./data",
        autotuner_dir: str = "./scripts/autotuner",
    ):
        super().__init__(
            agent_id="profile",
            state_store=state_store,
            health_port=health_port
        )
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.autotuner_dir = Path(autotuner_dir)
        
        # Components
        self.orchestrator = ProfilerOrchestrator()
        self.verifier = ReportVerifier(strict=False)
        
        # Runtime state (checkpointed)
        self.phase: str = "idle"  # idle, running, profiling, cleanup
        self.collected_data: Dict[str, Dict[str, Any]] = {}
        self.completed_profilers: List[str] = []
        self.failed_profilers: List[str] = []
        self.skipped_profilers: List[str] = []
        self.current_step: Optional[str] = None
        
        # vLLM process (managed by this agent)
        self.vllm_process: Optional[asyncio.subprocess.Process] = None
        self.vllm_pid: Optional[int] = None
        
        # Task queue
        self._task_queue: asyncio.Queue = asyncio.Queue()
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next profiling task from queue."""
        try:
            return await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def submit_task(self, task: Task) -> None:
        """Submit a profiling task."""
        self._task_queue.put_nowait(task)
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Execute profiling task.
        
        Steps:
        1. Parse task payload (saturation point from Agent 1)
        2. Start vLLM with DEBUG logging
        3. Run steady-state benchmark at saturation rate
        4. Collect logs and run profilers
        5. Stop vLLM
        6. Return ProfilingResult
        
        Args:
            task: Task with ProfileTask payload
            
        Returns:
            Dict with ProfilingResult data
        """
        # Parse task payload
        profile_task = self._parse_task_payload(task)
        
        # Create output directory
        exp_dir = profile_task.output_dir / profile_task.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = exp_dir / "vllm_server_debug.log"
        
        try:
            # Phase 1: Start vLLM with DEBUG logging
            await self.log_event("phase_started", task, {"phase": "start_vllm_debug"})
            
            saturation_rate = profile_task.benchmark_result.saturation_point.rps
            print(f"\n🚀 Starting vLLM with DEBUG logging for profiling...")
            print(f"   Saturation rate: {saturation_rate:.2f} RPS")
            print(f"   Log level: DEBUG")
            
            await self._start_vllm_debug(
                model_name=profile_task.vllm_config.get("model", "gpt2"),
                port=profile_task.vllm_config.get("port", 8000),
                log_file=log_file,
                vllm_config=profile_task.vllm_config,
                task=task,
            )
            
            await self.log_event("phase_completed", task, {"phase": "start_vllm_debug"})
            
            # Phase 2: Run steady-state benchmark at saturation rate
            await self.log_event("phase_started", task, {"phase": "steady_state_profile"})
            
            steady_file = exp_dir / "steady_state_profile.json"
            benchmark_metrics = await self._run_steady_state_profile(
                model_name=profile_task.vllm_config.get("model", "gpt2"),
                rate=saturation_rate,
                port=profile_task.vllm_config.get("port", 8000),
                steady_file=steady_file,
                guidellm_config=profile_task.guidellm_config,
                exp_dir=exp_dir,
                task=task,
            )
            
            await self.log_event("phase_completed", task, {"phase": "steady_state_profile", "metrics": benchmark_metrics})
            
            # Phase 3: Collect profiling data from logs
            await self.log_event("phase_started", task, {"phase": "collect_profiling_data"})
            
            # Create profiling context
            context = ProfilingContext(
                vllm_pid=self.vllm_pid,
                vllm_log_path=log_file,
                vllm_port=profile_task.vllm_config.get("port", 8000),
                output_dir=exp_dir,
                experiment_id=profile_task.experiment_id,
                model_name=profile_task.vllm_config.get("model", ""),
                config=profile_task.vllm_config,
                saturation_rate=saturation_rate,
            )
            
            # Create execution plan
            checkpoint = None
            if profile_task.resume_from_checkpoint:
                checkpoint_data = await self.load_checkpoint(task.id)
                if checkpoint_data:
                    checkpoint = self._dict_to_checkpoint(checkpoint_data)
                    self._restore_state(checkpoint_data)
            
            plan = self.orchestrator.create_plan(
                requirements=profile_task.data_requirements,
                platform=profile_task.platform_info,
                checkpoint=checkpoint
            )
            
            self.orchestrator.print_plan(plan)
            
            if not plan.steps:
                return {
                    "success": True,
                    "message": "No profiling steps required",
                    "profiling_result": self._create_result(
                        profile_task, context, plan
                    ).__dict__,
                }
            
            # Execute profilers
            total_steps = len(plan.steps)
            
            for i, step in enumerate(plan.steps):
                self.current_step = step.data_type
                progress = (i / total_steps) * 100
                self.update_progress(progress)
                
                await self.log_event("profiler_started", task, {
                    "data_type": step.data_type,
                    "profiler": step.profiler_id,
                    "required": step.required,
                })
                
                try:
                    result = await self._run_profiler_step(step, context, profile_task)
                    
                    if result:
                        self.collected_data[step.data_type] = result.extractors
                        self.completed_profilers.append(step.data_type)
                        self.orchestrator.mark_completed(step.data_type)
                        
                        await self.log_event("profiler_completed", task, {
                            "data_type": step.data_type,
                            "extractors_count": len(result.extractors),
                        })
                    else:
                        self.skipped_profilers.append(step.data_type)
                        self.orchestrator.mark_skipped(step.data_type)
                        
                except Exception as e:
                    self.failed_profilers.append(step.data_type)
                    self.orchestrator.mark_failed(step.data_type)
                    
                    await self.log_event("profiler_failed", task, {
                        "data_type": step.data_type,
                        "error": str(e),
                        "required": step.required,
                    })
                    
                    if step.required:
                        raise
                
                # Checkpoint after each profiler
                await self.checkpoint(task.id)
            
            await self.log_event("phase_completed", task, {"phase": "collect_profiling_data"})
            
            # Final progress
            self.update_progress(100)
            self.current_step = None
            
            # Verify completeness
            is_complete, missing = self.verifier.verify_completeness(
                self.collected_data,
                profile_task.data_requirements
            )
            
            if not is_complete:
                raise DataQualityError(
                    f"Required data types not collected: {', '.join(missing)}"
                )
            
            # Create final result
            profiling_result = self._create_result(
                profile_task, context, plan
            )
            
            return {
                "success": True,
                "profiling_result": self._profiling_result_to_dict(profiling_result),
                "steady_state_file": str(steady_file),
                "benchmark_metrics": benchmark_metrics,
                "collected_data": self.collected_data,
            }
        
        except Exception as e:
            # Return partial results even if profiling failed
            print(f"   Profile execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "profiling_result": {},
                "steady_state_file": str(steady_file) if 'steady_file' in locals() else "",
                "benchmark_metrics": benchmark_metrics if 'benchmark_metrics' in locals() else {},
                "collected_data": self.collected_data,
            }
            
        finally:
            # Cleanup - always stop vLLM
            await self._cleanup_vllm()
            self.phase = "idle"
    
    async def _start_vllm_debug(
        self,
        model_name: str,
        port: int,
        log_file: Path,
        vllm_config: Dict[str, Any],
        task: Task,
    ) -> None:
        """Start vLLM server with DEBUG logging enabled.
        
        This is the key difference from BenchmarkAgent - we use DEBUG level
        to get detailed logs for SME analysis.
        """
        # Build vLLM command with DEBUG logging
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--uvicorn-log-level", "debug",  # Key: DEBUG logging for detailed metrics
        ]
        
        # Add additional config
        if "max_num_seqs" in vllm_config:
            cmd.extend(["--max-num-seqs", str(vllm_config["max_num_seqs"])])
        if "max_model_len" in vllm_config:
            cmd.extend(["--max-model-len", str(vllm_config["max_model_len"])])
        if "tensor_parallel_size" in vllm_config:
            cmd.extend(["--tensor-parallel-size", str(vllm_config["tensor_parallel_size"])])
        if "gpu_memory_utilization" in vllm_config:
            cmd.extend(["--gpu-memory-utilization", str(vllm_config["gpu_memory_utilization"])])
        if "quantization" in vllm_config:
            cmd.extend(["--quantization", vllm_config["quantization"]])
        
        print(f"   Command: {' '.join(cmd[:8])}...")
        
        # Start process
        self.vllm_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.vllm_pid = self.vllm_process.pid
        
        # Checkpoint
        await self.checkpoint(task.id)
        
        # Wait for server to be ready (increased timeout for larger models)
        timeout = vllm_config.get("startup_timeout_seconds", 180)
        try:
            await self._wait_for_vllm_ready(port, timeout=timeout, log_file=log_file)
        except TimeoutError as e:
            # Print last lines of log for debugging
            print(f"\n   ❌ vLLM startup failed/timed out after {timeout}s")
            print(f"   Last 20 lines of log:")
            print(f"   {'-'*60}")
            try:
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-20:]:
                            print(f"   {line.rstrip()}")
            except Exception as log_err:
                print(f"   (Could not read log: {log_err})")
            print(f"   {'-'*60}")
            raise e
        
        print(f"   ✓ vLLM started (PID: {self.vllm_pid}) with DEBUG logging")
    
    async def _wait_for_vllm_ready(self, port: int, timeout: float = 120, log_file: Path = None) -> None:
        """Wait for vLLM server health endpoint."""
        import aiohttp
        
        start = asyncio.get_event_loop().time()
        check_count = 0
        last_error = None
        
        while (asyncio.get_event_loop().time() - start) < timeout:
            check_count += 1
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            return
            except Exception as e:
                last_error = str(e)
            
            # Print progress every 30 seconds
            elapsed = asyncio.get_event_loop().time() - start
            if check_count % 30 == 0:
                print(f"   ... still waiting for vLLM ({elapsed:.0f}s elapsed, last error: {last_error or 'none'})")
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"vLLM server not ready after {timeout}s (last error: {last_error or 'connection refused'})")
    
    async def _run_steady_state_profile(
        self,
        model_name: str,
        rate: float,
        port: int,
        steady_file: Path,
        guidellm_config: Dict[str, Any],
        exp_dir: Path,
        task: Task,
    ) -> Dict[str, float]:
        """Run steady-state benchmark at saturation rate for profiling.
        
        This runs at the saturation point identified by Agent 1.
        The difference: we collect DEBUG logs during this run.
        
        Returns:
            Dict with extracted benchmark metrics
        """
        # Construct data config
        dataset = guidellm_config.get("dataset", "wikitext")
        prompt_tokens = guidellm_config.get("prompt_tokens", 512)
        output_tokens = guidellm_config.get("output_tokens", 128)
        max_requests = guidellm_config.get("max_requests", 1000)
        duration = guidellm_config.get("duration_seconds", 60)  # Run long enough for profiling
        
        print(f"\n📊 Running steady-state benchmark at {rate:.2f} RPS for profiling...")
        print(f"   Duration: {duration}s")
        print(f"   Max requests: {max_requests}")
        
        # Build guidellm command for steady-state run
        cmd = [
            "guidellm", "benchmark", "run",
            "--target", f"http://localhost:{port}",
            "--model", model_name,
            "--profile", "constant",
            "--rate", str(int(rate)),
            "--data", f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
            "--request-type", "text_completions",
            "--output-path", str(steady_file),
            "--max-requests", str(max_requests),
            "--max-seconds", str(duration),
        ]
        
        print(f"   Starting GuideLLM...")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            print(f"GuideLLM stdout: {stdout.decode()}")
            print(f"GuideLLM stderr: {stderr.decode()}")
            raise RuntimeError(f"Steady-state profiling failed: {stderr.decode()}")
        
        print(f"   ✓ Steady-state profiling complete")
        print(f"   Results saved to: {steady_file}")
        
        # Parse and return metrics
        return self._extract_metrics_from_file(steady_file)
    
    def _extract_metrics_from_file(self, steady_file: Path) -> Dict[str, float]:
        """Extract metrics from GuideLLM output file."""
        if not steady_file.exists():
            return {"throughput": 0, "ttft_p99": 0, "tpot_p99": 0, "latency_p99": 0, "error_rate": 0}
        
        try:
            with open(steady_file) as f:
                data = json.load(f)
            
            if "benchmarks" in data and data["benchmarks"]:
                metrics = data["benchmarks"][0].get("metrics", {})
            else:
                metrics = data.get("metrics", {})
            
            # Extract throughput
            rps_data = metrics.get("requests_per_second", {}).get("successful", {})
            throughput = rps_data.get("mean", 0) if isinstance(rps_data, dict) else 0
            
            # Extract TTFT
            ttft_data = metrics.get("time_to_first_token_ms", {})
            ttft_p99 = ttft_data.get("percentiles", {}).get("p99", 0) if isinstance(ttft_data, dict) else 0
            if ttft_p99 == 0:
                ttft_p99 = ttft_data.get("mean", 0) if isinstance(ttft_data, dict) else 0
            ttft_p99 = ttft_p99 * 1000 if ttft_p99 < 10 else ttft_p99
            
            # Extract TPOT
            tpot_data = metrics.get("time_per_output_token_ms", {})
            tpot_p99 = tpot_data.get("percentiles", {}).get("p99", 0) if isinstance(tpot_data, dict) else 0
            if tpot_p99 == 0:
                tpot_p99 = tpot_data.get("mean", 0) if isinstance(tpot_data, dict) else 0
            tpot_p99 = tpot_p99 * 1000 if tpot_p99 < 10 else tpot_p99
            
            # Extract latency
            latency_data = metrics.get("request_latency", {})
            lat_key = "successful" if "successful" in latency_data else None
            if lat_key:
                latency_data = latency_data[lat_key]
            latency_p99 = latency_data.get("percentiles", {}).get("p99", 0) if isinstance(latency_data, dict) else 0
            if latency_p99 == 0:
                latency_p99 = latency_data.get("mean", 0) if isinstance(latency_data, dict) else 0
            latency_p99 = latency_p99 * 1000 if latency_p99 < 10 else latency_p99
            
            # Extract error rate
            totals = metrics.get("request_totals", {})
            total = totals.get("total", 0)
            errored = totals.get("errored", 0)
            error_rate = errored / total if total > 0 else 0
            
            return {
                "throughput": throughput,
                "ttft_p99": ttft_p99,
                "tpot_p99": tpot_p99,
                "latency_p99": latency_p99,
                "error_rate": error_rate,
            }
        except Exception as e:
            print(f"   Warning: Could not parse metrics: {e}")
            return {"throughput": 0, "ttft_p99": 0, "tpot_p99": 0, "latency_p99": 0, "error_rate": 0}
    
    async def _cleanup_vllm(self) -> None:
        """Stop vLLM server."""
        if self.vllm_process:
            import os
            
            try:
                os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)
                await asyncio.wait_for(self.vllm_process.wait(), timeout=10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            
            self.vllm_process = None
            self.vllm_pid = None
            print(f"   ✓ vLLM stopped")
    
    def _parse_task_payload(self, task: Task) -> ProfileTask:
        """Parse task payload into ProfileTask."""
        payload = task.payload
        
        benchmark_data = payload["benchmark_result"]
        
        # Parse SaturationPoint
        sp_data = benchmark_data["saturation_point"]
        saturation_point = SaturationPoint(
            rps=sp_data["rps"],
            latency_ms=sp_data.get("latency_ms", 0),
            error_rate=sp_data.get("error_rate", 0.0),
            confidence=sp_data.get("confidence", "medium"),
        )
        
        # Parse HardwareInfo
        hw_data = benchmark_data.get("hardware_info", {})
        hardware_info = HardwareInfo(
            gpu_count=hw_data.get("gpu_count", 1),
            gpu_type=hw_data.get("gpu_type", "Unknown"),
            gpu_memory_gb=hw_data.get("gpu_memory_gb", 0),
            driver_version=hw_data.get("driver_version", "unknown"),
            cuda_version=hw_data.get("cuda_version"),
            platform=Platform(hw_data.get("platform", "unknown")),
        )
        
        # Create minimal BenchmarkResult (metrics will be filled by profiling)
        benchmark_result = BenchmarkResult(
            experiment_id=UUID(benchmark_data["experiment_id"]),
            config_flags=benchmark_data.get("config_flags", {}),
            saturation_point=saturation_point,
            metrics=BenchmarkMetrics(
                ttft_ms=MetricsDistribution(p50=0, p95=0, p99=0),
                tpot_ms=MetricsDistribution(p50=0, p95=0, p99=0),
                itl_ms=MetricsDistribution(p50=0, p95=0, p99=0),
                output_tokens_per_sec=0.0,
                total_tokens_per_sec=0.0,
                requests_per_sec=0.0,
                duration_seconds=0.0,
                total_requests=0,
                failed_requests=0,
            ),
            guidellm_report_path=Path(benchmark_data.get("guidellm_report_path", "")),
            vllm_logs_path=Path(benchmark_data.get("vllm_logs_path", "")),
            hardware_info=hardware_info,
            vllm_command=benchmark_data.get("vllm_command", ""),
        )
        
        return ProfileTask(
            experiment_id=payload["experiment_id"],
            benchmark_result=benchmark_result,
            data_requirements=payload["data_requirements"],
            platform_info=Platform(payload.get("platform_info", "unknown")),
            output_dir=Path(payload["output_dir"]),
            guidellm_config=payload.get("guidellm_config", {}),
            vllm_config=payload.get("vllm_config", {}),
            resume_from_checkpoint=payload.get("resume_from_checkpoint", True),
        )
    
    async def _run_profiler_step(
        self,
        step: ExecutionStep,
        context: ProfilingContext,
        profile_task: ProfileTask
    ) -> Optional[ProfilerResult]:
        """Run a single profiler step."""
        profiler = step.profiler_class()
        
        if not profiler.is_available():
            if step.required:
                raise ProfilerNotAvailableError(
                    f"{step.profiler_id} is required but not available"
                )
            return None
        
        print(f"   Running {step.profiler_id}...")
        
        try:
            raw_output = await asyncio.wait_for(
                profiler.run(context),
                timeout=profiler.ESTIMATED_DURATION_SECONDS * 2
            )
        except asyncio.TimeoutError:
            raise ProfilerTimeoutError(f"{step.profiler_id} timed out")
        
        extracted = await profiler.extract(raw_output, step.extractors)
        
        try:
            self.verifier.verify(extracted, step.extractors, step.data_type)
        except DataQualityError as e:
            if step.required:
                raise
            print(f"   Warning: Data quality issue in {step.data_type}: {e}")
        
        return ProfilerResult(
            data_type=step.data_type,
            extractors=extracted,
            raw_output=raw_output,
            success=True
        )
    
    def _create_result(
        self,
        profile_task: ProfileTask,
        context: ProfilingContext,
        plan: Any
    ) -> ProfilingResult:
        """Create ProfilingResult from collected data."""
        reports = {}
        for data_type, data in self.collected_data.items():
            reports[data_type] = context.output_dir / f"{data_type}.json"
        
        return ProfilingResult(
            experiment_id=UUID(profile_task.experiment_id),
            platform=profile_task.platform_info,
            reports=reports,
            benchmark_result=profile_task.benchmark_result,
            profilers_attempted=[step.data_type for step in plan.steps],
            profilers_succeeded=self.completed_profilers,
            profilers_failed=self.failed_profilers,
        )
    
    def _profiling_result_to_dict(self, result: ProfilingResult) -> Dict[str, Any]:
        """Convert ProfilingResult to dict for JSON serialization."""
        return {
            "experiment_id": str(result.experiment_id),
            "platform": result.platform.value if isinstance(result.platform, Platform) else result.platform,
            "reports": {k: str(v) for k, v in result.reports.items()},
            "profilers_attempted": result.profilers_attempted,
            "profilers_succeeded": result.profilers_succeeded,
            "profilers_failed": result.profilers_failed,
            "created_at": result.created_at.isoformat() if hasattr(result, 'created_at') else None,
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "collected_data": self.collected_data,
            "completed_profilers": self.completed_profilers,
            "failed_profilers": self.failed_profilers,
            "skipped_profilers": self.skipped_profilers,
            "current_step": self.current_step,
            "current_progress": self.current_progress,
            "vllm_pid": self.vllm_pid,
        }
    
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._restore_state(data)
    
    def _restore_state(self, data: Dict[str, Any]) -> None:
        """Internal state restoration."""
        self.collected_data = data.get("collected_data", {})
        self.completed_profilers = data.get("completed_profilers", [])
        self.failed_profilers = data.get("failed_profilers", [])
        self.skipped_profilers = data.get("skipped_profilers", [])
        self.current_step = data.get("current_step")
        self.current_progress = data.get("current_progress", 0)
        self.vllm_pid = data.get("vllm_pid")
    
    def _dict_to_checkpoint(self, data: Dict[str, Any]) -> ProfilerCheckpoint:
        """Convert dict to ProfilerCheckpoint."""
        return ProfilerCheckpoint(
            task_id=data.get("task_id", ""),
            completed_steps=data.get("completed_profilers", []),
            data=data.get("collected_data", {}),
            failed_steps=data.get("failed_profilers", []),
            skipped_steps=data.get("skipped_profilers", []),
        )
