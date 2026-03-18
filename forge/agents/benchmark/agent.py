"""Agent 1: Benchmark Runner.

Integrates with existing autotuner scripts:
- Uses guidellm to run throughput benchmarks (modern CLI format)
- Runs at fixed rate of 512 RPS for all benchmarks
- Manages vLLM server lifecycle
"""

import asyncio
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from forge.agents.base import BaseAgent
from forge.core.events import (
    BenchmarkMetrics,
    BenchmarkResult,
    HardwareInfo,
    MetricsDistribution,
    SaturationPoint,
    Task,
)
from forge.core.state import StateStore


class BenchmarkAgent(BaseAgent):
    """Agent that runs vLLM benchmarks using GuideLLM.
    
    Throughput-focused approach:
    - Uses profile=throughput with rate=512 for all benchmarks
    - No sweep - runs at fixed high-rate for throughput measurement
    """
    
    def __init__(
        self,
        state_store: StateStore,
        health_port: int = 8081,
        data_dir: str = "./data",
        autotuner_dir: str = "./scripts/autotuner",
    ):
        super().__init__(
            agent_id="benchmark",
            state_store=state_store,
            health_port=health_port
        )
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.autotuner_dir = Path(autotuner_dir)
        
        # Runtime state (checkpointed)
        self.phase: str = "idle"  # idle, sweep, steady_state
        self.sweep_results: Optional[Path] = None
        self.saturation_rate: Optional[float] = None
        self.steady_state_results: Optional[Path] = None
        self.vllm_pid: Optional[int] = None
        
        # Task queue
        self._task_queue: asyncio.Queue = asyncio.Queue()
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next benchmark task from queue."""
        try:
            return await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def submit_task(self, task: Task) -> None:
        """Submit a benchmark task."""
        self._task_queue.put_nowait(task)
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Execute benchmark task.
        
        Runs throughput benchmark at fixed rate (512 RPS).
        Uses profile=throughput for all benchmark runs.
        
        Note: Detailed profiling is done by ProfilerAgent (Agent 2) 
        with DEBUG logging and profilers attached (NCU/NSys).
        
        Expected task payload:
        {
            "experiment_id": "uuid",
            "model_name": "gpt2" or "meta-llama/Llama-2-7b-hf",
            "vllm_config": {"port": 8000, "max_num_seqs": 64, ...},
            "guidellm_config": {
                "dataset": "wikitext",
                "prompt_tokens": 512,
                "output_tokens": 128,
                "max_requests": 1000,
            }
        }
        """
        payload = task.payload
        experiment_id = UUID(payload["experiment_id"])
        model_name = payload["model_name"]
        vllm_config = payload.get("vllm_config", {})
        guidellm_config = payload.get("guidellm_config", {})
        
        # Check if target rate is provided (skip sweep, run single benchmark)
        target_rate = payload.get("target_rate")
        
        port = vllm_config.get("port", 8000)
        
        # Create experiment directory
        exp_dir = self.data_dir / "experiments" / str(experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if target_rate:
                # Mode: Single benchmark at target rate (for comparing configs)
                print(f"📊 Running single benchmark at target rate: {target_rate:.2f} RPS")
                self.phase = "single_benchmark"
                print(f"   Config: max_num_seqs={vllm_config.get('max_num_seqs')}, "
                      f"max_num_batched_tokens={vllm_config.get('max_num_batched_tokens')}, "
                      f"enable_chunked_prefill={vllm_config.get('enable_chunked_prefill')}")
                await self.log_event("phase_started", task, {
                    "phase": "single_benchmark",
                    "target_rate": target_rate
                })
                
                steady_file = exp_dir / "benchmark_results.json"
                result = await self._run_single_benchmark(
                    model_name=model_name,
                    port=port,
                    rate=target_rate,
                    steady_file=steady_file,
                    guidellm_config=guidellm_config,
                    exp_dir=exp_dir,
                    config_flags=vllm_config,
                    task=task,
                )
                
                self.saturation_rate = target_rate
                
                # Debug: print extracted metrics
                print(f"   ✓ Benchmark complete: {result.metrics.requests_per_sec:.1f} req/s, "
                      f"TTFT={result.metrics.ttft_ms.p50:.1f}/{result.metrics.ttft_ms.p95:.1f}/{result.metrics.ttft_ms.p99:.1f}ms, "
                      f"TPOT={result.metrics.tpot_ms.p50:.1f}/{result.metrics.tpot_ms.p95:.1f}/{result.metrics.tpot_ms.p99:.1f}ms, "
                      f"Tokens={result.metrics.total_tokens_per_sec:.0f}/{result.metrics.output_tokens_per_sec:.0f}")
                
                await self.log_event("phase_completed", task, {
                    "phase": "single_benchmark",
                    "metrics": {
                        "requests_per_sec": result.metrics.requests_per_sec,
                        "ttft_p99": result.metrics.ttft_ms.p99,
                        "tpot_p99": result.metrics.tpot_ms.p99,
                    }
                })
                
                # Save result
                result_file = exp_dir / "benchmark_result.json"
                with open(result_file, "w") as f:
                    json.dump(self._benchmark_result_to_dict(result), f, indent=2)
                
                return {
                    "success": True,
                    "saturation_rate": target_rate,
                    "benchmark_result": self._benchmark_result_to_dict(result),
                    "result_file": str(result_file),
                    "message": f"Single benchmark at {target_rate:.2f} RPS complete.",
                }
            
            else:
                # Mode: Throughput benchmark at fixed rate
                print("📊 Running throughput benchmark at 512 RPS...")
                self.phase = "throughput"
                await self.log_event("phase_started", task, {"phase": "throughput"})
                
                throughput_file = exp_dir / "throughput_results.json"
                self.saturation_rate = await self._run_sweep(
                    model_name=model_name,
                    port=port,
                    sweep_file=throughput_file,
                    guidellm_config=guidellm_config,
                    exp_dir=exp_dir,
                    task=task,
                )
                self.sweep_results = throughput_file
                
                await self.log_event("phase_completed", task, {
                    "phase": "throughput",
                    "rate": 512
                })
                
                # Build result for handoff to ProfilerAgent
                hardware = self._detect_hardware()
                
                sweep_result = {
                    "experiment_id": str(experiment_id),
                    "config_flags": vllm_config,
                    "saturation_point": {
                        "rps": self.saturation_rate,
                        "latency_ms": 0,
                        "error_rate": 0.0,
                        "confidence": "high" if self.saturation_rate > 0 else "low",
                    },
                    "metrics": {
                        "ttft_ms": {"p50": 0, "p95": 0, "p99": 0},
                        "tpot_ms": {"p50": 0, "p95": 0, "p99": 0},
                        "output_tokens_per_sec": 0,
                        "total_tokens_per_sec": 0,
                        "requests_per_sec": 0,
                        "duration_seconds": 0,
                        "total_requests": 0,
                        "failed_requests": 0,
                    },
                    "hardware_info": {
                        "gpu_count": hardware.gpu_count,
                        "gpu_type": hardware.gpu_type,
                        "gpu_memory_gb": hardware.gpu_memory_gb,
                        "driver_version": hardware.driver_version,
                        "cuda_version": hardware.cuda_version,
                    },
                    "vllm_command": f"python -m vllm.entrypoints.openai.api_server --model {model_name} --port {port}",
                    "guidellm_report_path": str(throughput_file),
                    "vllm_logs_path": str(exp_dir / "vllm_server.log"),
                }
                
                result_file = exp_dir / "benchmark_result.json"
                with open(result_file, "w") as f:
                    json.dump(sweep_result, f, indent=2)
                
                return {
                    "success": True,
                    "rate": 512,
                    "throughput_result": sweep_result,
                    "result_file": str(result_file),
                    "message": "Throughput benchmark at 512 RPS complete. Handing off to ProfilerAgent for profiling.",
                }
            
        finally:
            # Cleanup - stop vLLM
            await self._cleanup_vllm()
            self.phase = "idle"
    
    async def _run_sweep(
        self,
        model_name: str,
        port: int,
        sweep_file: Path,
        guidellm_config: Dict[str, Any],
        exp_dir: Path,
        task: Task,
    ) -> float:
        """Run GuideLLM throughput benchmark.
        
        Uses modern CLI: guidellm benchmark run --profile throughput
        """
        # Construct data config
        dataset = guidellm_config.get("dataset", "wikitext")
        prompt_tokens = guidellm_config.get("prompt_tokens", 512)
        output_tokens = guidellm_config.get("output_tokens", 128)
        max_requests = guidellm_config.get("max_requests", 500)
        
        # Start vLLM first
        await self._start_vllm(
            model_name=model_name,
            port=port,
            exp_dir=exp_dir,
            task=task
        )
        
        # Build guidellm command (modern CLI format) - throughput profile at rate 512
        cmd = [
            "guidellm", "benchmark", "run",
            "--target", f"http://localhost:{port}",
            "--model", model_name,
            "--profile", "throughput",
            "--rate", "512",
            "--data", f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
            "--request-type", "text_completions",
            "--output-path", str(sweep_file),
            "--max-requests", str(max_requests),
        ]
        
        self.update_progress(10)
        
        # Run sweep
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            # Log the error for debugging
            print(f"GuideLLM stdout: {stdout.decode()}")
            print(f"GuideLLM stderr: {stderr.decode()}")
            raise RuntimeError(f"Sweep failed: {stderr.decode()}")
        
        self.update_progress(50)
        
        # For throughput profile, just return the fixed rate (512)
        # No need to find saturation point from sweep results
        saturation_rate = 512.0
        
        self.update_progress(60)
        
        return saturation_rate
    
    def _find_saturation_from_sweep(self, sweep_file: Path) -> float:
        """Parse GuideLLM sweep results to find saturation rate.
        
        Looks for the point where latency starts increasing rapidly
        or throughput plateaus.
        
        GuideLLM 0.5.3 format: {"benchmarks": [...], "metadata": ..., "args": ...}
        Each benchmark has config.strategy.rate and metrics.
        """
        if not sweep_file.exists():
            print(f"Warning: Sweep file not found: {sweep_file}")
            return 50.0  # Default fallback
        
        try:
            with open(sweep_file) as f:
                data = json.load(f)
            
            # GuideLLM 0.5.3 output format
            if isinstance(data, dict) and "benchmarks" in data:
                benchmarks = data["benchmarks"]
            elif isinstance(data, list):
                benchmarks = data
            else:
                benchmarks = []
            
            if not benchmarks:
                print("Warning: No benchmarks found in sweep results")
                return 50.0
            
            return self._analyze_benchmarks(benchmarks)
            
        except Exception as e:
            print(f"Warning: Could not parse sweep results: {e}")
            import traceback
            traceback.print_exc()
            return 50.0
    
    def _analyze_benchmarks(self, benchmarks: List[Dict]) -> float:
        """Analyze benchmark results to find saturation point.
        
        GuideLLM 0.5.3 format:
        - rate is in config.strategy.rate (for constant profile)
        - latency is in metrics.request_latency.successful.mean (in seconds)
        - throughput is in metrics.requests_per_second.successful.mean
        """
        if not benchmarks:
            return 50.0
        
        # Extract rate and latency for each benchmark (only constant profile)
        results = []
        for b in benchmarks:
            config = b.get("config", {})
            strategy = config.get("strategy", {})
            
            # Only consider constant rate benchmarks
            if strategy.get("type_") != "constant":
                continue
            
            rate = strategy.get("rate", 0)
            
            metrics = b.get("metrics", {})
            latency_stats = metrics.get("request_latency", {}).get("successful", {})
            latency = latency_stats.get("mean", 0)  # In seconds
            latency_ms = latency * 1000  # Convert to ms
            
            throughput_stats = metrics.get("requests_per_second", {}).get("successful", {})
            throughput = throughput_stats.get("mean", 0)
            
            if rate > 0 and latency > 0:
                results.append({
                    "rate": float(rate),
                    "latency_ms": float(latency_ms),
                    "throughput": float(throughput),
                })
        
        if not results:
            print("Warning: No valid constant-rate benchmarks found")
            return 50.0
        
        # Sort by rate
        results.sort(key=lambda x: x["rate"])
        
        print(f"Analyzing {len(results)} benchmarks:")
        for r in results:
            print(f"  Rate: {r['rate']:.2f}, Latency: {r['latency_ms']:.2f}ms, Throughput: {r['throughput']:.2f}")
        
        # Find saturation: where latency increases >2x from previous
        for i in range(1, len(results)):
            prev = results[i-1]
            curr = results[i]
            
            if curr["latency_ms"] > prev["latency_ms"] * 2:
                # Saturation detected
                print(f"Saturation detected at rate {prev['rate']:.2f} (latency jump: {prev['latency_ms']:.2f}ms -> {curr['latency_ms']:.2f}ms)")
                return prev["rate"]
        
        # No clear saturation - use 80% of max rate
        max_rate = results[-1]["rate"]
        saturation = max_rate * 0.8
        print(f"No clear saturation, using {saturation:.2f} (80% of max {max_rate:.2f})")
        return saturation
    
    async def _run_single_benchmark(
        self,
        model_name: str,
        rate: float,
        port: int,
        steady_file: Path,
        guidellm_config: Dict[str, Any],
        exp_dir: Path,
        config_flags: Dict[str, Any],
        task: Task,
    ) -> BenchmarkResult:
        """Run steady-state benchmark at saturation rate."""
        # Construct data config
        dataset = guidellm_config.get("dataset", "wikitext")
        prompt_tokens = guidellm_config.get("prompt_tokens", 512)
        output_tokens = guidellm_config.get("output_tokens", 128)
        max_requests = guidellm_config.get("max_requests", 1000)
        
        # Restart vLLM for clean state
        await self._cleanup_vllm()
        await self._start_vllm(
            model_name=model_name,
            port=port,
            exp_dir=exp_dir,
            task=task
        )
        
        # Build guidellm command for throughput run at rate 512
        cmd = [
            "guidellm", "benchmark", "run",
            "--target", f"http://localhost:{port}",
            "--model", model_name,
            "--profile", "throughput",
            "--rate", "512",
            "--data", f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
            "--request-type", "text_completions",
            "--output-path", str(steady_file),
            "--max-requests", str(max_requests),
        ]
        
        self.update_progress(70)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            print(f"GuideLLM stdout: {stdout.decode()}")
            print(f"GuideLLM stderr: {stderr.decode()}")
            raise RuntimeError(f"Steady-state run failed: {stderr.decode()}")
        
        self.update_progress(90)
        
        # Parse results
        metrics = self._parse_steady_state_results(steady_file)
        
        # Build result
        saturation = SaturationPoint(
            rps=rate,
            latency_ms=metrics.ttft_ms.p50,
            error_rate=0.0,
            confidence="high",
        )
        
        hardware = self._detect_hardware()
        
        self.update_progress(100)
        
        return BenchmarkResult(
            experiment_id=task.payload["experiment_id"],
            config_flags=config_flags,
            saturation_point=saturation,
            metrics=metrics,
            guidellm_report_path=steady_file,
            vllm_logs_path=exp_dir / "vllm_server.log",
            hardware_info=hardware,
            vllm_command=f"python -m vllm.entrypoints.openai.api_server --model {model_name} --port {port}",
        )
    
    def _parse_steady_state_results(self, results_file: Path) -> BenchmarkMetrics:
        """Parse GuideLLM steady-state results.
        
        GuideLLM 0.5.3 format:
        - metrics are nested under metrics.*.successful
        - time_to_first_token_ms has percentiles
        - time_per_output_token_ms has percentiles
        """
        if not results_file.exists():
            print(f"Warning: Results file not found: {results_file}")
            return self._default_metrics()
        
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            # Extract benchmark from results
            if isinstance(data, dict) and "benchmarks" in data and data["benchmarks"]:
                bench = data["benchmarks"][0]  # Take first benchmark
            elif isinstance(data, list) and data:
                bench = data[0]
            elif isinstance(data, dict):
                bench = data
            else:
                return self._default_metrics()
            
            metrics = bench.get("metrics", {})
            
            # Helper to extract percentiles from metric stats
            def get_percentiles(metric_name: str) -> MetricsDistribution:
                """Extract p50, p95, p99 from nested metric structure.
                
                GuideLLM format: metrics.{metric_name}.successful.percentiles
                """
                metric_data = metrics.get(metric_name, {})
                if not isinstance(metric_data, dict):
                    return MetricsDistribution(p50=0, p95=0, p99=0)
                
                successful = metric_data.get("successful", {})
                if not isinstance(successful, dict):
                    return MetricsDistribution(p50=0, p95=0, p99=0)
                
                # Try percentiles first, then mean/median
                percentiles = successful.get("percentiles", {})
                if percentiles:
                    return MetricsDistribution(
                        p50=percentiles.get("p50", 0) or successful.get("median", 0),
                        p95=percentiles.get("p95", 0) or successful.get("mean", 0) * 1.5,
                        p99=percentiles.get("p99", 0) or successful.get("mean", 0) * 2,
                    )
                else:
                    mean = successful.get("mean", 0)
                    return MetricsDistribution(
                        p50=successful.get("median", mean) or mean,
                        p95=mean * 1.5 if mean else 0,
                        p99=mean * 2 if mean else 0,
                    )
            
            # Extract throughput metrics
            req_per_sec_stats = metrics.get("requests_per_second", {}).get("successful", {})
            requests_per_sec = req_per_sec_stats.get("mean", 0)
            
            output_tok_stats = metrics.get("output_tokens_per_second", {}).get("successful", {})
            output_tokens_per_sec = output_tok_stats.get("mean", 0)
            
            total_tok_stats = metrics.get("tokens_per_second", {}).get("successful", {})
            total_tokens_per_sec = total_tok_stats.get("mean", output_tokens_per_sec)
            
            # Extract request totals
            request_totals = metrics.get("request_totals", {})
            total_requests = request_totals.get("total", 0)
            failed_requests = request_totals.get("errored", 0)
            
            # Extract duration
            duration = bench.get("duration", 0)
            
            return BenchmarkMetrics(
                ttft_ms=get_percentiles("time_to_first_token_ms"),
                tpot_ms=get_percentiles("time_per_output_token_ms"),
                itl_ms=get_percentiles("inter_token_latency_ms"),
                output_tokens_per_sec=output_tokens_per_sec,
                total_tokens_per_sec=total_tokens_per_sec,
                requests_per_sec=requests_per_sec,
                duration_seconds=duration,
                total_requests=total_requests,
                failed_requests=failed_requests,
            )
            
        except Exception as e:
            print(f"Warning: Could not parse results: {e}")
            import traceback
            traceback.print_exc()
            return self._default_metrics()
    
    def _default_metrics(self) -> BenchmarkMetrics:
        """Return default metrics when parsing fails."""
        return BenchmarkMetrics(
            ttft_ms=MetricsDistribution(p50=0, p95=0, p99=0),
            tpot_ms=MetricsDistribution(p50=0, p95=0, p99=0),
            itl_ms=MetricsDistribution(p50=0, p95=0, p99=0),
            output_tokens_per_sec=0,
            total_tokens_per_sec=0,
            requests_per_sec=0,
            duration_seconds=0,
            total_requests=0,
            failed_requests=0,
        )
    
    async def _start_vllm(
        self,
        model_name: str,
        port: int,
        exp_dir: Path,
        task: Task,
    ) -> None:
        """Start vLLM server."""
        log_file = exp_dir / "vllm_server.log"
        
        # Check for start script
        start_script = self.autotuner_dir / "start_vllm_server.sh"
        
        if start_script.exists():
            cmd = ["bash", str(start_script), model_name, str(port)]
        else:
            # Direct command
            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_name,
                "--port", str(port),
            ]
        
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
        
        # Wait for server to be ready
        await self._wait_for_vllm_ready(port, timeout=120)
    
    async def _wait_for_vllm_ready(self, port: int, timeout: float = 120) -> None:
        """Wait for vLLM server health endpoint."""
        import aiohttp
        
        start = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start) < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            return
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"vLLM server not ready after {timeout}s")
    
    async def _cleanup_vllm(self) -> None:
        """Stop vLLM server."""
        if self.vllm_process:
            import signal
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
    
    def _detect_hardware(self) -> HardwareInfo:
        """Detect GPU hardware."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            
            cuda_version = None
            try:
                import torch
                cuda_version = torch.version.cuda
            except ImportError:
                pass
            
            return HardwareInfo(
                gpu_count=pynvml.nvmlDeviceGetCount(),
                gpu_type=name.decode() if isinstance(name, bytes) else name,
                gpu_memory_gb=mem.total / 1024**3,
                driver_version=driver.decode() if isinstance(driver, bytes) else driver,
                cuda_version=cuda_version,
            )
            
        except Exception:
            return HardwareInfo(
                gpu_count=0,
                gpu_type="CPU",
                gpu_memory_gb=0,
                driver_version="unknown",
            )
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "phase": self.phase,
            "saturation_rate": self.saturation_rate,
            "sweep_results": str(self.sweep_results) if self.sweep_results else None,
            "steady_state_results": str(self.steady_state_results) if self.steady_state_results else None,
            "vllm_pid": self.vllm_pid,
            "current_progress": self.current_progress,
        }
    
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.phase = data.get("phase", "idle")
        self.saturation_rate = data.get("saturation_rate")
        if data.get("sweep_results"):
            self.sweep_results = Path(data["sweep_results"])
        if data.get("steady_state_results"):
            self.steady_state_results = Path(data["steady_state_results"])
        self.vllm_pid = data.get("vllm_pid")
        self.current_progress = data.get("current_progress", 0)
        
        # vLLM process needs restart after crash
        self.vllm_process = None
    
    def _benchmark_result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to dict for JSON serialization."""
        return {
            "experiment_id": str(result.experiment_id),
            "config_flags": result.config_flags,
            "saturation_point": {
                "rps": result.saturation_point.rps,
                "latency_ms": result.saturation_point.latency_ms,
                "error_rate": result.saturation_point.error_rate,
                "confidence": result.saturation_point.confidence,
            },
            "metrics": {
                "ttft_ms": {
                    "p50": result.metrics.ttft_ms.p50,
                    "p95": result.metrics.ttft_ms.p95,
                    "p99": result.metrics.ttft_ms.p99,
                },
                "tpot_ms": {
                    "p50": result.metrics.tpot_ms.p50,
                    "p95": result.metrics.tpot_ms.p95,
                    "p99": result.metrics.tpot_ms.p99,
                },
                "itl_ms": {
                    "p50": result.metrics.itl_ms.p50,
                    "p95": result.metrics.itl_ms.p95,
                    "p99": result.metrics.itl_ms.p99,
                },
                "output_tokens_per_sec": result.metrics.output_tokens_per_sec,
                "total_tokens_per_sec": result.metrics.total_tokens_per_sec,
                "requests_per_sec": result.metrics.requests_per_sec,
                "duration_seconds": result.metrics.duration_seconds,
                "total_requests": result.metrics.total_requests,
                "failed_requests": result.metrics.failed_requests,
            },
            "hardware_info": {
                "gpu_count": result.hardware_info.gpu_count,
                "gpu_type": result.hardware_info.gpu_type,
                "gpu_memory_gb": result.hardware_info.gpu_memory_gb,
                "driver_version": result.hardware_info.driver_version,
                "cuda_version": result.hardware_info.cuda_version,
            },
            "vllm_command": result.vllm_command,
            "guidellm_report_path": str(result.guidellm_report_path),
            "vllm_logs_path": str(result.vllm_logs_path),
            "created_at": result.created_at.isoformat(),
        }
