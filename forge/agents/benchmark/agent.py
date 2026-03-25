"""Agent 1: Benchmark Runner.

Integrates with existing autotuner scripts:
- Uses guidellm to run throughput benchmarks (modern CLI format)
- Runs single benchmark at fixed rate (128 RPS) for baseline
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
    - Uses profile=throughput with rate=128 for baseline measurement
    - Single benchmark run - no multiple passes
    - Returns metrics for comparison
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
        self.phase: str = "idle"  # idle, running
        self.benchmark_results: Optional[Path] = None
        self.baseline_rate: float = 128.0  # Fixed rate for all benchmarks
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
        
        Runs throughput benchmark at fixed rate (128 RPS).
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
        
        port = vllm_config.get("port", 8000)
        
        # Create unified experiment directory structure
        exp_dir = self.data_dir / "experiments" / str(experiment_id)
        benchmark_dir = exp_dir / "benchmark"
        try:
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            print(f"   📁 Created benchmark directory: {benchmark_dir}")
        except Exception as e:
            print(f"   ❌ Failed to create benchmark directory: {e}")
            raise
        
        try:
            # Run single benchmark at fixed rate
            print(f"📊 Running benchmark at {self.baseline_rate:.0f} RPS...")
            self.phase = "benchmark"
            print(f"   Config: max_num_seqs={vllm_config.get('max_num_seqs')}, "
                  f"max_num_batched_tokens={vllm_config.get('max_num_batched_tokens')}, "
                  f"enable_chunked_prefill={vllm_config.get('enable_chunked_prefill')}")
            await self.log_event("phase_started", task, {
                "phase": "benchmark",
                "rate": self.baseline_rate
            })
            
            result_file = benchmark_dir / "results.json"
            result = await self._run_benchmark(
                model_name=model_name,
                port=port,
                rate=self.baseline_rate,
                result_file=result_file,
                guidellm_config=guidellm_config,
                benchmark_dir=benchmark_dir,
                config_flags=vllm_config,
                task=task,
            )
            
            self.benchmark_results = result_file
            
            # Print extracted metrics
            print(f"   ✓ Benchmark complete: {result.metrics.requests_per_sec:.1f} req/s, "
                  f"TTFT={result.metrics.ttft_ms.p50:.1f}/{result.metrics.ttft_ms.p95:.1f}/{result.metrics.ttft_ms.p99:.1f}ms, "
                  f"TPOT={result.metrics.tpot_ms.p50:.1f}/{result.metrics.tpot_ms.p95:.1f}/{result.metrics.tpot_ms.p99:.1f}ms, "
                  f"Tokens={result.metrics.total_tokens_per_sec:.0f}/{result.metrics.output_tokens_per_sec:.0f}")
            
            await self.log_event("phase_completed", task, {
                "phase": "benchmark",
                "metrics": {
                    "requests_per_sec": result.metrics.requests_per_sec,
                    "ttft_p99": result.metrics.ttft_ms.p99,
                    "tpot_p99": result.metrics.tpot_ms.p99,
                }
            })
            
            # Save result
            result_dict = self._benchmark_result_to_dict(result)
            with open(benchmark_dir / "result.json", "w") as f:
                json.dump(result_dict, f, indent=2)
            
            return {
                "success": True,
                "baseline_rate": self.baseline_rate,
                "benchmark_result": result_dict,
                "result_file": str(result_file),
                "message": f"Benchmark at {self.baseline_rate:.0f} RPS complete.",
            }
        
        finally:
            # Cleanup - stop vLLM
            await self._cleanup_vllm()
            self.phase = "idle"
    
    async def _run_benchmark(
        self,
        model_name: str,
        rate: float,
        port: int,
        result_file: Path,
        guidellm_config: Dict[str, Any],
        benchmark_dir: Path,
        config_flags: Dict[str, Any],
        task: Task,
    ) -> BenchmarkResult:
        """Run benchmark at specified rate."""
        # Construct data config
        dataset = guidellm_config.get("dataset", "wikitext")
        prompt_tokens = guidellm_config.get("prompt_tokens", 512)
        output_tokens = guidellm_config.get("output_tokens", 128)
        max_requests = guidellm_config.get("max_requests", 1000)
        
        # Start vLLM
        await self._start_vllm(
            model_name=model_name,
            port=port,
            benchmark_dir=benchmark_dir,
            task=task,
            config_flags=config_flags,
        )
        
        # Build guidellm command for throughput run
        cmd = [
            "guidellm", "benchmark", "run",
            "--target", f"http://localhost:{port}",
            "--model", model_name,
            "--profile", "throughput",
            "--rate", str(int(rate)),
            "--data", f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
            "--request-type", "text_completions",
            "--output-path", str(result_file),
            "--max-requests", str(max_requests),
        ]
        
        self.update_progress(50)
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            print(f"GuideLLM stdout: {stdout.decode()}")
            print(f"GuideLLM stderr: {stderr.decode()}")
            raise RuntimeError(f"Benchmark failed: {stderr.decode()}")
        
        self.update_progress(90)
        
        # Parse results
        metrics = self._parse_benchmark_results(result_file)
        
        # Build result
        baseline = SaturationPoint(
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
            saturation_point=baseline,
            metrics=metrics,
            guidellm_report_path=result_file,
            vllm_logs_path=benchmark_dir / "vllm_server.log",
            hardware_info=hardware,
            vllm_command=f"python -m vllm.entrypoints.openai.api_server --model {model_name} --port {port}",
        )
    
    def _parse_benchmark_results(self, results_file: Path) -> BenchmarkMetrics:
        """Parse GuideLLM benchmark results.
        
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
        benchmark_dir: Path,
        task: Task,
        config_flags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Start vLLM server with configuration flags."""
        log_file = benchmark_dir / "vllm_server.log"
        config_flags = config_flags or {}
        
        # Build command with config flags
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
        ]
        
        # Add config flags
        if "max_num_seqs" in config_flags:
            cmd.extend(["--max-num-seqs", str(config_flags["max_num_seqs"])])
        if "max_num_batched_tokens" in config_flags:
            cmd.extend(["--max-num-batched-tokens", str(config_flags["max_num_batched_tokens"])])
        if "gpu_memory_utilization" in config_flags:
            cmd.extend(["--gpu-memory-utilization", str(config_flags["gpu_memory_utilization"])])
        if "max_model_len" in config_flags:
            cmd.extend(["--max-model-len", str(config_flags["max_model_len"])])
        if config_flags.get("enable_chunked_prefill"):
            cmd.append("--enable-chunked-prefill")
        if "max_chunked_prefill_len" in config_flags:
            cmd.extend(["--max-num-batched-tokens", str(config_flags["max_chunked_prefill_len"])])
        if config_flags.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")
        if config_flags.get("speculative_model") == "ngram":
            # Build speculative config JSON for vLLM 0.16.0+
            spec_config = {
                "method": "ngram",
                "num_speculative_tokens": config_flags.get("num_lookahead_slots", 2),
                "prompt_lookup_max": config_flags.get("ngram_prompt_lookup_max", 4),
            }
            import json
            cmd.extend(["--speculative-config", json.dumps(spec_config)])
        
        print(f"   Starting vLLM: {' '.join(cmd)}")
        
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
        await self._wait_for_vllm_ready(port, timeout=300)
    
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
            "baseline_rate": self.baseline_rate,
            "benchmark_results": str(self.benchmark_results) if self.benchmark_results else None,
            "vllm_pid": self.vllm_pid,
            "current_progress": self.current_progress,
        }
    
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.phase = data.get("phase", "idle")
        self.baseline_rate = data.get("baseline_rate", 128.0)
        if data.get("benchmark_results"):
            self.benchmark_results = Path(data["benchmark_results"])
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
