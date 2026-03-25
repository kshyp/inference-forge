"""Agent 2: Profile Executor - NSYS SEQUENCE VERSION.

Exact working sequence:
1. nsys launch vLLM (no output file here)
2. Run benchmark 
3. nsys start --gpu-metrics-devices=all -o <output>
4. sleep 5
5. nsys stop
"""

import asyncio
import json
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from forge.agents.base import BaseAgent
from forge.core.events import Task
from forge.core.state import StateStore
from .profilers.base import ProfilingContext


class ProfilerAgent(BaseAgent):
    """Minimal profiler that replicates the working nsys sequence."""
    
    def __init__(
        self,
        state_store: StateStore,
        health_port: int = 8082,
        data_dir: str = "./data",
    ):
        super().__init__(
            agent_id="profile",
            state_store=state_store,
            health_port=health_port
        )
        
        self.data_dir = Path(data_dir)
        self.vllm_process: Optional[asyncio.subprocess.Process] = None
        self.nsys_output_name: Optional[str] = None
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """Execute profiling with working nsys sequence."""
        payload = task.payload
        experiment_id = payload["experiment_id"]
        model_name = payload.get("model_name", "gpt2")
        vllm_config = payload.get("vllm_config", {})
        guidellm_config = payload.get("guidellm_config", {})
        enable_nsys = payload.get("enable_nsys", True)
        
        port = vllm_config.get("port", 8000)
        
        # Check if port is available
        if not self._check_port_available(port):
            print(f"   ⚠️  WARNING: Port {port} appears to be in use!")
            print(f"   You may need to kill any existing vLLM processes first.")
        
        # Create output directory
        exp_dir = self.data_dir / "experiments" / experiment_id
        profile_dir = exp_dir / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = profile_dir / "vllm_debug.log"
        
        # NSys output name (without extension)
        self.nsys_output_name = str(profile_dir / "profile")
        nsys_report = Path(f"{self.nsys_output_name}.nsys-rep")
        
        try:
            # Step 1: Start vLLM with nsys launch
            print(f"\n🚀 Step 1: nsys launch vLLM...")
            await self._start_vllm_nsys(
                model_name=model_name,
                port=port,
                log_file=log_file,
                vllm_config=vllm_config,
                enable_nsys=enable_nsys,
            )
            
            # Step 2: Warmup (send a few requests to warm up vLLM)
            print(f"📊 Step 2: Warmup requests...")
            await self._send_requests(model_name, port, num_requests=10)
            
            if enable_nsys:
                # Step 3: nsys start
                print(f"🔍 Step 3: nsys start...")
                await self._nsys_start()
                
                # Step 4: Send requests DURING nsys capture
                print(f"⏱️  Step 4: Sending requests during capture (5s)...")
                await self._send_requests_during_capture(model_name, port, duration_seconds=5)
                
                # Step 5: nsys stop
                print(f"🛑 Step 5: nsys stop...")
                await self._nsys_stop()
            else:
                # No nsys - just run normal benchmark
                print(f"📊 Running benchmark...")
                await self._run_benchmark(
                    model_name=model_name,
                    port=port,
                    guidellm_config=guidellm_config,
                )
            
            # Stop vLLM
            print(f"🛑 Stopping vLLM...")
            await self._stop_vllm()
            
            # Collect system metrics
            print(f"   Collecting system metrics...")
            from .profilers.system_metrics import SystemMetricsCollector
            metrics_collector = SystemMetricsCollector()
            
            system_metrics_result = await metrics_collector.run(
                ProfilingContext(output_dir=profile_dir, experiment_id=experiment_id)
            )
            
            if system_metrics_result.success:
                print(f"   ✓ System metrics collected")
                system_metrics = system_metrics_result.extractors
            else:
                print(f"   ⚠️  System metrics failed: {system_metrics_result.error_message}")
                system_metrics = {}
            
            # Check results
            nsys_exists = nsys_report.exists()
            if enable_nsys:
                if nsys_exists:
                    size_mb = nsys_report.stat().st_size / (1024 * 1024)
                    print(f"   ✓ NSys report: {nsys_report.name} ({size_mb:.1f} MB)")
                else:
                    print(f"   ⚠️  NSys report not found at {nsys_report}")
            
            return {
                "success": True,
                "log_file": str(log_file),
                "nsys_report": str(nsys_report) if nsys_exists else None,
                "profile_dir": str(profile_dir),
                "system_metrics": system_metrics,
            }
            
        except Exception as e:
            print(f"\n❌ Profiling failed: {e}")
            await self._nsys_stop()  # Try to stop nsys if running
            await self._stop_vllm()
            raise
    
    async def _start_vllm_nsys(
        self,
        model_name: str,
        port: int,
        log_file: Path,
        vllm_config: Dict[str, Any],
        enable_nsys: bool,
    ) -> None:
        """Start vLLM with nsys launch (exact command from working script)."""
        # Build vLLM command (base)
        vllm_cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "1024",
            "--max-num-seqs", "128",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "2048",
            "--enable-prefix-caching",
            "--port", str(port),
            "--model", model_name,
        ]
        
        # Wrap with nsys launch if enabled
        if enable_nsys:
            cmd = [
                "nsys", "launch",
                "--trace=cuda,nvtx,osrt",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
            ] + vllm_cmd
            print(f"   Using: nsys launch --trace=cuda,nvtx,osrt ...")
        else:
            cmd = vllm_cmd
            print(f"   Using: python -m vllm.entrypoints.openai.api_server ...")
        
        # Start process
        print(f"   Starting process...")
        print(f"   Log file: {log_file}")
        self.vllm_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        
        print(f"   PID: {self.vllm_process.pid}")
        
        # Wait for ready
        timeout = vllm_config.get("startup_timeout_seconds", 180)
        await self._wait_for_ready(port, timeout)
        print(f"   ✓ vLLM ready")
        
        # Extra delay to ensure vLLM is fully warmed up
        print(f"   Waiting 5s for warmup...")
        await asyncio.sleep(5)
    
    def _check_port_available(self, port: int) -> bool:
        """Check if port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result != 0  # 0 means port is in use
        except Exception:
            return False
    
    async def _wait_for_ready(self, port: int, timeout: float) -> None:
        """Wait for vLLM health endpoint."""
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
        
        raise TimeoutError(f"vLLM not ready after {timeout}s")
    
    async def _send_requests(self, model_name: str, port: int, num_requests: int) -> None:
        """Send a number of requests to vLLM."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                payload = {
                    "model": model_name,
                    "prompt": f"Warmup request {i}. Hello world.",
                    "max_tokens": 20,
                }
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            if (i + 1) % 5 == 0:
                                print(f"     {i + 1}/{num_requests} warmup done")
                        else:
                            print(f"     Request {i+1}: HTTP {resp.status}")
                except Exception as e:
                    print(f"     Request {i+1}: {e}")
                await asyncio.sleep(0.1)
        
        print(f"   ✓ Warmup complete")
    
    async def _send_requests_during_capture(self, model_name: str, port: int, duration_seconds: int) -> None:
        """Send requests continuously during nsys capture."""
        import aiohttp
        import time
        
        start_time = time.time()
        request_count = 0
        
        print(f"   Sending requests for {duration_seconds}s...")
        
        async with aiohttp.ClientSession() as session:
            while (time.time() - start_time) < duration_seconds:
                payload = {
                    "model": model_name,
                    "prompt": f"Capture request {request_count}. Testing inference performance.",
                    "max_tokens": 50,
                }
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            request_count += 1
                        else:
                            print(f"     HTTP {resp.status}")
                except Exception as e:
                    print(f"     Error: {e}")
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.05)
        
        print(f"   ✓ Sent {request_count} requests during capture")
    
    async def _run_benchmark(
        self,
        model_name: str,
        port: int,
        guidellm_config: Dict[str, Any],
    ) -> None:
        """Run benchmark using GuideLLM with JSON output only (no HTML)."""
        dataset = guidellm_config.get("dataset", "wikitext")
        prompt_tokens = guidellm_config.get("prompt_tokens", 128)
        output_tokens = guidellm_config.get("output_tokens", 64)
        max_requests = guidellm_config.get("max_requests", 200)
        
        # Create temp output directory for guidellm results
        import tempfile
        output_dir = tempfile.mkdtemp(prefix="guidellm_")
        
        # Use the working command format from the user
        cmd = [
            "guidellm", "benchmark", "run",
            "--target", f"http://localhost:{port}",
            "--model", model_name,
            "--profile", "throughput",
            "--rate", "128",
            "--data", f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}",
            "--request-type", "text_completions",
            "--max-requests", str(max_requests),
            "--outputs", "json",  # Only JSON, skip HTML to avoid remote fetch
            "--output-dir", output_dir,
        ]
        
        data_str = f"dataset={dataset},prompt_tokens={prompt_tokens},output_tokens={output_tokens}"
        print(f"   Running: guidellm benchmark run ...")
        print(f"   Data: {data_str}")
        print(f"   Request-type: text_completions")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()
        
        # Cleanup temp dir
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        
        if proc.returncode != 0:
            print(f"   Benchmark failed!")
            print(f"   stdout: {stdout_str[:2000]}")
            print(f"   stderr: {stderr_str[:2000]}")
            raise RuntimeError(f"Benchmark failed: {stderr_str[:500]}")
        
        print(f"   ✓ Benchmark complete")
    
    async def _nsys_start(self) -> None:
        """Run nsys start --gpu-metrics-devices=all -o <name>"""
        cmd = [
            "nsys", "start",
            "--gpu-metrics-devices=all",
            "-o", self.nsys_output_name,
        ]
        
        print(f"   Running: nsys start --gpu-metrics-devices=all -o {self.nsys_output_name}")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err = stderr.decode().strip()
            print(f"   nsys start stderr: {err}")
            raise RuntimeError(f"nsys start failed: {err}")
        
        print(f"   ✓ nsys start complete")
    
    async def _nsys_stop(self) -> None:
        """Run nsys stop"""
        cmd = ["nsys", "stop"]
        
        print(f"   Running: nsys stop")
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, stderr = await proc.communicate()
        
        # nsys stop may fail if no capture was active, that's ok
        if proc.returncode != 0:
            print(f"   (nsys stop: {stderr.decode().strip()[:100]})")
        else:
            print(f"   ✓ nsys stop complete")
    
    async def _stop_vllm(self) -> None:
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
            print(f"   ✓ vLLM stopped")
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        return {}
    
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        pass
