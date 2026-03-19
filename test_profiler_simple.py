#!/usr/bin/env python3
"""Minimal profiler test - just vLLM + nsys + simple requests.

This replicates your manual workflow:
1. nsys launch vLLM
2. Send some requests
3. nsys start
4. sleep 5
5. nsys stop
"""

import asyncio
import json
import signal
import subprocess
from pathlib import Path
from uuid import uuid4


async def run_cmd(cmd, desc=None, check=True):
    """Run a shell command."""
    if desc:
        print(f"   {desc}: {' '.join(cmd[:5])}...")
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    
    if check and proc.returncode != 0:
        err = stderr.decode()
        print(f"   ERROR: {err[:500]}")
        raise RuntimeError(f"Command failed: {err[:200]}")
    
    return stdout.decode(), stderr.decode()


async def test_simple(model: str = "gpt2", port: int = 8000):
    """Simple profiler test."""
    
    print("=" * 70)
    print("SIMPLE PROFILER TEST")
    print("=" * 70)
    print(f"Model: {model}, Port: {port}")
    print()
    
    # Setup
    exp_id = str(uuid4())[:8]
    data_dir = Path(f"./data/test_simple/{exp_id}")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = data_dir / "vllm.log"
    nsys_output = data_dir / "profile"
    
    vllm_process = None
    
    try:
        # Step 1: nsys launch vLLM
        print("Step 1: nsys launch vLLM")
        cmd = [
            "nsys", "launch",
            "--trace=cuda,nvtx,osrt",
            "--trace-fork-before-exec=true",
            "--cuda-graph-trace=node",
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--gpu-memory-utilization", "0.85",
            "--max-model-len", "1024",
            "--max-num-seqs", "128",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "2048",
            "--enable-prefix-caching",
            "--port", str(port),
            "--model", model,
        ]
        
        print(f"   Starting vLLM on port {port}...")
        vllm_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"   PID: {vllm_process.pid}")
        
        # Wait for vLLM to be ready
        print(f"   Waiting for vLLM to be ready...")
        import aiohttp
        start = asyncio.get_event_loop().time()
        ready = False
        while (asyncio.get_event_loop().time() - start) < 180:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{port}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            ready = True
                            break
            except Exception:
                pass
            await asyncio.sleep(1)
        
        if not ready:
            raise TimeoutError("vLLM failed to start")
        print(f"   ✓ vLLM ready")
        
        # Step 2: Send some requests
        print()
        print("Step 2: Sending test requests...")
        
        async with aiohttp.ClientSession() as session:
            # Send a few completion requests
            for i in range(5):
                payload = {
                    "model": model,
                    "prompt": "Hello, this is a test.",
                    "max_tokens": 50,
                }
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            print(f"   Request {i+1}: OK")
                        else:
                            print(f"   Request {i+1}: HTTP {resp.status}")
                except Exception as e:
                    print(f"   Request {i+1}: {e}")
                await asyncio.sleep(0.5)
        
        print(f"   ✓ Requests complete")
        
        # Step 3: nsys start
        print()
        print("Step 3: nsys start")
        await run_cmd(
            ["nsys", "start", "--gpu-metrics-devices=all", "-o", str(nsys_output)],
            "Starting nsys capture"
        )
        print(f"   ✓ nsys start complete")
        
        # Step 4: Send more requests during capture
        print()
        print("Step 4: Sending requests during capture...")
        
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                payload = {
                    "model": model,
                    "prompt": f"Test prompt number {i}.",
                    "max_tokens": 30,
                }
                try:
                    async with session.post(
                        f"http://localhost:{port}/v1/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            tokens = len(result.get("choices", [{}])[0].get("text", "").split())
                            print(f"   Request {i+1}: {tokens} tokens")
                        else:
                            print(f"   Request {i+1}: HTTP {resp.status}")
                except Exception as e:
                    print(f"   Request {i+1}: {e}")
                await asyncio.sleep(0.3)
        
        # Step 5: nsys stop
        print()
        print("Step 5: nsys stop")
        stdout, stderr = await run_cmd(
            ["nsys", "stop"],
            "Stopping nsys",
            check=False  # Don't fail if stop returns error
        )
        if "error" in stderr.lower():
            print(f"   (nsys stop warning: {stderr[:200]})")
        else:
            print(f"   ✓ nsys stop complete")
        
        # Step 6: Stop vLLM
        print()
        print("Step 6: Stopping vLLM")
        if vllm_process:
            try:
                import os
                os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
                await asyncio.wait_for(vllm_process.wait(), timeout=10)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            vllm_process = None
            print(f"   ✓ vLLM stopped")
        
        # Check results
        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        nsys_report = Path(f"{nsys_output}.nsys-rep")
        print(f"Data directory: {data_dir}")
        print(f"Log file: {log_file} ({log_file.stat().st_size / 1024:.1f} KB)")
        
        if nsys_report.exists():
            size_mb = nsys_report.stat().st_size / (1024 * 1024)
            print(f"NSys report: {nsys_report} ({size_mb:.2f} MB)")
            print()
            print("✅ TEST PASSED - NSys report created successfully")
        else:
            print(f"NSys report: NOT FOUND at {nsys_report}")
            print()
            print("❌ TEST FAILED - No NSys report")
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if vllm_process:
            try:
                import os
                os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
            except:
                pass


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    asyncio.run(test_simple(model, port))
