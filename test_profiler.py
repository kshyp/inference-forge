#!/usr/bin/env python3
"""Standalone test for the profiler agent.

Usage:
    python test_profiler.py [--model MODEL] [--port PORT] [--no-nsys]

Example:
    python test_profiler.py --model gpt2 --port 8000
"""

import asyncio
import argparse
import json
from pathlib import Path
from uuid import uuid4

# Add forge to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from forge.agents.profile.agent import ProfilerAgent
from forge.core.events import Task
from forge.core.state import StateStore


async def test_profiler(model: str, port: int, enable_nsys: bool):
    """Test the profiler agent in isolation."""
    
    print("=" * 70)
    print("PROFILER AGENT STANDALONE TEST")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Port: {port}")
    print(f"NSys: {'enabled' if enable_nsys else 'disabled'}")
    print()
    
    # Create agent
    data_dir = "./data/test_profiler"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    state_store = StateStore(str(Path(data_dir) / "state.db"))
    agent = ProfilerAgent(
        state_store=state_store,
        health_port=38081,
        data_dir=data_dir,
    )
    
    # Create task
    experiment_id = str(uuid4())
    task = Task(
        type="profile",
        payload={
            "experiment_id": experiment_id,
            "model_name": model,
            "vllm_config": {
                "port": port,
                "startup_timeout_seconds": 180,
            },
            "guidellm_config": {
                "dataset": "wikitext",
                "prompt_tokens": 128,
                "output_tokens": 64,
                "max_requests": 100,  # Reduced for faster testing
            },
            "enable_nsys": enable_nsys,
        }
    )
    
    print(f"Experiment ID: {experiment_id}")
    print()
    
    try:
        result = await agent.execute(task)
        
        print()
        print("=" * 70)
        print("RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2, default=str))
        
        if result.get("success"):
            print()
            print("✅ PROFILER TEST PASSED")
            
            # Show output files
            profile_dir = Path(result.get("profile_dir", ""))
            if profile_dir.exists():
                print()
                print(f"Output files in {profile_dir}:")
                for f in profile_dir.iterdir():
                    size = f.stat().st_size / 1024  # KB
                    print(f"  - {f.name}: {size:.1f} KB")
        else:
            print()
            print("❌ PROFILER TEST FAILED")
            print(f"Error: {result.get('_error', 'Unknown error')}")
            
    except Exception as e:
        print()
        print("=" * 70)
        print("EXCEPTION")
        print("=" * 70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test profiler agent in isolation")
    parser.add_argument("--model", default="gpt2", help="Model to test with")
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM")
    parser.add_argument("--no-nsys", action="store_true", help="Disable nsys profiling")
    
    args = parser.parse_args()
    
    asyncio.run(test_profiler(
        model=args.model,
        port=args.port,
        enable_nsys=not args.no_nsys,
    ))
