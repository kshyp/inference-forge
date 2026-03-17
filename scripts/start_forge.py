#!/usr/bin/env python3
"""Entry point for Inference Forge.

Starts the optimization swarm with initial configuration.
Runs the continuous optimization loop until convergence.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from forge.orchestrator import InferenceEngine


async def main():
    parser = argparse.ArgumentParser(
        description="Inference Forge - VLLM Optimization Swarm"
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--model", default="gpt2", help="Model to optimize")
    parser.add_argument("--port", type=int, default=9995, help="vLLM server port")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--no-improvement-limit", type=int, default=3,
                        help="Stop after N iterations without improvement")
    parser.add_argument("--test", action="store_true", help="Run a single test iteration")
    args = parser.parse_args()
    
    # Check for LLM API keys
    has_llm = any([
        os.environ.get("MOONSHOT_API_KEY"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("VLLM_LOCAL_URL"),
    ])
    
    if not has_llm:
        print("❌ ERROR: At least one LLM API key required:")
        print("   - MOONSHOT_API_KEY")
        print("   - GEMINI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - OPENAI_API_KEY")
        print("   - VLLM_LOCAL_URL")
        return 1
    
    # Load config if exists
    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    # Get settings from config or args
    model = args.model
    port = args.port
    data_dir = args.data_dir
    max_iterations = args.max_iterations
    no_improvement_limit = args.no_improvement_limit
    
    if config:
        model = config.get("model", {}).get("name", model)
        port = config.get("agents", {}).get("benchmark", {}).get("port", port)
        data_dir = config.get("data_dir", data_dir)
        max_iterations = config.get("optimization", {}).get("max_iterations", max_iterations)
        no_improvement_limit = config.get("optimization", {}).get("no_improvement_limit", no_improvement_limit)
    
    if args.test:
        # Run a single test iteration (legacy mode)
        print("="*80)
        print("  INFERENCE FORGE - SINGLE ITERATION TEST MODE")
        print("="*80)
        print("\nFor continuous optimization, run without --test")
        
        from forge.agents.benchmark.agent import BenchmarkAgent
        from forge.core.events import Task
        from forge.core.state import StateStore
        
        state_store = StateStore(f"{data_dir}/state.db")
        agent = BenchmarkAgent(
            state_store=state_store,
            health_port=8081,
            data_dir=data_dir,
        )
        
        task = Task(
            type="benchmark",
            payload={
                "experiment_id": "test-exp-001",
                "model_name": model,
                "vllm_config": {"port": port},
                "guidellm_config": {
                    "dataset": "wikitext",
                    "prompt_tokens": 128,
                    "output_tokens": 64,
                    "max_requests": 100,
                }
            }
        )
        
        result = await agent.run_single_task(task)
        print(f"\nResult: {result}")
        
    else:
        # Run the continuous optimization engine
        engine = InferenceEngine(
            model_name=model,
            port=port,
            data_dir=data_dir,
            max_iterations=max_iterations,
            no_improvement_limit=no_improvement_limit,
        )
        
        convergence = await engine.run()
        
        if convergence.converged:
            print("\n✅ Optimization converged successfully!")
            return 0
        else:
            print("\n⚠️  Optimization did not converge")
            return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
