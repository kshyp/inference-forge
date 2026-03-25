#!/usr/bin/env python3
"""
RL-based vLLM Configuration Optimizer

This script runs a Thompson Sampling bandit to find optimal vLLM configurations.
It compares performance against a baseline and respects latency constraints.

Example usage:
    # Run with default settings (comprehensive config space)
    python scripts/run_rl_optimizer.py

    # Quick test with minimal config space
    python scripts/run_rl_optimizer.py --config-preset minimal --max-episodes 20

    # Focus on scheduling only
    python scripts/run_rl_optimizer.py --config-preset scheduling_only

    # Custom latency tolerance (10% instead of 5%)
    python scripts/run_rl_optimizer.py --latency-tolerance 0.10
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge.rl.runner import main

if __name__ == "__main__":
    exit(asyncio.run(main()))
