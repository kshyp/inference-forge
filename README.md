# 🔥 Inference Forge

Autonomous multi-agent system for optimizing vLLM inference configurations using LLM-driven analysis.

## Overview

Inference Forge uses a swarm of specialized AI agents to automatically find optimal vLLM configurations for your hardware and workload:

1. **Benchmark Agent** - Finds saturation point and measures performance
2. **Profiler Agent** - Collects detailed metrics with DEBUG logging
3. **Coordinator Agent** - Consults SMEs (Subject Matter Experts) and generates experiment plans

The system iterates until convergence, trying different configurations and comparing results against a baseline.

## Features

- 🎯 **Automatic Saturation Detection** - Finds the maximum sustainable request rate
- 🧠 **LLM-Driven Analysis** - Uses multiple LLMs (Claude, GPT-4, Gemini, etc.) for intelligent optimization
- 📊 **Performance Comparison Table** - Clear visualization of improvements/regressions across iterations
- 🔧 **SME System** - Modular experts for different optimization areas:
  - Scheduling (chunked prefill, batch sizes)
  - Quantization (FP8, INT8, etc.)
  - Speculative Decoding
  - Model Parallelism (multi-GPU)
- 🔄 **Continuous Optimization** - Runs until convergence or max iterations

## Quick Start

### Prerequisites

```bash
# Install vLLM
pip install vllm

# Install GuideLLM for benchmarking
pip install guidellm

# Set up LLM API keys (at least one required)
export MOONSHOT_API_KEY="your-key"      # Recommended
export GEMINI_API_KEY="your-key"        # Alternative
export ANTHROPIC_API_KEY="your-key"     # Alternative
export OPENAI_API_KEY="your-key"        # Alternative
```

### Run Optimization

```bash
# Basic usage with default config
python scripts/start_forge.py --model gpt2 --port 8081

# With custom settings
python scripts/start_forge.py \
    --model meta-llama/Llama-2-7b-hf \
    --port 8081 \
    --max-iterations 10 \
    --no-improvement-limit 3
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Enable/disable specific SMEs
smes:
  - id: scheduling
    enabled: true
  - id: quantization
    enabled: true
  - id: speculative
    enabled: true

# Optimization parameters
optimization:
  max_iterations: 10
  no_improvement_limit: 3
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Benchmark      │────▶│  Profile        │────▶│  Coordinate     │
│  Agent          │     │  Agent          │     │  Agent          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                                               │
        │                                               ▼
        │                                       ┌─────────────────┐
        │                                       │  SME Registry   │
        │                                       │  - Scheduling   │
        │                                       │  - Quantization │
        │                                       │  - Speculative  │
        │                                       └─────────────────┘
        │                                               │
        ▼                                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Comparison Table                  │
│  Iter │ Config                    │ Req/s │ TTFT     │ TPOT     │
│    1  │ max_num_seqs=64           │ 45.5  │ 67/134/161  │ 13/14/15 │ 📊 BASE
│    2  │ max_num_seqs=256,...      │ 53.2  │ 65/131/149  │ 13/15/16 │ 🟢 +17%
└─────────────────────────────────────────────────────────────────┘
```

## Output

Results are stored in `data/`:
- `experiments/<id>/` - Individual experiment results
- `optimization_state.json` - Full optimization history
- `state.db` - SQLite database of all runs

## Project Structure

```
inference-forge/
├── forge/                    # Core framework
│   ├── agents/              # Agent implementations
│   │   ├── benchmark/       # Benchmark agent
│   │   ├── coordinator/     # Coordinator agent
│   │   └── profile/         # Profiler agent
│   ├── core/                # Core utilities (events, state, health)
│   ├── llm/                 # LLM pool and consensus
│   ├── smes/                # Subject Matter Experts
│   └── orchestrator.py      # Main optimization loop
├── scripts/
│   └── start_forge.py       # Entry point
├── config.yaml              # Configuration
├── pyproject.toml           # Dependencies
└── DESIGN.md                # Architecture documentation
```

## Architecture

See [DESIGN.md](DESIGN.md) for detailed architecture documentation.

## License

MIT
