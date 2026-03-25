# RL-Based vLLM Configuration Optimizer

A clean-slate reinforcement learning approach to finding optimal vLLM server configurations using **Thompson Sampling bandit algorithm**.

---

## Overview

This optimizer replaces the multi-SME (Subject Matter Expert) approach with a simple black-box RL method:

| Aspect | Multi-SME Approach | RL Approach |
|--------|-------------------|-------------|
| **Profiling** | DEBUG logs, NSys, NCU | None (black-box) |
| **Expert Knowledge** | 6 SMEs with LLM consensus | None |
| **Decision Making** | Rule-based synthesis | Thompson Sampling |
| **Exploration** | Ranked experiment backlog | Posterior sampling |
| **Latency Guarantee** | None explicit | Hard 5% limit (or custom absolute) |

---

## Quick Start

```bash
# Quick test with minimal config space (108 configs)
python scripts/run_rl_optimizer.py --config-preset minimal --max-episodes 10

# Standard run with comprehensive config space (~36K configs)
python scripts/run_rl_optimizer.py --max-episodes 100

# Focus on scheduling parameters only
python scripts/run_rl_optimizer.py --config-preset scheduling_only --max-episodes 50

# Watch convergence in real-time with live dashboard
python scripts/run_rl_optimizer.py --max-episodes 100 --dashboard terminal

# Real-time matplotlib plots (requires: pip install matplotlib)
python scripts/run_rl_optimizer.py --max-episodes 100 --dashboard plot

# Run until convergence (auto-detect when best config stabilizes)
python scripts/run_rl_optimizer.py --convergence

# Run until convergence with custom stability window
python scripts/run_rl_optimizer.py --convergence --convergence-window 30

# Prioritize speculative decoding configs first
python scripts/run_rl_optimizer.py --convergence --warm-start-speculative

# Maximize throughput with strict 100ms latency constraint
python scripts/run_rl_optimizer.py --max-latency-ms 100 --convergence

# Allow 10% latency degradation for higher throughput
python scripts/run_rl_optimizer.py --latency-tolerance 0.10 --max-episodes 100
```

---

## How It Works

### Algorithm: Thompson Sampling

```
1. Initialize: Each config gets Gaussian posterior N(mean=0.5, var=1.0)

2. For each episode:
   a. Sample reward estimate from each config's posterior
   b. Select config with highest sample
   c. Run GuideLLM benchmark
   d. Calculate reward (0 if latency > 105% baseline)
   e. Update posterior: N(mean', var') based on observed reward

3. Repeat until max-episodes or convergence
```

### Key Property: Automatic Focus on Promising Regions

The bandit **automatically concentrates exploration** on high-reward configs:

- High-reward config tried once → Higher posterior mean → More likely to be selected again
- Low-reward config tried once → Lower posterior mean → Abandoned
- Untried configs → Wide variance → Occasionally sampled (exploration)

**Example Progression:**

```
Episode 1:  Random across space (all posteriors equal)
Episode 5:  Focusing on 3-4 promising configs
Episode 20: 80% of selections from top 2 configs
Episode 50: Converged on best config
```

---

## Reward Function

The reward function maximizes **throughput within strict latency constraints**:

```python
if latency_p50 > max_latency_threshold:
    reward = 0.0  # Constraint violated
elif failed_requests > 0:
    reward = 0.0  # Hard failure
else:
    reward = min(throughput / baseline_throughput, 2.0)
```

### Latency Constraints (Two Modes)

The optimizer supports two ways to specify latency constraints:

| Mode | Parameter | Description | Use Case |
|------|-----------|-------------|----------|
| **Relative** (default) | `--latency-tolerance` | Max % degradation from baseline | SLA relative to current performance |
| **Absolute** | `--max-latency-ms` | Hard limit in milliseconds | Strict SLA (e.g., "must be < 100ms") |

**Default behavior:** 5% latency tolerance (`--latency-tolerance 0.05`)

**Priority:** If `--max-latency-ms` is set, it **overrides** `--latency-tolerance`

### Examples

```bash
# Relative: Allow 5% degradation (default)
python scripts/run_rl_optimizer.py --latency-tolerance 0.05

# Relative: Aggressive - allow 15% for max throughput
python scripts/run_rl_optimizer.py --latency-tolerance 0.15

# Absolute: Hard 80ms limit (ignores baseline)
python scripts/run_rl_optimizer.py --max-latency-ms 80

# Absolute: 150ms limit with convergence mode
python scripts/run_rl_optimizer.py --max-latency-ms 150 --convergence
```

**Properties:**
- Rewards throughput improvement up to 200%
- Zero reward if latency exceeds threshold (hard constraint)
- Zero reward on any failures
- Automatically finds best throughput **within** latency budget

---

## Configuration Space

### Comprehensive (Default)

| Parameter | Values | Conditional |
|-----------|--------|-------------|
| `max_num_seqs` | 64, 128, 256, 512, 1024 | - |
| `max_num_batched_tokens` | 1024, 2048, 4096, 8192 | - |
| `enable_chunked_prefill` | False, True | - |
| `max_chunked_prefill_len` | 512, 1024, 2048 | If chunked_prefill=True |
| `num_scheduler_steps` | 1, 2, 4, 8 | - |
| `quantization` | None, "fp8" | - |
| `speculative_model` | None, "ngram" | - |
| `num_lookahead_slots` | 1, 2, 4 | If speculative=ngram |
| `ngram_prompt_lookup_max` | 2, 4, 8 | If speculative=ngram |
| `ngram_prompt_lookup_min` | 1, 2 | If speculative=ngram |
| `attention_backend` | None, "flash_attn", "flashinfer" | - |

**Total configs: ~36,480**

### Presets

```python
# Minimal: 108 configs
--config-preset minimal

# Scheduling-only: 320 configs  
--config-preset scheduling_only

# Comprehensive: ~36K configs (default)
--config-preset comprehensive
```

---

## Command-Line Options

```bash
python scripts/run_rl_optimizer.py \
  --model "microsoft/Phi-3-mini-4k-instruct" \  # Model to optimize
  --port 8000 \                                  # vLLM server port
  --data-dir "./data_rl" \                       # Output directory
  --max-episodes 100 \                           # Max configs to test (ignored with --convergence)
  --rate 128 \                                   # Request rate (RPS)
  --max-requests 100 \                           # Requests per benchmark
  --latency-tolerance 0.05 \                     # 5% max degradation (relative mode)
  --max-latency-ms 100 \                         # Absolute max p50 latency in ms (overrides tolerance)
  --config-preset comprehensive \                # Config space size
  --dashboard terminal \                         # Live dashboard mode
  --seed 42 \                                    # Random seed
  --convergence \                                # Run until convergence
  --convergence-window 20 \                      # Episodes of stability needed
  --max-convergence-episodes 10000               # Safety limit for convergence mode
```

### Latency Constraint Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--latency-tolerance` | `0.05` | Max allowed latency degradation as ratio (0.05 = 5%). Used when `--max-latency-ms` is not set. |
| `--max-latency-ms` | `None` | **Absolute** maximum p50 latency in milliseconds. If set, overrides `--latency-tolerance`. |

**Choosing between modes:**
- Use `--latency-tolerance` when you want to improve throughput while maintaining "similar" latency to baseline
- Use `--max-latency-ms` when you have a strict SLA requirement (e.g., "p50 must be < 100ms")

### Dashboard Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `terminal` | Full live dashboard with progress bar, recent episodes, top configs | **Default** - Complete visibility |
| `plot` | Real-time matplotlib plots (4 panels: reward, best reward, throughput/latency, histogram) | Visual learners, presentations |
| `simple` | Minimal progress bar with best reward indicator | Quick tests, logging to file |
| `none` | Text-only output | Debugging, CI/CD pipelines |

#### Terminal Dashboard (`--dashboard terminal`)

```
========================================================================
                    🔥 RL VLLM OPTIMIZER - LIVE DASHBOARD 🔥                    
========================================================================

Progress: [████████████████████░░░░░░░░░░░░░░░░░░░░] 50/100 (50.0%)
Elapsed: 12:34 | Rate: 4.0 ep/min | ETA: 12:34
Configs tried: 48/36480 (0.1%)
Best reward: 1.450 | Avg posterior var: 0.234

🏆 BEST CONFIG (reward=1.450):
  {"max_num_seqs": 512, "enable_chunked_prefill": true, "num_scheduler_steps": 4}

📊 LAST 5 EPISODES:
  Ep   Reward  Throughput    Latency Config
----------------------------------------------------------------------
  46    0.923        48.2      102.3 max_num_seqs=256
  47    1.340 ⭐      70.1       98.5 max_num_seqs=512, enable_chunked_prefill=True
  48    0.000         0.0        inf max_num_seqs=1024
  49    1.120        58.6       95.2 max_num_seqs=512, num_scheduler_steps=2
  50    1.280        67.0       97.1 max_num_seqs=512, enable_chunked_prefill=true

🥇 TOP 3 CONFIGS (by posterior mean):
  1. 1.420: {"max_num_seqs": 512, "enable_chunked_prefill": true, ...}
  2. 1.280: {"max_num_seqs": 512, "num_scheduler_steps": 4, ...}
  3. 1.150: {"max_num_seqs": 256, "speculative_model": "ngram", ...}

========================================================================
```

#### Plot Dashboard (`--dashboard plot`)

Opens a matplotlib window with 4 live-updating panels:
1. **Reward per Episode** - Individual episode rewards with baseline reference
2. **Best Reward Over Time** - Cumulative maximum (convergence curve)
3. **Throughput vs Latency** - Scatter plot colored by reward quality
4. **Reward Distribution** - Histogram showing exploration progress

Requires: `pip install matplotlib`

---

## Output Files

```
data_rl/
├── episodes.jsonl              # One line per episode (detailed log)
├── bandit_state.json           # Full bandit state (can resume)
├── final_result.json           # Summary with best config
└── state.db                    # SQLite checkpoint data
```

### Example final_result.json

```json
{
  "total_episodes": 100,
  "best_config": {
    "max_num_seqs": 512,
    "enable_chunked_prefill": true,
    "num_scheduler_steps": 4,
    "speculative_model": "ngram",
    "num_lookahead_slots": 4
  },
  "best_reward": 1.45,
  "improvement_pct": 45.0,
  "baseline": {
    "throughput": 52.3,
    "latency_p50": 95.2
  },
  "top_5_configs": [...],
  "bandit_stats": {
    "configs_tried": 89,
    "coverage_pct": 0.24,
    "avg_posterior_var": 0.08
  }
}
```

---

## Convergence Mode

### Automatic Convergence Detection (`--convergence`)

Instead of running a fixed number of episodes, you can run until the optimizer converges on a stable best configuration:

```bash
# Run until convergence (default settings)
python scripts/run_rl_optimizer.py --convergence

# Customize convergence criteria
python scripts/run_rl_optimizer.py \
  --convergence \
  --convergence-window 30 \              # Require 30 stable episodes
  --max-convergence-episodes 500         # Safety limit
```

### Convergence Criteria

The optimizer stops when **ALL** of the following are met:

1. **Best config stability**: Same config has been best for `convergence_window` episodes (default: 20)
2. **Low uncertainty**: Average posterior variance < 0.1
3. **Minimum exploration**: At least 100 episodes completed

Or when **ANY** of the following safety limits are hit:

- `--max-convergence-episodes` (default: 10,000)
- 50% of config space explored
- 10% coverage with very low variance (< 0.05)

### Convergence vs Fixed Episodes

| Mode | Flag | Best For | Typical Duration |
|------|------|----------|------------------|
| Fixed | `--max-episodes N` | Known time budget | N × 2 minutes |
| Convergence | `--convergence` | Best results, unknown time | 50-200 episodes |

**Recommendation:**
- Use `--max-episodes` for initial testing and CI/CD
- Use `--convergence` for final optimization runs
- Comprehensive preset typically converges in **50-100 episodes**
- Minimal preset typically converges in **20-30 episodes**

### Monitoring Convergence

Watch for these indicators in the terminal dashboard:

```
Best reward: 1.450 | Avg posterior var: 0.08
```

- **Avg posterior var < 0.1**: Approaching convergence
- **Avg posterior var < 0.05**: Well converged
- **Best reward stable for 20+ episodes**: Ready to stop

Example convergence timeline:

```
Episode 1-20:   Exploring diverse configs, high variance
Episode 21-40:  Focusing on promising region (chunked_prefill + higher batch sizes)
Episode 41-60:  Fine-tuning within region, variance decreasing
Episode 61-80:  Stable best config, posterior var < 0.1
Episode 82:     🎯 CONVERGENCE DETECTED
```

### Warm-Start with Speculative Decoding (`--warm-start-speculative`)

If you want to prioritize speculative decoding configs before random exploration:

```bash
python scripts/run_rl_optimizer.py --convergence --warm-start-speculative
```

This tries 5 promising speculative configs **first**, then switches to Thompson Sampling:

1. `max_num_seqs=256, speculative_model=ngram, num_lookahead_slots=2`
2. `max_num_seqs=256, speculative_model=ngram, num_lookahead_slots=4`
3. `max_num_seqs=512, speculative_model=ngram, num_lookahead_slots=4`
4. `max_num_seqs=256, chunked_prefill=True, speculative_model=ngram`
5. `max_num_seqs=512, chunked_prefill=True, speculative_model=ngram, ngram_prompt_lookup_max=8`

**When to use:**
- You suspect speculative decoding will help your model
- You want to ensure speculative configs get tested even with low random probability
- Previous runs showed speculative worked well

**Dashboard display:** Warm-start episodes are labeled with `[warm]`:

```
--- Episode 1 [warm] ---
--- Episode 2 [warm] ---
--- Episode 3 ---  (Thompson Sampling starts)
```

---

## Latency-Constrained Optimization

### Maximizing Throughput Within Latency Budget

The primary goal of this optimizer is to **maximize throughput while respecting latency constraints**. This is a classic constrained optimization problem:

```
Maximize:   throughput(config)
Subject to: latency_p50(config) ≤ max_latency_threshold
```

### How It Works

1. **Baseline Establishment**: Run default config to establish baseline throughput and latency
2. **Threshold Calculation**:
   - Relative mode: `threshold = baseline_latency × (1 + tolerance)`
   - Absolute mode: `threshold = max_latency_ms`
3. **Constrained Search**: RL agent only gets reward for configs under threshold
4. **Throughput Maximization**: Among valid configs, highest throughput wins

### Use Cases

#### 1. Strict SLA Requirements
You have a hard requirement (e.g., "p50 latency must be < 50ms"):

```bash
python scripts/run_rl_optimizer.py \
  --max-latency-ms 50 \
  --convergence \
  --config-preset comprehensive
```

**Output shows:**
```
📊 BASELINE ESTABLISHED:
  Throughput: 45.2 req/s
  Latency p50: 42.1 ms
  Max allowed latency: 50.0 ms (absolute)

🏆 Best Config Found:
  {"max_num_seqs": 256, "enable_chunked_prefill": true, ...}
  
📈 Improvement:
  Throughput improvement: +35.0%
  Final latency: 48.5 ms (within 50ms limit)
```

#### 2. Balanced Optimization
Allow small latency increase for significant throughput gains:

```bash
python scripts/run_rl_optimizer.py --latency-tolerance 0.10 --convergence
```

This allows up to 10% latency increase to find higher-throughput configs.

#### 3. Production Safety
Set absolute ceiling to never exceed, even if baseline is already close:

```bash
# Never exceed 200ms, regardless of baseline
python scripts/run_rl_optimizer.py --max-latency-ms 200 --convergence
```

### Understanding Results

When using `--max-latency-ms`, the dashboard shows:

```
🏆 BEST CONFIG (reward=1.450):
  {"max_num_seqs": 512, "enable_chunked_prefill": true, "num_scheduler_steps": 4}
  Throughput: 75.8 req/s (+45% vs baseline)
  Latency p50: 95.2 ms (within 100ms limit)
```

Key indicators:
- **Reward > 1.0**: Throughput improved vs baseline
- **Latency shown**: Current p50 vs threshold
- **Within limit**: ✅ Config respects constraint

### Tips for Latency-Constrained Optimization

1. **Start with relative mode** (`--latency-tolerance`) for general improvement
2. **Switch to absolute mode** (`--max-latency-ms`) when you have strict SLAs
3. **Use convergence mode** to automatically find best config within constraint
4. **Check baseline first**: If baseline already exceeds your target, optimize may find nothing

```bash
# Check baseline before full optimization
python scripts/run_rl_optimizer.py --max-episodes 1 --dashboard none
# Then review data_rl/episodes.jsonl for baseline metrics
```

---

## Future Improvements

### 1. Bayesian Optimization with Gaussian Process

**Current limitation:** Each config is independent. Testing `max_num_seqs=256, chunked_prefill=True` provides no information about `max_num_seqs=512, chunked_prefill=True`.

**GP-based approach:**

```python
from sklearn.gaussian_process import GaussianProcessRegressor

# Model configs as points in continuous space
X = encode_configs_as_vectors(configs)  # e.g., [256, 1, 0, 4] 
y = observed_rewards

gp = GaussianProcessRegressor(kernel=Matern())
gp.fit(X, y)

# Acquisition function: where to sample next?
next_config = maximize_expected_improvement(gp)
```

**Benefits:**
- Learns that similar configs have similar rewards
- Explores promising **regions** not just individual configs
- Fewer evaluations needed for same performance

**Implementation sketch:**

```python
# forge/rl/bayesian_optimizer.py
class BayesianOptimizer:
    def __init__(self, config_space):
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=5
        )
        self.X = []  # Config vectors
        self.y = []  # Rewards
    
    def select_config(self):
        if len(self.X) < 5:
            # Random sample first few
            return random.choice(self.config_space)
        
        # Fit GP and maximize acquisition function
        self.gp.fit(self.X, self.y)
        
        def expected_improvement(x):
            mu, sigma = self.gp.predict([x], return_std=True)
            return (mu - max(self.y)) * norm.cdf((mu - max(self.y)) / sigma)
        
        return maximize(expected_improvement, self.config_space)
```

### 2. Multi-Fidelity Optimization

Use cheap approximations to screen configs before expensive benchmarks:

```python
# Level 1: 10 requests, 10 sec (cheap)
# Level 2: 100 requests, 2 min (standard)
# Level 3: 1000 requests, 20 min (expensive, only for best)
```

### 3. Constraint-Aware BO

Explicitly model latency constraint as second GP:

```python
# Two GPs:
# - f_throughput(x): Predicts throughput
# - f_latency(x): Predicts latency

# Feasible region: where f_latency(x) <= 1.05 * baseline
# Optimize: max f_throughput(x) subject to feasible
```

### 4. Meta-Learning

Learn across different models:

```python
# Pre-train on Llama-2, Phi-3, Mistral
# Fine-tune GP prior for new model
```

### 5. Early Stopping with Successive Halving

```python
# Start N configs with short benchmark (10 requests)
# Keep top 50%, double benchmark length
# Repeat until only best config remains
```

---

## Comparison: Thompson Sampling vs Bayesian Optimization

| Feature | Thompson Sampling (Current) | Bayesian Optimization |
|---------|----------------------------|----------------------|
| **Model** | Independent per config | Gaussian Process with kernel |
| **Generalization** | None | Learns similarity between configs |
| **Sample Efficiency** | O(n) configs needed | O(log n) configs needed |
| **Complexity** | Simple, 200 lines | Complex, needs GP library |
| **Assumptions** | None | Smoothness of reward landscape |
| **Best For** | Discrete, non-smooth spaces | Continuous, smooth spaces |

**Recommendation:**
- Current Thompson Sampling is **good enough** for 36K configs × 2 min = 50K min total
- Switch to GP if config space grows >100K or tests become expensive (>10 min each)

---

## Debugging

### View Episode History

```bash
# Pretty-print last 10 episodes
jq -r '. | "\(.episode): reward=\(.reward), throughput=\(.throughput)"' data_rl/episodes.jsonl | tail -10
```

### Resume from Checkpoint

```python
# In runner.py, add at start:
if (self.data_dir / "bandit_state.json").exists():
    self.bandit = ThompsonSamplingBandit.load(self.data_dir / "bandit_state.json")
    print(f"Resumed from checkpoint")
```

### Visualize Convergence

```python
import json
import matplotlib.pyplot as plt

episodes = [json.loads(line) for line in open("data_rl/episodes.jsonl")]
rewards = [e["reward"] for e in episodes]
best_so_far = [max(rewards[:i+1]) for i in range(len(rewards))]

plt.plot(best_so_far)
plt.xlabel("Episode")
plt.ylabel("Best Reward")
plt.title("Convergence")
plt.savefig("convergence.png")
```

---

## References

- **Thompson Sampling:** Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another
- **Gaussian Processes for ML:** Rasmussen & Williams (2006)
- **Bayesian Optimization:** Snoek et al. (2012), Practical Bayesian Optimization of ML Algorithms

---

## License

Same as Inference Forge project.
