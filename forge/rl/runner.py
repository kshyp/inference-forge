"""Main RL runner for vLLM config optimization.

Runs the Thompson Sampling loop:
1. Establish baseline with default config
2. Iteratively select configs, run benchmarks, update beliefs
3. Track convergence and report results
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4

from forge.agents.benchmark.agent import BenchmarkAgent
from forge.core.state import StateStore
from forge.core.events import Task

from .config_space import VLLMConfigSpace
from .bandit import ThompsonSamplingBandit
from .reward import RewardCalculator, BaselineMetrics
from .dashboard import TerminalDashboard, LivePlotter, SimpleProgressBar


@dataclass
class Episode:
    """A single RL episode."""
    episode: int
    timestamp: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    reward: float
    explanation: str
    posterior_mean: float
    posterior_var: float
    config_id: str


@dataclass
class OptimizationResult:
    """Final optimization result."""
    total_episodes: int
    best_config: Dict[str, Any]
    best_reward: float
    baseline_metrics: BaselineMetrics
    improvement_pct: float
    episodes: List[Episode]


class RLRunner:
    """
    Main RL runner for vLLM config optimization.
    
    Workflow:
    1. Run baseline benchmark to establish reference metrics
    2. For each episode:
       a. Select config using Thompson Sampling
       b. Run GuideLLM benchmark
       c. Calculate reward vs baseline
       d. Update posterior beliefs
       e. Log results
    3. Report best config found
    
    Convergence:
    - Monitors posterior variance reduction
    - Tracks best config stability
    - Can run for fixed episodes or until convergence
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        port: int = 8000,
        data_dir: str = "./data_rl",
        max_episodes: int = 1000,
        benchmark_rate: float = 128.0,
        max_requests: int = 100,
        config_space_preset: str = "comprehensive",
        latency_tolerance: float = 0.05,
        max_latency_ms: Optional[float] = None,
        seed: Optional[int] = 42,
        dashboard_mode: str = "terminal",  # "terminal", "plot", "simple", "none"
        run_until_convergence: bool = False,
        convergence_window: int = 20,
        max_episodes_for_convergence: int = 10000,
        warm_start_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize RL runner.
        
        Args:
            model_name: Model to benchmark
            port: vLLM server port
            data_dir: Directory for logs and results
            max_episodes: Maximum number of episodes to run (ignored if run_until_convergence=True)
            benchmark_rate: Fixed request rate for benchmarks
            max_requests: Number of requests per benchmark
            config_space_preset: "minimal", "comprehensive", or "scheduling_only"
            latency_tolerance: Max allowed latency degradation (5% = 0.05)
            max_latency_ms: Absolute max p50 latency in ms (overrides latency_tolerance)
            seed: Random seed for reproducibility
            dashboard_mode: "terminal" (full dashboard), "plot" (matplotlib), "simple" (progress bar), "none"
            run_until_convergence: If True, run until convergence detected (ignores max_episodes)
            convergence_window: Number of episodes with stable best config to consider converged
            max_episodes_for_convergence: Hard limit for convergence mode (safety cap)
            warm_start_configs: Optional list of configs to try before Thompson Sampling
        """
        self.run_until_convergence = run_until_convergence
        self.convergence_window = convergence_window
        self.max_episodes_for_convergence = max_episodes_for_convergence
        self.dashboard_mode = dashboard_mode
        self._effective_max_episodes = max_episodes_for_convergence if run_until_convergence else max_episodes
        self.warm_start_configs = warm_start_configs or []
        self.model_name = model_name
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_episodes = max_episodes
        self.benchmark_rate = benchmark_rate
        self.max_requests = max_requests
        self.latency_tolerance = latency_tolerance
        self.max_latency_ms = max_latency_ms
        self.seed = seed
        
        # Initialize config space
        from .config_space import ConfigSpacePresets
        if config_space_preset == "minimal":
            self.config_space = ConfigSpacePresets.minimal()
        elif config_space_preset == "scheduling_only":
            self.config_space = ConfigSpacePresets.scheduling_only()
        elif config_space_preset == "awq":
            self.config_space = ConfigSpacePresets.awq_quantized()
        elif config_space_preset == "memory_constrained":
            self.config_space = ConfigSpacePresets.memory_constrained()
        else:
            self.config_space = VLLMConfigSpace()
        
        self.configs = self.config_space.enumerate_configs()
        print(f"Config space size: {len(self.configs)}")
        
        # Initialize bandit
        self.bandit = ThompsonSamplingBandit(self.configs, seed=seed)
        
        # Initialize benchmark agent
        self.state_store = StateStore(str(self.data_dir / "state.db"))
        self.benchmark_agent = BenchmarkAgent(
            state_store=self.state_store,
            health_port=38080,
            data_dir=str(data_dir)
        )
        
        # State
        self.baseline: Optional[BaselineMetrics] = None
        self.reward_calc: Optional[RewardCalculator] = None
        self.episodes: List[Episode] = []
        self.log_file: Path = self.data_dir / "episodes.jsonl"
        
        # Convergence tracking
        self.best_config_history: List[Dict[str, Any]] = []
    
    async def run(self) -> OptimizationResult:
        """Run the full RL optimization loop."""
        # Determine effective max episodes
        if self.run_until_convergence:
            effective_max = self.max_episodes_for_convergence
            mode_str = f"convergence (max {effective_max})"
        else:
            effective_max = self.max_episodes
            mode_str = f"{effective_max} episodes"
        
        # Initialize dashboard based on mode
        if self.dashboard_mode == "terminal":
            self.dashboard = TerminalDashboard(effective_max)
            plotter = None
        elif self.dashboard_mode == "plot":
            self.dashboard = SimpleProgressBar(effective_max)
            plotter = LivePlotter(update_interval=5)
            plotter.start()
        elif self.dashboard_mode == "simple":
            self.dashboard = SimpleProgressBar(effective_max)
            plotter = None
        else:  # "none"
            self.dashboard = None
            plotter = None
        
        if self.dashboard and hasattr(self.dashboard, 'start'):
            self.dashboard.start()
        else:
            print("="*80)
            print("  🔥 RL-BASED VLLM CONFIG OPTIMIZATION 🔥")
            print("="*80)
            print(f"\nModel: {self.model_name}")
            print(f"Config Space: {len(self.configs)} configurations")
            print(f"Mode: {mode_str}")
            print(f"Latency Tolerance: {self.latency_tolerance*100:.0f}%")
            print(f"Benchmark Rate: {self.benchmark_rate} RPS")
            print(f"Requests per Benchmark: {self.max_requests}")
            print(f"\nAlgorithm: Thompson Sampling with Gaussian posteriors")
            print(f"Reward: Throughput ratio (capped at 2.0)")
            print(f"Constraint: Latency p50 ≤ {self.latency_tolerance*100:.0f}% degradation")
            if self.run_until_convergence:
                print(f"Convergence: {self.convergence_window} episodes stable + low variance")
            print("="*80)
        
        # Start health server
        await self.benchmark_agent.start_health()
        
        try:
            # Phase 1: Establish baseline
            await self._run_baseline()
            
            # Phase 2: Warm start (if specified)
            episode_num = 0
            for warm_config in self.warm_start_configs:
                episode_num += 1
                if episode_num > effective_max:
                    break
                should_stop = await self._run_episode_with_config(
                    episode_num, warm_config, plotter, label="warm"
                )
                if should_stop:
                    break
            
            # Phase 3: RL exploration
            for episode_num in range(episode_num + 1, effective_max + 1):
                should_stop = await self._run_episode(episode_num, plotter)
                
                if should_stop:
                    if self.dashboard and hasattr(self.dashboard, '_clear_screen'):
                        pass  # Dashboard will show final state
                    else:
                        print(f"\n🎯 Convergence detected at episode {episode_num}")
                    break
                
                # In convergence mode, also check if we've explored enough of the space
                if self.run_until_convergence and episode_num > 100:
                    stats = self.bandit.get_stats()
                    # If we've tried >50% of configs or coverage >10% with good confidence, stop
                    if (stats['configs_tried'] > len(self.configs) * 0.5 or 
                        (stats['coverage_pct'] > 10 and stats['avg_posterior_var'] < 0.05)):
                        if not self.dashboard or self.dashboard_mode == "none":
                            print(f"\n🎯 Sufficient exploration at episode {episode_num}")
                        break
        
        except KeyboardInterrupt:
            if not self.dashboard or self.dashboard_mode == "none":
                print("\n\n⚠️ Interrupted by user")
        
        except Exception as e:
            if not self.dashboard or self.dashboard_mode == "none":
                print(f"\n\n❌ Error: {e}")
            result = None
        
        finally:
            await self.benchmark_agent.stop_health()
            
            # Only create result if baseline was established
            if self.baseline is not None:
                result = self._create_final_result()
            
            self._save_results(result)
            
            if self.dashboard and hasattr(self.dashboard, 'finish'):
                self.dashboard.finish(result)
            if plotter:
                plotter.finish()
        
        return result
    
    async def _run_baseline(self):
        """Run baseline benchmark with default config."""
        print("\n" + "="*80)
        print("PHASE 1: ESTABLISHING BASELINE")
        print("="*80)
        
        # Default config (conservative for memory-constrained GPUs)
        default_config = {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.75,
            # max_model_len is not set - let vLLM derive from model config
        }
        
        print(f"\nRunning baseline with default config:")
        print(f"  {json.dumps(default_config, indent=2)}")
        
        result = await self._run_benchmark(default_config, "baseline")
        
        if result is None:
            raise RuntimeError("Baseline benchmark failed - cannot continue")
        
        self.baseline = BaselineMetrics.from_benchmark_result(result)
        self.reward_calc = RewardCalculator(
            self.baseline,
            latency_tolerance=self.latency_tolerance,
            max_latency_ms=self.max_latency_ms
        )
        
        print(f"\n📊 BASELINE ESTABLISHED:")
        print(f"  Throughput: {self.baseline.throughput:.1f} req/s")
        print(f"  Latency p50: {self.baseline.latency_p50:.1f} ms")
        print(f"  Latency p99: {self.baseline.latency_p99:.1f} ms")
        print(f"  TTFT p50: {self.baseline.ttft_p50:.1f} ms")
        print(f"  TPOT p50: {self.baseline.tpot_p50:.1f} ms")
        if self.max_latency_ms:
            print(f"  Max allowed latency: {self.reward_calc.max_latency_threshold:.1f} ms (absolute)")
        else:
            print(f"  Max allowed latency: {self.reward_calc.max_latency_threshold:.1f} ms ({self.latency_tolerance*100:.0f}% tolerance)")
        print("="*80)
    
    async def _run_episode(self, episode_num: int, plotter=None) -> bool:
        """
        Run a single episode.
        
        Args:
            episode_num: Current episode number
            plotter: Optional LivePlotter for visualization
        
        Returns:
            True if convergence detected, False otherwise
        """
        # Only print episode header in non-dashboard modes
        if not self.dashboard or self.dashboard_mode == "none":
            print(f"\n--- Episode {episode_num}/{self.max_episodes} ---")
        
        # Select config using Thompson Sampling
        config = self.bandit.select_config(explore=True)
        config_id = self.config_space.get_config_id(config)
        
        if not self.dashboard or self.dashboard_mode == "none":
            print(f"Selected config: {json.dumps(config)}")
        
        # Run benchmark
        result = await self._run_benchmark(config, f"ep{episode_num}")
        
        if result is None:
            if not self.dashboard or self.dashboard_mode == "none":
                print("❌ Benchmark failed - assigning zero reward")
            metrics = {"throughput": 0, "latency_p50": float('inf'), "failed_requests": 1}
            reward = 0.0
            explanation = "Benchmark failed"
        else:
            # Extract metrics
            metrics = self._extract_metrics(result)
            reward, explanation = self.reward_calc.calculate_with_explanation(metrics)
            if not self.dashboard or self.dashboard_mode == "none":
                print(f"🎯 {explanation}")
        
        # Update bandit
        self.bandit.update(config, reward)
        
        # Get updated posterior
        config_stats = self.bandit.get_config_stats(config)
        posterior = config_stats.posterior if config_stats else None
        
        # Log episode
        episode = Episode(
            episode=episode_num,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics=metrics,
            reward=reward,
            explanation=explanation,
            posterior_mean=posterior.mean if posterior else 0.0,
            posterior_var=posterior.var if posterior else 1.0,
            config_id=config_id,
        )
        self.episodes.append(episode)
        self._log_episode(episode)
        
        # Update dashboard/plotter
        if self.dashboard:
            if hasattr(self.dashboard, 'update'):
                # Terminal dashboard needs full stats
                if self.dashboard_mode == "terminal":
                    self.dashboard.update(
                        episode=episode_num,
                        reward=reward,
                        metrics=metrics,
                        config=config,
                        bandit_stats=self.bandit.get_stats(),
                        top_configs=self.bandit.get_top_configs(3),
                    )
                else:
                    # Simple progress bar
                    self.dashboard.update(episode_num, reward, metrics)
        
        if plotter:
            plotter.update(episode_num, reward, metrics)
        
        # Check convergence
        return self._check_convergence()
    
    async def _run_episode_with_config(
        self,
        episode_num: int,
        config: Dict[str, Any],
        plotter=None,
        label: str = "warm"
    ) -> bool:
        """
        Run an episode with a specific config (for warm-start).
        
        Similar to _run_episode but uses provided config instead of selecting.
        """
        if not self.dashboard or self.dashboard_mode == "none":
            print(f"\n--- Episode {episode_num} [{label}] ---")
            print(f"Using config: {json.dumps(config)}")
        
        config_id = self.config_space.get_config_id(config)
        
        # Run benchmark
        result = await self._run_benchmark(config, f"{label}{episode_num}")
        
        if result is None:
            if not self.dashboard or self.dashboard_mode == "none":
                print("❌ Benchmark failed - assigning zero reward")
            metrics = {"throughput": 0, "latency_p50": float('inf'), "failed_requests": 1}
            reward = 0.0
            explanation = "Benchmark failed"
        else:
            # Extract metrics
            metrics = self._extract_metrics(result)
            reward, explanation = self.reward_calc.calculate_with_explanation(metrics)
            if not self.dashboard or self.dashboard_mode == "none":
                print(f"🎯 {explanation}")
        
        # Update bandit
        self.bandit.update(config, reward)
        
        # Get updated posterior
        config_stats = self.bandit.get_config_stats(config)
        posterior = config_stats.posterior if config_stats else None
        
        # Log episode
        episode = Episode(
            episode=episode_num,
            timestamp=datetime.now().isoformat(),
            config=config,
            metrics=metrics,
            reward=reward,
            explanation=explanation,
            posterior_mean=posterior.mean if posterior else 0.0,
            posterior_var=posterior.var if posterior else 1.0,
            config_id=config_id,
        )
        self.episodes.append(episode)
        self._log_episode(episode)
        
        # Update dashboard/plotter
        if self.dashboard:
            if hasattr(self.dashboard, 'update'):
                if self.dashboard_mode == "terminal":
                    self.dashboard.update(
                        episode=episode_num,
                        reward=reward,
                        metrics=metrics,
                        config=config,
                        bandit_stats=self.bandit.get_stats(),
                        top_configs=self.bandit.get_top_configs(3),
                    )
                else:
                    self.dashboard.update(episode_num, reward, metrics)
        
        if plotter:
            plotter.update(episode_num, reward, metrics)
        
        # Check convergence
        return self._check_convergence()
    
    async def _run_benchmark(
        self,
        config: Dict[str, Any],
        label: str
    ) -> Optional[Dict[str, Any]]:
        """Run a single benchmark with given config."""
        task = Task(
            type="benchmark",
            payload={
                "experiment_id": str(uuid4()),
                "model_name": self.model_name,
                "vllm_config": {
                    "port": self.port,
                    "startup_timeout_seconds": 300,
                    **config
                },
                "guidellm_config": {
                    "dataset": "wikitext",
                    "prompt_tokens": 128,
                    "output_tokens": 64,
                    "max_requests": self.max_requests,
                }
            }
        )
        
        result = await self.benchmark_agent.run_single_task(task)
        
        if not result.get("_success"):
            print(f"  Benchmark failed: {result.get('_error')}")
            return None
        
        return result
    
    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from benchmark result."""
        metrics_data = result.get("benchmark_result", {}).get("metrics", {})
        
        def get_percentile(metric: str, p: str) -> float:
            val = metrics_data.get(metric, {}).get(p, 0)
            return val * 1000 if val and val < 10 else val
        
        return {
            "throughput": metrics_data.get("requests_per_sec", 0),
            "latency_p50": get_percentile("itl_ms", "p50"),
            "latency_p99": get_percentile("itl_ms", "p99"),
            "ttft_p50": get_percentile("ttft_ms", "p50"),
            "tpot_p50": get_percentile("tpot_ms", "p50"),
            "failed_requests": 0,  # Would be non-zero if benchmark failed
        }
    
    def _log_episode(self, episode: Episode):
        """Append episode to log file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "episode": episode.episode,
                "timestamp": episode.timestamp,
                "config": episode.config,
                "reward": episode.reward,
                "throughput": episode.metrics.get("throughput", 0),
                "latency_p50": episode.metrics.get("latency_p50", 0),
                "posterior_mean": episode.posterior_mean,
                "posterior_var": episode.posterior_var,
                "explanation": episode.explanation,
            }, default=str) + "\n")
    
    def _check_convergence(self) -> bool:
        """
        Check if optimization has converged.
        
        Convergence criteria:
        - Best config hasn't changed for N episodes
        - Average posterior variance is low (< 0.1)
        """
        stats = self.bandit.get_stats()
        
        # Track best config
        current_best = self.bandit.get_best_config()
        self.best_config_history.append(current_best)
        
        if len(self.best_config_history) < self.convergence_window:
            return False
        
        # Keep only recent history
        self.best_config_history = self.best_config_history[-self.convergence_window:]
        
        # Check if best config is stable
        best_config_str = self.config_space.get_config_id(current_best)
        stable = all(
            self.config_space.get_config_id(c) == best_config_str
            for c in self.best_config_history
        )
        
        # Check if variance is low
        low_variance = stats["avg_posterior_var"] < 0.1
        
        return stable and low_variance
    
    def _print_progress(self):
        """Print current progress summary."""
        stats = self.bandit.get_stats()
        best = self.bandit.get_best_config()
        best_stats = self.bandit.get_config_stats(best)
        
        print("\n" + "-"*60)
        print(f"PROGRESS (Episode {len(self.episodes)})")
        print("-"*60)
        print(f"Configs tried: {stats['configs_tried']}/{stats['config_space_size']} "
              f"({stats['coverage_pct']:.1f}%)")
        print(f"Best posterior mean: {stats['best_posterior_mean']:.3f}")
        print(f"Avg posterior var: {stats['avg_posterior_var']:.3f}")
        print(f"\nBest config:")
        print(f"  {json.dumps(best, indent=2)}")
        if best_stats and best_stats.rewards:
            print(f"  Rewards: {best_stats.rewards}")
            print(f"  Avg reward: {best_stats.avg_reward:.3f}")
        print("-"*60)
    
    def _create_final_result(self) -> OptimizationResult:
        """Create final optimization result."""
        best_config = self.bandit.get_best_config()
        best_stats = self.bandit.get_config_stats(best_config)
        best_reward = best_stats.best_reward if best_stats else 0.0
        
        improvement_pct = (best_reward - 1.0) * 100 if best_reward > 0 else 0.0
        
        return OptimizationResult(
            total_episodes=len(self.episodes),
            best_config=best_config,
            best_reward=best_reward,
            baseline_metrics=self.baseline,
            improvement_pct=improvement_pct,
            episodes=self.episodes,
        )
    
    def _save_results(self, result: Optional[OptimizationResult]):
        """Save final results to disk."""
        # Save bandit state even if failed
        self.bandit.save(self.data_dir / "bandit_state.json")
        
        if result is None:
            print("\n" + "="*80)
            print("  ❌ OPTIMIZATION FAILED")
            print("="*80)
            print("\nBaseline benchmark failed - no results to save")
            print(f"\n💾 Partial state saved to: {self.data_dir}")
            print("="*80)
            return
        
        # Save summary
        summary = {
            "total_episodes": result.total_episodes,
            "best_config": result.best_config,
            "best_reward": result.best_reward,
            "improvement_pct": result.improvement_pct,
            "baseline": {
                "throughput": result.baseline_metrics.throughput,
                "latency_p50": result.baseline_metrics.latency_p50,
                "latency_p99": result.baseline_metrics.latency_p99,
            },
            "top_5_configs": [
                {
                    "config": c,
                    "posterior_mean": m,
                }
                for c, m in self.bandit.get_top_configs(5)
            ],
            "bandit_stats": self.bandit.get_stats(),
        }
        
        with open(self.data_dir / "final_result.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print final summary
        print("\n" + "="*80)
        print("  🏆 OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"\nTotal Episodes: {result.total_episodes}")
        print(f"\n📊 Baseline Performance:")
        print(f"  Throughput: {result.baseline_metrics.throughput:.1f} req/s")
        print(f"  Latency p50: {result.baseline_metrics.latency_p50:.1f} ms")
        print(f"\n🎯 Best Config Found:")
        print(f"  {json.dumps(result.best_config, indent=2)}")
        print(f"\n📈 Improvement:")
        print(f"  Best reward: {result.best_reward:.3f}")
        print(f"  Throughput improvement: {result.improvement_pct:+.1f}%")
        print(f"\n💾 Results saved to: {self.data_dir}")
        print("="*80)


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RL-based vLLM Config Optimizer"
    )
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct",
                       help="Model to optimize")
    parser.add_argument("--port", type=int, default=8000,
                       help="vLLM server port")
    parser.add_argument("--data-dir", default="./data_rl",
                       help="Data directory for logs")
    parser.add_argument("--max-episodes", type=int, default=100,
                       help="Maximum episodes to run")
    parser.add_argument("--rate", type=float, default=128.0,
                       help="Benchmark request rate (RPS)")
    parser.add_argument("--max-requests", type=int, default=100,
                       help="Requests per benchmark")
    parser.add_argument("--latency-tolerance", type=float, default=0.05,
                       help="Max latency degradation (0.05 = 5%%)")
    parser.add_argument("--max-latency-ms", type=float, default=None,
                       help="Absolute max p50 latency in ms (overrides --latency-tolerance)")
    parser.add_argument("--config-preset", default="comprehensive",
                       choices=["minimal", "comprehensive", "scheduling_only", "awq", "memory_constrained"],
                       help="Config space preset: comprehensive=36K configs, minimal=108, "
                            "awq=for AWQ quantized models, memory_constrained=<16GB GPUs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dashboard", default="terminal",
                       choices=["terminal", "plot", "simple", "none"],
                       help="Dashboard mode: terminal=full live dashboard, "
                            "plot=matplotlib graphs, simple=progress bar, none=quiet")
    parser.add_argument("--convergence", action="store_true",
                       help="Run until convergence detected (ignores --max-episodes)")
    parser.add_argument("--convergence-window", type=int, default=20,
                       help="Episodes of stability to consider converged (default: 20)")
    parser.add_argument("--max-convergence-episodes", type=int, default=10000,
                       help="Safety limit for convergence mode (default: 10000)")
    parser.add_argument("--warm-start-speculative", action="store_true",
                       help="Try speculative decoding configs before Thompson Sampling")
    
    args = parser.parse_args()
    
    # Build warm start configs if requested
    warm_start_configs = None
    if args.warm_start_speculative:
        warm_start_configs = [
            # Good baseline + speculative variants
            {"max_num_seqs": 256, "max_num_batched_tokens": 2048, "speculative_model": "ngram", "num_lookahead_slots": 2},
            {"max_num_seqs": 256, "max_num_batched_tokens": 2048, "speculative_model": "ngram", "num_lookahead_slots": 4},
            {"max_num_seqs": 512, "max_num_batched_tokens": 4096, "speculative_model": "ngram", "num_lookahead_slots": 4},
            {"max_num_seqs": 256, "max_num_batched_tokens": 2048, "enable_chunked_prefill": True, "speculative_model": "ngram", "num_lookahead_slots": 2},
            {"max_num_seqs": 512, "enable_chunked_prefill": True, "speculative_model": "ngram", "num_lookahead_slots": 4, "ngram_prompt_lookup_max": 8},
        ]
    
    runner = RLRunner(
        model_name=args.model,
        port=args.port,
        data_dir=args.data_dir,
        max_episodes=args.max_episodes,
        benchmark_rate=args.rate,
        max_requests=args.max_requests,
        config_space_preset=args.config_preset,
        latency_tolerance=args.latency_tolerance,
        max_latency_ms=args.max_latency_ms,
        seed=args.seed,
        dashboard_mode=args.dashboard,
        run_until_convergence=args.convergence,
        convergence_window=args.convergence_window,
        max_episodes_for_convergence=args.max_convergence_episodes,
        warm_start_configs=warm_start_configs,
    )
    
    result = await runner.run()
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
