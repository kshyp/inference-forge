"""Real-time dashboard for RL optimizer.

Provides live terminal and web-based visualization of optimization progress.
"""

import json
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class EpisodeSummary:
    """Summary of a single episode for display."""
    episode: int
    reward: float
    throughput: float
    latency_p50: float
    config_preview: str


class TerminalDashboard:
    """
    Live terminal dashboard for RL optimization.
    
    Updates in-place using ANSI escape codes for clean display.
    """
    
    def __init__(self, max_episodes: int):
        self.max_episodes = max_episodes
        self.terminal_width = shutil.get_terminal_size().columns
        self.episodes: List[EpisodeSummary] = []
        self.best_reward = 0.0
        self.best_config: Optional[Dict[str, Any]] = None
        self.start_time: Optional[datetime] = None
    
    def start(self):
        """Initialize dashboard."""
        self.start_time = datetime.now()
        self._clear_screen()
        print("Starting optimization...\n")
    
    def update(
        self,
        episode: int,
        reward: float,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        bandit_stats: Dict[str, Any],
        top_configs: List[tuple],
    ):
        """Update dashboard with latest episode."""
        # Track best
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_config = config
        
        # Add to history
        summary = EpisodeSummary(
            episode=episode,
            reward=reward,
            throughput=metrics.get("throughput", 0),
            latency_p50=metrics.get("latency_p50", 0),
            config_preview=self._format_config(config),
        )
        self.episodes.append(summary)
        
        # Render
        self._render(bandit_stats, top_configs)
    
    def _render(self, bandit_stats: Dict[str, Any], top_configs: List[tuple]):
        """Render dashboard to terminal."""
        self._clear_screen()
        
        lines = []
        width = min(self.terminal_width, 100)
        
        # Header
        lines.append("=" * width)
        lines.append("  🔥 RL VLLM OPTIMIZER - LIVE DASHBOARD 🔥".center(width))
        lines.append("=" * width)
        
        # Progress bar
        progress = len(self.episodes) / self.max_episodes
        bar_width = 40
        filled = int(progress * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"\nProgress: [{bar}] {len(self.episodes)}/{self.max_episodes} ({progress*100:.1f}%)")
        
        # Stats
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        eps_per_min = (len(self.episodes) / (elapsed / 60)) if elapsed > 0 else 0
        eta_mins = (self.max_episodes - len(self.episodes)) / eps_per_min if eps_per_min > 0 else 0
        
        lines.append(f"Elapsed: {self._format_time(elapsed)} | "
                    f"Rate: {eps_per_min:.1f} ep/min | "
                    f"ETA: {self._format_time(eta_mins * 60)}")
        
        lines.append(f"Configs tried: {bandit_stats['configs_tried']}/{bandit_stats['config_space_size']} "
                    f"({bandit_stats['coverage_pct']:.1f}%)")
        lines.append(f"Best reward: {self.best_reward:.3f} | "
                    f"Avg posterior var: {bandit_stats['avg_posterior_var']:.3f}")
        
        # Best config
        if self.best_config:
            lines.append(f"\n🏆 BEST CONFIG (reward={self.best_reward:.3f}):")
            config_str = json.dumps(self.best_config, default=str)
            if len(config_str) > width - 4:
                config_str = config_str[:width-7] + "..."
            lines.append(f"  {config_str}")
        
        # Recent episodes (last 5)
        lines.append(f"\n📊 LAST {min(5, len(self.episodes))} EPISODES:")
        lines.append(f"{'Ep':>4} {'Reward':>8} {'Throughput':>12} {'Latency':>10} Config")
        lines.append("-" * width)
        
        for ep in self.episodes[-5:]:
            reward_str = f"{ep.reward:.3f}"
            if ep.reward == self.best_reward:
                reward_str += " ⭐"
            lines.append(f"{ep.episode:>4} {reward_str:>8} {ep.throughput:>11.1f} "
                        f"{ep.latency_p50:>9.1f} {ep.config_preview}")
        
        # Top configs
        if top_configs:
            lines.append(f"\n🥇 TOP {min(3, len(top_configs))} CONFIGS (by posterior mean):")
            for i, (config, mean) in enumerate(top_configs[:3], 1):
                config_str = json.dumps(config, default=str)
                if len(config_str) > width - 25:
                    config_str = config_str[:width-28] + "..."
                lines.append(f"  {i}. {mean:.3f}: {config_str}")
        
        lines.append("\n" + "=" * width)
        
        print("\n".join(lines))
    
    def _clear_screen(self):
        """Clear terminal using ANSI escape codes."""
        print("\033[2J\033[H", end="")
    
    def _format_config(self, config: Dict[str, Any]) -> str:
        """Format config for display."""
        # Prioritize showing speculative status since it's important
        parts = []
        
        # Always show speculative first if present
        if config.get("speculative_model"):
            spec = config["speculative_model"]
            slots = config.get("num_lookahead_slots", "?")
            parts.append(f"SPEC={spec}({slots})")
        
        # Show other non-default values
        for k, v in sorted(config.items()):
            if v is not None and v is not False and k not in ["speculative_model", "num_lookahead_slots"]:
                # Abbreviate long keys
                key = k.replace("max_num_", "").replace("enable_", "").replace("_tokens", "")
                parts.append(f"{key}={v}")
        
        return ", ".join(parts)[:55]
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 3600:
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        else:
            return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"
    
    def finish(self, result: Any):
        """Show final results."""
        self._clear_screen()
        print("\n" + "=" * 70)
        print("  🏆 OPTIMIZATION COMPLETE")
        print("=" * 70)
        print(f"\nTotal episodes: {result.total_episodes}")
        print(f"Best reward: {result.best_reward:.3f}")
        print(f"Improvement: {result.improvement_pct:+.1f}%")
        print(f"\nBest config:")
        print(json.dumps(result.best_config, indent=2))
        print("=" * 70)


class LivePlotter:
    """
    Real-time matplotlib plot of optimization progress.
    
    Shows:
    - Reward over episodes
    - Best reward so far (cumulative max)
    - Throughput vs latency scatter
    """
    
    def __init__(self, update_interval: int = 5):
        """
        Initialize plotter.
        
        Args:
            update_interval: Update plot every N episodes
        """
        self.update_interval = update_interval
        self.episodes: List[int] = []
        self.rewards: List[float] = []
        self.best_rewards: List[float] = []
        self.throughputs: List[float] = []
        self.latencies: List[float] = []
        self._fig = None
        self._axes = None
    
    def start(self):
        """Initialize matplotlib plot."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation
            
            plt.ion()  # Interactive mode
            self._fig, self._axes = plt.subplots(2, 2, figsize=(12, 8))
            self._fig.suptitle("RL Optimizer - Live Progress")
            
            # Subplot titles
            self._axes[0, 0].set_title("Reward per Episode")
            self._axes[0, 0].set_xlabel("Episode")
            self._axes[0, 0].set_ylabel("Reward")
            
            self._axes[0, 1].set_title("Best Reward Over Time")
            self._axes[0, 1].set_xlabel("Episode")
            self._axes[0, 1].set_ylabel("Best Reward")
            
            self._axes[1, 0].set_title("Throughput vs Latency")
            self._axes[1, 0].set_xlabel("Latency p50 (ms)")
            self._axes[1, 0].set_ylabel("Throughput (req/s)")
            
            self._axes[1, 1].set_title("Reward Distribution")
            self._axes[1, 1].set_xlabel("Reward")
            self._axes[1, 1].set_ylabel("Count")
            
            plt.tight_layout()
            plt.show(block=False)
            
        except ImportError:
            print("Warning: matplotlib not installed. Live plotting disabled.")
            print("Install with: pip install matplotlib")
    
    def update(
        self,
        episode: int,
        reward: float,
        metrics: Dict[str, Any],
    ):
        """Update plot data."""
        if self._fig is None:
            return
        
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.best_rewards.append(max(self.best_rewards + [reward]))
        self.throughputs.append(metrics.get("throughput", 0))
        self.latencies.append(metrics.get("latency_p50", 0))
        
        # Only update plot every N episodes to avoid slowdown
        if episode % self.update_interval != 0:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Clear and redraw
            for ax in self._axes.flat:
                ax.clear()
            
            # Reward per episode
            self._axes[0, 0].plot(self.episodes, self.rewards, 'b-', alpha=0.5, label='Reward')
            self._axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            self._axes[0, 0].set_title("Reward per Episode")
            self._axes[0, 0].set_xlabel("Episode")
            self._axes[0, 0].set_ylabel("Reward")
            self._axes[0, 0].legend()
            self._axes[0, 0].grid(True, alpha=0.3)
            
            # Best reward over time
            self._axes[0, 1].plot(self.episodes, self.best_rewards, 'g-', linewidth=2)
            self._axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            self._axes[0, 1].set_title("Best Reward Over Time")
            self._axes[0, 1].set_xlabel("Episode")
            self._axes[0, 1].set_ylabel("Best Reward")
            self._axes[0, 1].legend()
            self._axes[0, 1].grid(True, alpha=0.3)
            
            # Throughput vs latency scatter
            colors = ['green' if r > 1.0 else 'red' for r in self.rewards]
            self._axes[1, 0].scatter(self.latencies, self.throughputs, c=colors, alpha=0.6)
            self._axes[1, 0].set_title("Throughput vs Latency")
            self._axes[1, 0].set_xlabel("Latency p50 (ms)")
            self._axes[1, 0].set_ylabel("Throughput (req/s)")
            self._axes[1, 0].grid(True, alpha=0.3)
            
            # Reward histogram
            self._axes[1, 1].hist(self.rewards, bins=20, alpha=0.7, edgecolor='black')
            self._axes[1, 1].axvline(x=1.0, color='r', linestyle='--', label='Baseline')
            self._axes[1, 1].set_title("Reward Distribution")
            self._axes[1, 1].set_xlabel("Reward")
            self._axes[1, 1].set_ylabel("Count")
            self._axes[1, 1].legend()
            
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            
        except Exception as e:
            # Silently ignore plot errors
            pass
    
    def finish(self):
        """Keep plot open at end."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()


class SimpleProgressBar:
    """Simple progress bar for minimal output mode."""
    
    def __init__(self, max_episodes: int):
        self.max_episodes = max_episodes
        self.best_reward = 0.0
    
    def update(self, episode: int, reward: float, metrics: Dict[str, Any]):
        """Print simple progress update."""
        if reward > self.best_reward:
            self.best_reward = reward
            marker = " 🏆"
        else:
            marker = ""
        
        throughput = metrics.get("throughput", 0)
        latency = metrics.get("latency_p50", 0)
        
        print(f"[{episode}/{self.max_episodes}] reward={reward:.3f} "
              f"throughput={throughput:.1f} latency={latency:.1f}ms{marker}")
