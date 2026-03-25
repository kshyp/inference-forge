"""Thompson Sampling bandit implementation for vLLM config optimization."""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class GaussianPosterior:
    """Gaussian posterior distribution for a single config's reward."""
    mean: float = 0.5       # Prior mean (neutral)
    var: float = 1.0        # Prior variance (uninformative)
    count: int = 0          # Number of observations
    
    def sample(self) -> float:
        """Thompson sample: draw from posterior distribution."""
        if self.var <= 0:
            return self.mean
        return np.random.normal(self.mean, np.sqrt(self.var))
    
    def update(self, reward: float, noise_var: float = 1.0):
        """
        Bayesian update with observed reward.
        
        Using Gaussian conjugate prior with known noise variance.
        """
        # Precision-weighted update
        prior_precision = 1.0 / self.var if self.var > 0 else 0
        likelihood_precision = 1.0 / noise_var
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_var = 1.0 / posterior_precision
        
        posterior_mean = posterior_var * (
            self.mean * prior_precision + reward * likelihood_precision
        )
        
        self.mean = posterior_mean
        self.var = posterior_var
        self.count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": float(self.mean),
            "var": float(self.var),
            "count": self.count,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GaussianPosterior":
        return cls(
            mean=d.get("mean", 0.5),
            var=d.get("var", 1.0),
            count=d.get("count", 0),
        )


@dataclass
class ConfigStats:
    """Statistics for a single configuration."""
    config: Dict[str, Any]
    posterior: GaussianPosterior
    rewards: List[float] = field(default_factory=list)
    
    @property
    def config_id(self) -> str:
        return json.dumps(self.config, sort_keys=True, default=str)
    
    @property
    def avg_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return sum(self.rewards) / len(self.rewards)
    
    @property
    def best_reward(self) -> float:
        if not self.rewards:
            return 0.0
        return max(self.rewards)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "posterior": self.posterior.to_dict(),
            "rewards": self.rewards,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfigStats":
        return cls(
            config=d["config"],
            posterior=GaussianPosterior.from_dict(d["posterior"]),
            rewards=d.get("rewards", []),
        )


class ThompsonSamplingBandit:
    """
    Thompson Sampling bandit for discrete config space.
    
    Maintains a Gaussian posterior for each config.
    Naturally explores via posterior uncertainty (no epsilon-greedy needed).
    
    Algorithm:
    1. Sample reward estimate from each config's posterior
    2. Select config with highest sampled reward
    3. Run benchmark, observe actual reward
    4. Update posterior with observed reward
    5. Repeat
    
    Properties:
    - Exploration is automatic via posterior variance
    - Configs with high uncertainty get explored more
    - Converges to best config as variance decreases
    """
    
    def __init__(self, config_space: List[Dict[str, Any]], seed: Optional[int] = None):
        """
        Initialize bandit with config space.
        
        Args:
            config_space: List of valid configurations
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.config_space = config_space
        self.configs: Dict[str, ConfigStats] = {}
        
        # Initialize all configs with same prior
        for config in config_space:
            stats = ConfigStats(
                config=config,
                posterior=GaussianPosterior(mean=0.5, var=1.0),
            )
            self.configs[stats.config_id] = stats
        
        self._total_episodes = 0
    
    def select_config(self, explore: bool = True) -> Dict[str, Any]:
        """
        Select next configuration to try.
        
        Args:
            explore: If True, use Thompson sampling. If False, pick best posterior mean.
        
        Returns:
            Selected configuration
        """
        if explore:
            # Thompson sampling: sample from each posterior, pick max
            samples = []
            for config_id, stats in self.configs.items():
                sample = stats.posterior.sample()
                samples.append((sample, config_id, stats.config))
            
            samples.sort(reverse=True, key=lambda x: x[0])
            return samples[0][2]
        else:
            # Exploitation only: pick config with highest posterior mean
            return self.get_best_config()
    
    def select_config_ucb(self, exploration_factor: float = 2.0) -> Dict[str, Any]:
        """
        Select using Upper Confidence Bound (alternative to Thompson sampling).
        
        Args:
            exploration_factor: Controls exploration vs exploitation tradeoff
        
        Returns:
            Selected configuration
        """
        ucb_scores = []
        
        for config_id, stats in self.configs.items():
            mean = stats.posterior.mean
            # UCB = mean + exploration_factor * sqrt(variance)
            uncertainty = np.sqrt(stats.posterior.var)
            ucb = mean + exploration_factor * uncertainty
            ucb_scores.append((ucb, config_id, stats.config))
        
        ucb_scores.sort(reverse=True, key=lambda x: x[0])
        return ucb_scores[0][2]
    
    def update(self, config: Dict[str, Any], reward: float):
        """
        Update posterior after observing reward.
        
        Args:
            config: The configuration that was tested
            reward: Observed reward (should be in [0, 1])
        """
        config_id = json.dumps(config, sort_keys=True, default=str)
        
        if config_id not in self.configs:
            # This shouldn't happen if config_space is complete
            raise ValueError(f"Unknown config: {config}")
        
        stats = self.configs[config_id]
        stats.posterior.update(reward)
        stats.rewards.append(reward)
        self._total_episodes += 1
    
    def get_best_config(self) -> Dict[str, Any]:
        """
        Get config with highest posterior mean.
        
        Returns:
            Best configuration found so far
        """
        best_mean = -float('inf')
        best_config = None
        
        for stats in self.configs.values():
            if stats.posterior.mean > best_mean:
                best_mean = stats.posterior.mean
                best_config = stats.config
        
        return best_config or self.config_space[0]
    
    def get_top_configs(self, n: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Get top N configs by posterior mean.
        
        Returns:
            List of (config, posterior_mean) tuples
        """
        scored = [
            (stats.config, stats.posterior.mean)
            for stats in self.configs.values()
        ]
        scored.sort(reverse=True, key=lambda x: x[1])
        return scored[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        total_visits = sum(s.posterior.count for s in self.configs.values())
        configs_tried = sum(1 for s in self.configs.values() if s.posterior.count > 0)
        
        means = [s.posterior.mean for s in self.configs.values()]
        variances = [s.posterior.var for s in self.configs.values()]
        
        return {
            "total_episodes": total_visits,
            "config_space_size": len(self.config_space),
            "configs_tried": configs_tried,
            "coverage_pct": 100 * configs_tried / len(self.config_space),
            "best_posterior_mean": max(means),
            "worst_posterior_mean": min(means),
            "avg_posterior_mean": sum(means) / len(means),
            "avg_posterior_var": sum(variances) / len(variances),
            "exploration_remaining": sum(1 for v in variances if v > 0.5),
        }
    
    def get_config_stats(self, config: Dict[str, Any]) -> Optional[ConfigStats]:
        """Get stats for a specific config."""
        config_id = json.dumps(config, sort_keys=True, default=str)
        return self.configs.get(config_id)
    
    def save(self, path: Path):
        """Save bandit state to file."""
        data = {
            "config_space": self.config_space,
            "configs": {
                k: v.to_dict() for k, v in self.configs.items()
            },
            "total_episodes": self._total_episodes,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path) -> "ThompsonSamplingBandit":
        """Load bandit state from file."""
        with open(path) as f:
            data = json.load(f)
        
        bandit = cls(data["config_space"])
        bandit._total_episodes = data.get("total_episodes", 0)
        
        for config_id, stats_dict in data["configs"].items():
            bandit.configs[config_id] = ConfigStats.from_dict(stats_dict)
        
        return bandit
    
    def print_summary(self):
        """Print a summary of current bandit state."""
        stats = self.get_stats()
        
        print("="*60)
        print("Bandit State Summary")
        print("="*60)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Config Coverage: {stats['configs_tried']}/{stats['config_space_size']} "
              f"({stats['coverage_pct']:.1f}%)")
        print(f"\nPosterior Mean Stats:")
        print(f"  Best:  {stats['best_posterior_mean']:.4f}")
        print(f"  Worst: {stats['worst_posterior_mean']:.4f}")
        print(f"  Avg:   {stats['avg_posterior_mean']:.4f}")
        print(f"\nTop 5 Configs (by posterior mean):")
        
        for i, (config, mean) in enumerate(self.get_top_configs(5), 1):
            config_str = json.dumps(config, default=str)
            if len(config_str) > 50:
                config_str = config_str[:47] + "..."
            print(f"  {i}. {mean:.4f}: {config_str}")
        
        print("="*60)
