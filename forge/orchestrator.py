"""Inference Forge Orchestrator - The main state machine.

Runs the full agentic system in a continuous loop:
  Benchmark → Profile → Coordinate → Converge? → [loop or exit]

This is the "brain" that coordinates all agents and maintains the
optimization state across iterations.
"""

import asyncio
import json
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from forge.agents.benchmark.agent import BenchmarkAgent
from forge.agents.coordinator.agent import CoordinatorAgent
from forge.agents.coordinator.synthesis import SynthesisConfig
from forge.agents.profile.agent import ProfilerAgent
from forge.core.events import ExperimentPlan, RankedExperiment, Task
from forge.core.state import StateStore


@dataclass
class IterationResult:
    """Results from a single optimization iteration."""
    iteration: int
    experiment_id: str
    timestamp: str
    config: Dict[str, Any]
    baseline_rate: float
    metrics: Dict[str, Any]
    experiment_plan: Optional[ExperimentPlan] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "baseline_rate": self.baseline_rate,
            "metrics": self.metrics,
        }


@dataclass
class ConvergenceState:
    """Tracks convergence state across iterations."""
    history: List[IterationResult] = field(default_factory=list)
    converged: bool = False
    reason: Optional[str] = None
    best_iteration: Optional[int] = None
    best_config: Optional[Dict[str, Any]] = None
    
    def add_result(self, result: IterationResult) -> None:
        """Add a new iteration result."""
        self.history.append(result)
    
    def check_convergence(
        self,
        current_config: Dict[str, Any],
        max_iterations: int = 10,
        no_improvement_limit: int = 3,
        improvement_threshold: float = 0.02,
        backlog_size: int = 0
    ) -> tuple[bool, str]:
        """Check if optimization has converged.
        
        Args:
            current_config: The configuration to check
            max_iterations: Maximum number of iterations allowed
            no_improvement_limit: Stop after N iterations without improvement
            improvement_threshold: Minimum improvement to count as progress
            backlog_size: Number of experiments still in backlog (if > 0, don't converge)
        
        Returns:
            (converged, reason)
        """
        if len(self.history) >= max_iterations:
            return True, f"Max iterations ({max_iterations}) reached"
        
        if len(self.history) < 1:
            return False, ""
        
        # Check for config convergence (same config already in history)
        # NOTE: Skip this check if there are still experiments in the backlog.
        # We want to exhaust all ranked experiments before declaring convergence.
        current_config_str = json.dumps(current_config, sort_keys=True)
        for prev in self.history:
            if json.dumps(prev.config, sort_keys=True) == current_config_str:
                if backlog_size > 0:
                    # Config is in history, but we still have backlog experiments to try.
                    # Don't converge yet - continue with the backlog.
                    return False, ""
                self.converged = True
                self.reason = "Configuration converged (same config repeated)"
                return True, self.reason
        
        # Check for throughput improvement stagnation
        # NOTE: Also skip this check if we have backlog experiments remaining
        if backlog_size > 0:
            return False, ""
        
        if len(self.history) >= no_improvement_limit + 1:
            recent = self.history[-(no_improvement_limit + 1):]
            best_throughput = max(r.metrics.get("throughput", 0) for r in recent[:-1])
            current_throughput = recent[-1].metrics.get("throughput", 0)
            
            if current_throughput <= best_throughput * (1 + improvement_threshold):
                self.converged = True
                self.reason = f"No significant improvement for {no_improvement_limit} iterations"
                return True, self.reason
        
        return False, ""
    
    def find_best(self) -> tuple[int, IterationResult]:
        """Find the best iteration based on throughput and latency."""
        if not self.history:
            raise ValueError("No history to find best from")
        
        best_idx = 0
        best_score = 0.0
        
        for i, result in enumerate(self.history):
            throughput = result.metrics.get("throughput", 0)
            latency = result.metrics.get("latency_p99", 1)
            error_rate = result.metrics.get("error_rate", 0)
            
            # Score = throughput / (latency penalty) / (error penalty)
            latency_penalty = 1 + (latency / 1000)  # Normalize latency
            error_penalty = 1 + error_rate * 10
            score = throughput / latency_penalty / error_penalty
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx + 1, self.history[best_idx]


class InferenceEngine:
    """
    The main Inference Forge engine - a state machine that runs the optimization loop.
    
    State Machine:
        IDLE → BENCHMARK → PROFILE → COORDINATE → CHECK_CONVERGENCE
                                         ↑___________________|
    
    The engine runs until:
    1. Convergence is detected (no improvement, config repeats)
    2. Max iterations reached
    3. User interrupts (Ctrl+C)
    4. Error occurs
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        port: int = 8000,
        data_dir: str = "./data",
        max_iterations: int = 10,
        no_improvement_limit: int = 3,
        force_iterations: bool = False,
    ):
        self.model_name = model_name
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Convergence settings
        self.max_iterations = max_iterations
        self.no_improvement_limit = no_improvement_limit
        self.force_iterations = force_iterations  # Run all iterations regardless of convergence
        
        # Experiment backlog - work through ranked experiments before re-consulting SMEs
        self.experiment_backlog: List[RankedExperiment] = []
        self.current_plan_iteration: int = 0  # Which SME consultation this is
        
        # Track configs we've already tried to avoid duplicates
        self.tried_configs: set[str] = set()
        
        # State
        self.convergence = ConvergenceState()
        self.current_config: Dict[str, Any] = {}
        self.iteration: int = 0
        self.running: bool = False
        self.shutdown_requested: bool = False
        
        # Baseline rate for all benchmarks (128 RPS)
        self.baseline_rate: float = 128.0
        
        # Initialize state store
        self.state_store = StateStore(str(self.data_dir / "state.db"))
        
        # Initialize agents with sequential health ports
        # Base port is configurable, agents get base, base+1, base+2
        base_health_port = 38080  # High port range to avoid conflicts
        
        self.benchmark_agent = BenchmarkAgent(
            state_store=self.state_store,
            health_port=base_health_port,
            data_dir=str(data_dir)
        )
        
        self.profiler_agent = ProfilerAgent(
            state_store=self.state_store,
            health_port=base_health_port + 1,
            data_dir=str(data_dir)
        )
        
        # Create synthesis config to include ALL experiments (no limit)
        synthesis_config = SynthesisConfig(
            max_experiments=None,  # Include ALL experiments from SME suggestions
            ranking_mode="consensus_first",  # Rank by consensus first, then confidence
            merge_similar=True,
            merge_threshold=0.8,
        )
        
        self.coordinator_agent = CoordinatorAgent(
            state_store=self.state_store,
            health_port=base_health_port + 2,
            data_dir=str(data_dir),
            synthesis_config=synthesis_config
        )
        
        # Store health URLs for external monitoring
        self.health_urls = {
            "benchmark": self.benchmark_agent.health_url,
            "profile": self.profiler_agent.health_url,
            "coordinator": self.coordinator_agent.health_url,
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Try to restore state from previous run
        self._restore_state()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n\n⚠️  Shutdown requested. Finishing current iteration...")
        self.shutdown_requested = True
    
    async def run(
        self,
        initial_config: Optional[Dict[str, Any]] = None
    ) -> ConvergenceState:
        """
        Run the optimization loop until convergence or shutdown.
        
        Args:
            initial_config: Starting configuration. Uses defaults if None.
            
        Returns:
            Final convergence state with history
        """
        self.current_config = initial_config or {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048,
            "enable_chunked_prefill": False,
        }
        
        print("="*80)
        print("  🔥 INFERENCE FORGE - CONTINUOUS OPTIMIZATION ENGINE 🔥")
        print("="*80)
        print(f"\n📝 Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Port: {self.port}")
        print(f"   Data Dir: {self.data_dir}")
        print(f"   Max Iterations: {self.max_iterations}")
        print(f"   No Improvement Limit: {self.no_improvement_limit}")
        if self.force_iterations:
            print(f"   ⚡ Force Iterations: ENABLED (will run all {self.max_iterations} iterations)")
        print(f"   📋 Backlog Mode: Will try all ranked experiments before re-consulting SMEs")
        print(f"\n🚀 Starting optimization loop...")
        print(f"   Initial config: {json.dumps(self.current_config, indent=2)}")
        print(f"\n📡 Health Endpoints:")
        for name, url in self.health_urls.items():
            print(f"   {name}: {url}")
        
        self.running = True
        self.iteration = 0
        
        # Start health servers for all agents
        print(f"\n🟢 Starting health servers...")
        await self.benchmark_agent.start_health()
        await self.profiler_agent.start_health()
        await self.coordinator_agent.start_health()
        print(f"   ✓ All health servers running")
        
        try:
            while self.running and not self.shutdown_requested:
                self.iteration += 1
                
                # Check convergence BEFORE running new iteration
                # BUT: Don't converge if there are still experiments in the backlog
                # (we want to exhaust all suggestions before declaring convergence)
                converged, reason = self.convergence.check_convergence(
                    self.current_config,
                    self.max_iterations,
                    self.no_improvement_limit,
                    backlog_size=len(self.experiment_backlog)
                )
                
                if converged:
                    print(f"\n{'='*80}")
                    print(f"🎯 CONVERGENCE ACHIEVED: {reason}")
                    print(f"{'='*80}")
                    self.convergence.converged = True
                    self.convergence.reason = reason
                    break
                
                # Check if max iterations reached
                if self.iteration > self.max_iterations:
                    print(f"\n{'='*80}")
                    print(f"🛑 MAX ITERATIONS ({self.max_iterations}) REACHED")
                    print(f"{'='*80}")
                    self.convergence.converged = True
                    self.convergence.reason = f"Max iterations ({self.max_iterations}) reached"
                    break
                
                # Get next experiment from backlog or consult SMEs
                next_experiment = self._get_next_experiment()
                
                if next_experiment is None:
                    # No more experiments to try - need to consult SMEs
                    print(f"\n📋 Backlog empty - consulting SMEs for new suggestions...")
                    result = await self._run_iteration()
                    
                    if result and result.experiment_plan:
                        # Populate backlog with ALL ranked experiments from the plan
                        added_count = self._populate_backlog(result.experiment_plan)
                        # Save result to history (but don't add to convergence yet)
                        self._save_state()
                        
                        if added_count == 0:
                            print(f"\n⚠️  No new unique experiments in plan")
                        
                        # Try to get next experiment from newly populated backlog
                        next_experiment = self._get_next_experiment()
                    
                    if next_experiment is None:
                        # Still no experiments after consulting SMEs
                        if self.force_iterations:
                            print(f"\n⚡ No new unique experiments, but continuing...")
                            continue
                        else:
                            print(f"\n⚠️  No new unique experiments available - stopping")
                            self.convergence.converged = True
                            self.convergence.reason = "Exhausted all unique experiment suggestions"
                            break
                
                # Apply the next experiment
                new_config = self._apply_experiment(self.current_config, next_experiment)
                config_key = json.dumps(new_config, sort_keys=True)
                
                # Check if we've already tried this config
                if config_key in self.tried_configs:
                    print(f"\n🔄 Skipping already-tried config, fetching next...")
                    continue
                
                self.current_config = new_config
                self.tried_configs.add(config_key)
                
                print(f"\n🔄 Trying experiment #{next_experiment.priority} from backlog")
                print(f"   Config: {json.dumps(self.current_config, indent=2)}")
                print(f"   Confidence: {next_experiment.confidence:.2f}")
                print(f"   Sources: {', '.join(next_experiment.source_experts)}")
                print(f"   Backlog remaining: {len(self.experiment_backlog)}")
                
                # Run benchmark with this config
                result = await self._run_benchmark_only(self.current_config)
                
                if result:
                    self.convergence.add_result(result)
                    self._print_comparison_table()
                    self._save_state()
                else:
                    print("\n❌ Benchmark failed, trying next experiment...")
                    continue
                
                # Brief pause between iterations
                if not self.shutdown_requested:
                    print("\n⏳ Pausing 3 seconds before next iteration...")
                    await asyncio.sleep(3)
        
        except Exception as e:
            print(f"\n❌ Error during optimization: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.running = False
            await self._shutdown()
        
        return self.convergence
    
    async def _run_iteration(self) -> Optional[IterationResult]:
        """Run a single optimization iteration."""
        experiment_id = str(uuid4())  # Full UUID for proper handling
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*80}")
        print(f"🔁 ITERATION {self.iteration}")
        print(f"{'='*80}")
        print(f"🆔 Experiment: {experiment_id}")
        print(f"⚙️  Config: {json.dumps(self.current_config, indent=2)}")
        
        try:
            # Phase 1: BENCHMARK
            if self.baseline_rate is None:
                # Use default baseline rate
                self.baseline_rate = 128.0
            
            print(f"\n📊 PHASE 1: Benchmark at {self.baseline_rate:.0f} RPS...")
            
            benchmark_task = Task(
                type="benchmark",
                payload={
                    "experiment_id": experiment_id,
                    "model_name": self.model_name,
                    "vllm_config": {
                        "port": self.port,
                        "startup_timeout_seconds": 300,
                        **self.current_config
                    },
                    "guidellm_config": {
                        "dataset": "wikitext",
                        "prompt_tokens": 128,
                        "output_tokens": 64,
                        "max_requests": 500,
                    }
                }
            )
            
            benchmark_result = await self.benchmark_agent.run_single_task(benchmark_task)
            
            if not benchmark_result.get("_success"):
                print(f"❌ Benchmark failed: {benchmark_result.get('_error')}")
                return None
            
            baseline_rate = self.baseline_rate
            
            # Extract full metrics from benchmark result
            benchmark_res = benchmark_result.get("benchmark_result", {})
            metrics_data = benchmark_res.get("metrics", {})
            
            def get_percentile(data, metric, percentile):
                """Safely extract percentile from metric data."""
                val = data.get(metric, {}).get(percentile, 0)
                return val * 1000 if val and val < 10 else val
            
            baseline_metrics = {
                "throughput": metrics_data.get("requests_per_sec", 0),
                "total_tokens_per_sec": metrics_data.get("total_tokens_per_sec", 0),
                "output_tokens_per_sec": metrics_data.get("output_tokens_per_sec", 0),
                "ttft_p50": get_percentile(metrics_data, "ttft_ms", "p50"),
                "ttft_p95": get_percentile(metrics_data, "ttft_ms", "p95"),
                "ttft_p99": get_percentile(metrics_data, "ttft_ms", "p99"),
                "tpot_p50": get_percentile(metrics_data, "tpot_ms", "p50"),
                "tpot_p95": get_percentile(metrics_data, "tpot_ms", "p95"),
                "tpot_p99": get_percentile(metrics_data, "tpot_ms", "p99"),
                "latency_p50": get_percentile(metrics_data, "itl_ms", "p50"),
                "latency_p95": get_percentile(metrics_data, "itl_ms", "p95"),
                "latency_p99": get_percentile(metrics_data, "itl_ms", "p99"),
                "error_rate": 0,
            }
            
            if self.iteration == 1:
                # First iteration: Display baseline with config
                config_items = [(k, v) for k, v in self.current_config.items() 
                              if k not in ["port", "startup_timeout_seconds"]]
                config_lines = self._wrap_config([f"{k}={v}" for k, v in config_items], max_line_len=50)
                
                print(f"\n   ╔══════════════════════════════════════════════════════════════════════════╗")
                print(f"   ║  📊 BASELINE ESTABLISHED (Iteration 1)                                   ║")
                print(f"   ╠══════════════════════════════════════════════════════════════════════════╣")
                print(f"   ║  Config: {config_lines[0]:<59}║")
                for config_line in config_lines[1:]:
                    print(f"   ║          {config_line:<59}║")
                print(f"   ╠══════════════════════════════════════════════════════════════════════════╣")
                print(f"   ║  Baseline Rate: {self.baseline_rate:>6.0f} RPS                                                ║")
                print(f"   ║  Throughput:      {baseline_metrics['throughput']:>6.1f} req/s                                          ║")
                print(f"   ║  Total Tok/s:     {baseline_metrics['total_tokens_per_sec']:>8.1f}                                        ║")
                print(f"   ║  Output Tok/s:    {baseline_metrics['output_tokens_per_sec']:>8.1f}                                        ║")
                print(f"   ║  TTFT p50/p95/p99: {baseline_metrics['ttft_p50']:>5.0f}/{baseline_metrics['ttft_p95']:>5.0f}/{baseline_metrics['ttft_p99']:>5.0f} ms                                    ║")
                print(f"   ║  TPOT p50/p95/p99: {baseline_metrics['tpot_p50']:>5.0f}/{baseline_metrics['tpot_p95']:>5.0f}/{baseline_metrics['tpot_p99']:>5.0f} ms                                    ║")
                print(f"   ╚══════════════════════════════════════════════════════════════════════════╝")
                print(f"   → All subsequent iterations will be compared against these metrics\n")
            else:
                # Subsequent iterations: Show comparison
                print(f"   ✓ Running at baseline rate: {self.baseline_rate:.0f} RPS")
                print(f"   📊 Current: {baseline_metrics['throughput']:.1f} req/s, "
                      f"TTFT={baseline_metrics['ttft_p50']:.0f}/{baseline_metrics['ttft_p95']:.0f}/{baseline_metrics['ttft_p99']:.0f}ms, "
                      f"TPOT={baseline_metrics['tpot_p50']:.0f}/{baseline_metrics['tpot_p95']:.0f}/{baseline_metrics['tpot_p99']:.0f}ms")
            
            # Phase 2: PROFILE - Run profilers at saturation (optional, uses fallback on failure)
            print(f"\n🔍 PHASE 2: Profile (steady-state at saturation)...")
            
            # Register SMEs first to get data requirements
            register_task = Task(
                type="coordinator",
                payload={"action": "register_smes"}
            )
            register_result = await self.coordinator_agent.run_single_task(register_task)
            
            if register_result.get("_success"):
                data_requirements = register_result.get("profiler_requirements", {})
            else:
                data_requirements = {"vllm_logs": {"required": True}}
            
            # Try profiler, but use baseline metrics if it fails
            collected_data = {}
            benchmark_metrics = baseline_metrics  # Use sweep metrics as default
            
            # Phase 2: PROFILE - Collect debug logs and nsys profile
            print(f"\n🔍 PHASE 2: Profile (debug logs + nsys)...")
            
            profile_task = Task(
                type="profile",
                payload={
                    "experiment_id": experiment_id,
                    "model_name": self.model_name,
                    "vllm_config": {"port": self.port, **self.current_config},
                    "guidellm_config": {
                        "dataset": "wikitext",
                        "prompt_tokens": 128,
                        "output_tokens": 64,
                        "max_requests": 200,
                    },
                    "enable_nsys": True,
                }
            )
            
            try:
                profile_result = await self.profiler_agent.run_single_task(profile_task)
                
                if profile_result.get("_success"):
                    print(f"   ✓ Profile complete")
                    print(f"   Log: {profile_result.get('log_file', 'N/A')}")
                    
                    # Build collected_data for SME analysis
                    collected_data = {
                        "vllm_logs": {"log_file": profile_result.get("log_file")},
                    }
                    
                    # Analyze NSYS report if available
                    nsys_report_path = profile_result.get('nsys_report')
                    if nsys_report_path:
                        print(f"   NSys report: {nsys_report_path}")
                        try:
                            from forge.llm.nsys_analyzer import NSYSAnalyzer
                            nsys_analyzer = NSYSAnalyzer()
                            print(f"   Analyzing NSYS report...")
                            nsys_analysis = await nsys_analyzer.analyze(
                                nsys_report_path, 
                                enable_llm_analysis=True
                            )
                            collected_data["nsys_report"] = nsys_analysis.to_dict()
                            print(f"   ✓ NSYS analysis complete")
                            print(f"     GPU util: {nsys_analysis.metrics.gpu_utilization_percent:.1f}%" 
                                  if nsys_analysis.metrics.gpu_utilization_percent else "     GPU util: N/A")
                            if nsys_analysis.bottlenecks:
                                print(f"     Bottlenecks: {len(nsys_analysis.bottlenecks)}")
                        except Exception as e:
                            print(f"   ⚠️  NSYS analysis failed: {e}")
                            # Fallback to just the path
                            collected_data["nsys_report"] = {"report_path": nsys_report_path}
                    
                    metrics = baseline_metrics  # Use benchmark metrics
                else:
                    print(f"   ⚠️  Profile failed: {profile_result.get('_error')}")
                    collected_data = {}
                    metrics = baseline_metrics
            except Exception as e:
                print(f"   ⚠️  Profile exception: {e}")
                import traceback
                traceback.print_exc()
                collected_data = {}
                metrics = baseline_metrics
            
            print(f"\n📈 Final Metrics:")
            print(f"   Throughput: {metrics['throughput']:.1f} req/s")
            print(f"   Total Tok/s: {metrics['total_tokens_per_sec']:.1f}")
            print(f"   Output Tok/s: {metrics['output_tokens_per_sec']:.1f}")
            print(f"   TTFT p50/p95/p99: {metrics['ttft_p50']:.0f}/{metrics['ttft_p95']:.0f}/{metrics['ttft_p99']:.0f} ms")
            print(f"   TPOT p50/p95/p99: {metrics['tpot_p50']:.0f}/{metrics['tpot_p95']:.0f}/{metrics['tpot_p99']:.0f} ms")
            
            # Phase 3: COORDINATE - Consult SMEs and generate experiment plan
            print(f"\n🧠 PHASE 3: Coordinate (SME analysis)...")
            
            analyze_task = Task(
                type="coordinator",
                payload={
                    "action": "analyze",
                    "profiling_result": {
                        "collected_data": collected_data,
                        "benchmark_metrics": metrics,
                    },
                    "current_config": self.current_config,
                    "iteration": self.iteration,
                    "parent_experiment_id": experiment_id,
                }
            )
            
            coord_result = await self.coordinator_agent.run_single_task(analyze_task)
            
            experiment_plan = None
            if coord_result.get("_success"):
                # Get the experiment plan from the coordinator's result queue
                experiment_plan = await self.coordinator_agent.get_experiment_plan()
            else:
                print(f"⚠️  Coordinator analysis failed: {coord_result.get('_error')}")
            
            return IterationResult(
                iteration=self.iteration,
                experiment_id=experiment_id,
                timestamp=timestamp,
                config=self.current_config.copy(),
                baseline_rate=baseline_rate,
                metrics=metrics,
                experiment_plan=experiment_plan
            )
        
        except Exception as e:
            print(f"\n❌ Error in iteration {self.iteration}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_metrics(
        self, 
        collected_data: Dict[str, Any], 
        benchmark_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract full metrics (p50/p95/p99) from profiling data."""
        # Try to get from benchmark_metrics first
        if benchmark_metrics:
            return {
                "throughput": benchmark_metrics.get("throughput", 0),
                "total_tokens_per_sec": benchmark_metrics.get("total_tokens_per_sec", 0),
                "output_tokens_per_sec": benchmark_metrics.get("output_tokens_per_sec", 0),
                "ttft_p50": benchmark_metrics.get("ttft_p50", 0),
                "ttft_p95": benchmark_metrics.get("ttft_p95", 0),
                "ttft_p99": benchmark_metrics.get("ttft_p99", 0),
                "tpot_p50": benchmark_metrics.get("tpot_p50", 0),
                "tpot_p95": benchmark_metrics.get("tpot_p95", 0),
                "tpot_p99": benchmark_metrics.get("tpot_p99", 0),
                "latency_p50": benchmark_metrics.get("latency_p50", 0),
                "latency_p95": benchmark_metrics.get("latency_p95", 0),
                "latency_p99": benchmark_metrics.get("latency_p99", 0),
                "error_rate": benchmark_metrics.get("error_rate", 0),
            }
        
        # Otherwise extract from collected data
        vllm_logs = collected_data.get("vllm_logs", {})
        return {
            "throughput": vllm_logs.get("throughput", 0),
            "total_tokens_per_sec": vllm_logs.get("total_tokens_per_sec", 0),
            "output_tokens_per_sec": vllm_logs.get("output_tokens_per_sec", 0),
            "ttft_p50": vllm_logs.get("ttft_p50", 0),
            "ttft_p95": vllm_logs.get("ttft_p95", 0),
            "ttft_p99": vllm_logs.get("ttft_p99", 0),
            "tpot_p50": vllm_logs.get("tpot_p50", 0),
            "tpot_p95": vllm_logs.get("tpot_p95", 0),
            "tpot_p99": vllm_logs.get("tpot_p99", 0),
            "latency_p50": vllm_logs.get("latency_p50", 0),
            "latency_p95": vllm_logs.get("latency_p95", 0),
            "latency_p99": vllm_logs.get("latency_p99", 0),
            "error_rate": vllm_logs.get("error_rate", 0),
        }
    
    def _print_comparison_table(self) -> None:
        """Print a comparison table of all iterations with multi-line config display."""
        if not self.convergence.history:
            return
        
        # Get baseline for comparison
        baseline = self.convergence.history[0].metrics if self.convergence.history else None
        baseline_throughput = baseline.get("throughput", 0) if baseline else 0
        
        # Fixed column widths for metrics (aligned)
        metrics_start_col = 42  # Column where metrics start
        
        print("\n" + "="*130)
        print("📊 PERFORMANCE COMPARISON TABLE")
        print("="*130)
        
        # Header - two lines: config header on first line, metrics headers aligned below
        print(f"{'Iter':>4} │ {'Configuration (full)':<35} │ "
              f"{'Req/s':>6} │ {'TotTok/s':>8} │ {'OutTok/s':>8} │ "
              f"{'TTFT p50/p95/p99':>20} │ {'TPOT p50/p95/p99':>20} │ {'Δ vs Base':>10}")
        print("-"*130)
        
        # Rows with multi-line config
        for result in self.convergence.history:
            m = result.metrics
            config = result.config
            
            # Build config lines (wrap at ~35 chars)
            config_items = [(k, v) for k, v in config.items() 
                          if k not in ["port", "startup_timeout_seconds"]]
            if config_items:
                config_pairs = [f"{k}={v}" for k, v in config_items]
                config_lines = self._wrap_config(config_pairs, max_line_len=35)
            else:
                config_lines = ["baseline"]
            
            # Format metrics
            throughput = m.get("throughput", 0)
            total_tok = m.get("total_tokens_per_sec", 0)
            out_tok = m.get("output_tokens_per_sec", 0)
            
            ttft_p50 = m.get("ttft_p50", 0)
            ttft_p95 = m.get("ttft_p95", 0)  
            ttft_p99 = m.get("ttft_p99", 0)
            
            tpot_p50 = m.get("tpot_p50", 0)
            tpot_p95 = m.get("tpot_p95", 0)
            tpot_p99 = m.get("tpot_p99", 0)
            
            # Delta
            if baseline_throughput > 0 and result.iteration > 1:
                delta_pct = ((throughput - baseline_throughput) / baseline_throughput) * 100
                delta_str = f"{delta_pct:+.1f}%"
                if delta_pct > 5:
                    delta_str = f"🟢 {delta_str}"
                elif delta_pct < -5:
                    delta_str = f"🔴 {delta_str}"
                else:
                    delta_str = f"⚪ {delta_str}"
            elif result.iteration == 1:
                delta_str = "📊 BASE"
            else:
                delta_str = "N/A"
            
            # Print first line with iteration number and metrics
            first_config = config_lines[0] if config_lines else ""
            print(f"{result.iteration:>4} │ {first_config:<35} │ "
                  f"{throughput:>6.1f} │ {total_tok:>8.1f} │ {out_tok:>8.1f} │ "
                  f"{ttft_p50:>4.0f}/{ttft_p95:>4.0f}/{ttft_p99:>5.0f} ms │ "
                  f"{tpot_p50:>4.0f}/{tpot_p95:>4.0f}/{tpot_p99:>5.0f} ms │ {delta_str:>10}")
            
            # Print additional config lines (continuation) without metrics
            for config_line in config_lines[1:]:
                print(f"     │ {config_line:<35} │ "
                      f"{'':>6} │ {'':>8} │ {'':>8} │ "
                      f"{'':>20} │ {'':>20} │ {'':>10}")
            
            # Add spacing between iterations for readability
            if result != self.convergence.history[-1]:
                print(f"     │{'':35} │ " + "-"*83)
        
        print("="*130)
        print("Legend: 🟢 = Improvement >5%, 🔴 = Regression >5%, ⚪ = Within 5%, 📊 = Baseline")
        print(f"Baseline: {baseline_throughput:.1f} req/s at {self.baseline_rate:.0f} RPS")
        print("="*130 + "\n")
    
    def _wrap_config(self, config_pairs: List[str], max_line_len: int = 35) -> List[str]:
        """Wrap config pairs into lines of max length."""
        if not config_pairs:
            return ["baseline"]
        
        lines = []
        current_line = ""
        
        for pair in config_pairs:
            # If adding this pair would exceed limit, start new line
            if current_line and len(current_line) + 2 + len(pair) > max_line_len:
                lines.append(current_line)
                current_line = pair
            else:
                if current_line:
                    current_line += ", " + pair
                else:
                    current_line = pair
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else ["baseline"]
    
    def _extract_metrics_from_sweep(self, sweep_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract metrics from sweep result for fallback when profiler fails."""
        # Try to get metrics from the sweep file
        sweep_file = sweep_result.get("guidellm_report_path", "")
        
        try:
            import json
            from pathlib import Path
            
            if sweep_file and Path(sweep_file).exists():
                with open(sweep_file) as f:
                    sweep_data = json.load(f)
                
                # Find the benchmark with highest rate that didn't error
                if "benchmarks" in sweep_data and sweep_data["benchmarks"]:
                    best_benchmark = None
                    best_throughput = 0
                    
                    for b in sweep_data["benchmarks"]:
                        config = b.get("config", {})
                        strategy = config.get("strategy", {})
                        
                        # Only consider constant rate benchmarks
                        if strategy.get("type_") != "constant":
                            continue
                        
                        metrics = b.get("metrics", {})
                        rps_data = metrics.get("requests_per_second", {}).get("successful", {})
                        throughput = rps_data.get("mean", 0) if isinstance(rps_data, dict) else 0
                        
                        # Check for errors
                        totals = metrics.get("request_totals", {})
                        total = totals.get("total", 0)
                        errored = totals.get("errored", 0)
                        error_rate = errored / total if total > 0 else 0
                        
                        if throughput > best_throughput and error_rate < 0.05:
                            best_throughput = throughput
                            best_benchmark = b
                    
                    if best_benchmark:
                        metrics = best_benchmark.get("metrics", {})
                        
                        # Extract latency
                        latency_data = metrics.get("request_latency", {}).get("successful", {})
                        latency_p99 = latency_data.get("percentiles", {}).get("p99", 0)
                        if latency_p99 == 0:
                            latency_p99 = latency_data.get("mean", 0)
                        latency_p99 = latency_p99 * 1000 if latency_p99 < 10 else latency_p99
                        
                        # Extract TTFT
                        ttft_data = metrics.get("time_to_first_token_ms", {}).get("successful", {})
                        ttft_p99 = ttft_data.get("percentiles", {}).get("p99", 0)
                        if ttft_p99 == 0:
                            ttft_p99 = ttft_data.get("mean", 0)
                        ttft_p99 = ttft_p99 * 1000 if ttft_p99 < 10 else ttft_p99
                        
                        # Extract TPOT
                        tpot_data = metrics.get("time_per_output_token_ms", {}).get("successful", {})
                        tpot_p99 = tpot_data.get("percentiles", {}).get("p99", 0)
                        if tpot_p99 == 0:
                            tpot_p99 = tpot_data.get("mean", 0)
                        tpot_p99 = tpot_p99 * 1000 if tpot_p99 < 10 else tpot_p99
                        
                        return {
                            "throughput": best_throughput,
                            "ttft_p99": ttft_p99,
                            "tpot_p99": tpot_p99,
                            "latency_p99": latency_p99,
                            "error_rate": 0,
                        }
        except Exception as e:
            print(f"   Warning: Could not extract metrics from sweep: {e}")
        
        # Return zeros if extraction fails
        return {"throughput": 0, "ttft_p99": 0, "tpot_p99": 0, "latency_p99": 0, "error_rate": 0}
    
    def _get_next_experiment(self) -> Optional[RankedExperiment]:
        """Get the next experiment from the backlog, skipping already-tried configs."""
        while self.experiment_backlog:
            exp = self.experiment_backlog.pop(0)
            test_config = self._apply_experiment(self.current_config, exp)
            config_key = json.dumps(test_config, sort_keys=True)
            
            if config_key not in self.tried_configs:
                return exp
            else:
                print(f"   (Skipping already-tried experiment #{exp.priority})")
        
        return None
    
    def _populate_backlog(self, plan: ExperimentPlan) -> int:
        """Populate the experiment backlog from a new plan.
        
        Only adds experiments with unique configs that haven't been tried
        and aren't already in the backlog.
        Returns the number of new experiments added.
        """
        # Build set of configs already in backlog
        backlog_configs = set()
        for backlog_exp in self.experiment_backlog:
            test_config = self._apply_experiment(self.current_config, backlog_exp)
            config_key = json.dumps(test_config, sort_keys=True)
            backlog_configs.add(config_key)
        
        new_experiments = []
        skipped_tried = 0
        skipped_duplicate = 0
        
        for exp in plan.experiments:
            # Check if this experiment's config is unique
            test_config = self._apply_experiment(self.current_config, exp)
            config_key = json.dumps(test_config, sort_keys=True)
            
            if config_key in self.tried_configs:
                skipped_tried += 1
                continue  # Skip already-tried configs
            
            if config_key in backlog_configs:
                skipped_duplicate += 1
                continue  # Skip duplicates already in backlog
            
            new_experiments.append(exp)
            backlog_configs.add(config_key)  # Track to avoid duplicates within this batch
        
        # Add new experiments to the backlog
        self.experiment_backlog.extend(new_experiments)
        self.current_plan_iteration = plan.iteration
        
        if new_experiments:
            print(f"\n📋 Added {len(new_experiments)} new experiments to backlog")
            if skipped_tried > 0:
                print(f"   (Skipped {skipped_tried} already-tried)")
            if skipped_duplicate > 0:
                print(f"   (Skipped {skipped_duplicate} duplicates)")
            print(f"   Backlog size: {len(self.experiment_backlog)} experiments")
            print(f"   SME consultation #{self.current_plan_iteration}")
        
        return len(new_experiments)
    
    async def _run_benchmark_only(
        self, 
        config: Dict[str, Any]
    ) -> Optional[IterationResult]:
        """Run only the benchmark phase with a given config (no profiling/SME consultation)."""
        experiment_id = str(uuid4())
        timestamp = datetime.now().isoformat()
        
        print(f"\n{'='*80}")
        print(f"🔁 ITERATION {self.iteration} (Backlog execution)")
        print(f"{'='*80}")
        print(f"🆔 Experiment: {experiment_id}")
        print(f"⚙️  Config: {json.dumps(config, indent=2)}")
        
        try:
            # Determine target rate
            if self.baseline_rate is None:
                # First time - need to find saturation
                print(f"\n📊 Finding saturation rate...")
                
                sweep_task = Task(
                    type="benchmark",
                    payload={
                        "experiment_id": experiment_id,
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
                            "max_requests": 500,
                        }
                    }
                )
                
                sweep_result = await self.benchmark_agent.run_single_task(sweep_task)
                
                if not sweep_result.get("_success"):
                    print(f"❌ Sweep failed: {sweep_result.get('_error')}")
                    return None
                
                sweep_data = sweep_result.get("sweep_result", {})
                saturation_point = sweep_data.get("saturation_point", {})
                baseline_rate = sweep_result.get("baseline_rate", 128.0)
                self.baseline_rate = baseline_rate
                print(f"   ✓ Baseline established: {baseline_rate:.0f} RPS")
                
                # Run at saturation for full metrics
                print(f"\n📊 Running benchmark at saturation...")
                benchmark_task = Task(
                    type="benchmark",
                    payload={
                        "experiment_id": experiment_id,
                        "model_name": self.model_name,
                        # No target_rate needed - benchmark agent uses fixed rate
                        "vllm_config": {
                            "port": self.port,
                            "startup_timeout_seconds": 300,
                            **config
                        },
                        "guidellm_config": {
                            "dataset": "wikitext",
                            "prompt_tokens": 128,
                            "output_tokens": 64,
                            "max_requests": 500,
                        }
                    }
                )
                benchmark_result = await self.benchmark_agent.run_single_task(benchmark_task)
                
                if not benchmark_result.get("_success"):
                    print(f"❌ Benchmark failed: {benchmark_result.get('_error')}")
                    return None
                    
            else:
                # Use baseline rate
                baseline_rate = self.baseline_rate
                print(f"\n📊 Benchmark at baseline {baseline_rate:.0f} RPS...")
                
                benchmark_task = Task(
                    type="benchmark",
                    payload={
                        "experiment_id": experiment_id,
                        "model_name": self.model_name,
                        # No target_rate needed - benchmark agent uses fixed rate
                        "vllm_config": {
                            "port": self.port,
                            "startup_timeout_seconds": 300,
                            **config
                        },
                        "guidellm_config": {
                            "dataset": "wikitext",
                            "prompt_tokens": 128,
                            "output_tokens": 64,
                            "max_requests": 500,
                        }
                    }
                )
                benchmark_result = await self.benchmark_agent.run_single_task(benchmark_task)
                
                if not benchmark_result.get("_success"):
                    print(f"❌ Benchmark failed: {benchmark_result.get('_error')}")
                    return None
            
            # Extract metrics
            benchmark_res = benchmark_result.get("benchmark_result", {})
            metrics_data = benchmark_res.get("metrics", {})
            
            def get_percentile(data, metric, percentile):
                val = data.get(metric, {}).get(percentile, 0)
                return val * 1000 if val and val < 10 else val
            
            metrics = {
                "throughput": metrics_data.get("requests_per_sec", 0),
                "total_tokens_per_sec": metrics_data.get("total_tokens_per_sec", 0),
                "output_tokens_per_sec": metrics_data.get("output_tokens_per_sec", 0),
                "ttft_p50": get_percentile(metrics_data, "ttft_ms", "p50"),
                "ttft_p95": get_percentile(metrics_data, "ttft_ms", "p95"),
                "ttft_p99": get_percentile(metrics_data, "ttft_ms", "p99"),
                "tpot_p50": get_percentile(metrics_data, "tpot_ms", "p50"),
                "tpot_p95": get_percentile(metrics_data, "tpot_ms", "p95"),
                "tpot_p99": get_percentile(metrics_data, "tpot_ms", "p99"),
                "latency_p50": get_percentile(metrics_data, "itl_ms", "p50"),
                "latency_p95": get_percentile(metrics_data, "itl_ms", "p95"),
                "latency_p99": get_percentile(metrics_data, "itl_ms", "p99"),
                "error_rate": 0,
            }
            
            print(f"\n📊 Results:")
            print(f"   Throughput: {metrics['throughput']:.1f} req/s")
            print(f"   Total Tok/s: {metrics['total_tokens_per_sec']:.1f}")
            print(f"   TTFT p50/p95/p99: {metrics['ttft_p50']:.0f}/{metrics['ttft_p95']:.0f}/{metrics['ttft_p99']:.0f} ms")
            
            return IterationResult(
                iteration=self.iteration,
                experiment_id=experiment_id,
                timestamp=timestamp,
                config=config.copy(),
                baseline_rate=baseline_rate,
                metrics=metrics,
                experiment_plan=None  # No plan from backlog execution
            )
            
        except Exception as e:
            print(f"\n❌ Error in benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_experiment(
        self, 
        base_config: Dict[str, Any], 
        experiment: RankedExperiment
    ) -> Dict[str, Any]:
        """Apply an experiment's config patch to base config."""
        new_config = base_config.copy()
        new_config.update(experiment.config_patch)
        
        # Handle removals
        for key in experiment.config_removals:
            new_config.pop(key, None)
        
        return new_config
    
    def _restore_state(self) -> None:
        """Restore optimization state from disk if exists."""
        state_file = self.data_dir / "optimization_state.json"
        
        if not state_file.exists():
            return  # No previous state
        
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            # Restore tried configs
            if "tried_configs" in state:
                self.tried_configs = set(state["tried_configs"])
                print(f"   📝 Restored {len(self.tried_configs)} tried configs")
            
            # Restore experiment backlog
            if "experiment_backlog" in state:
                from forge.core.events import RankedExperiment
                self.experiment_backlog = [
                    RankedExperiment(
                        priority=exp.get("priority", i + 1),
                        config_patch=exp.get("config_patch", {}),
                        config_removals=exp.get("config_removals", []),
                        hypothesis=exp.get("hypothesis", ""),
                        expected_improvement=exp.get("expected_improvement", ""),
                        success_criteria=exp.get("success_criteria", ""),
                        source_experts=exp.get("source_experts", []),
                        confidence=exp.get("confidence", 0.0),
                        abort_conditions=exp.get("abort_conditions", []),
                    )
                    for i, exp in enumerate(state["experiment_backlog"])
                ]
                print(f"   📋 Restored {len(self.experiment_backlog)} experiments in backlog")
            
            # Restore other state
            if "current_plan_iteration" in state:
                self.current_plan_iteration = state["current_plan_iteration"]
            
            if "baseline_rate" in state:
                self.baseline_rate = state["baseline_rate"]
                print(f"   📊 Restored baseline rate: {self.baseline_rate:.0f} RPS")
            
            print(f"   ✅ State restored from {state_file}")
            
        except Exception as e:
            print(f"   ⚠️  Could not restore state: {e}")
            # Continue with fresh state
    
    def _save_state(self) -> None:
        """Save current optimization state to disk."""
        state_file = self.data_dir / "optimization_state.json"
        
        # Find best iteration
        try:
            best_iter, best_result = self.convergence.find_best()
            self.convergence.best_iteration = best_iter
            self.convergence.best_config = best_result.config
        except ValueError:
            pass
        
        state = {
            "timestamp": datetime.now().isoformat(),
            "iteration": self.iteration,
            "current_config": self.current_config,
            "converged": self.convergence.converged,
            "convergence_reason": self.convergence.reason,
            "best_iteration": self.convergence.best_iteration,
            "best_config": self.convergence.best_config,
            "history": [r.to_dict() for r in self.convergence.history],
            # Persist tried configs and backlog for crash recovery
            "tried_configs": list(self.tried_configs),
            "experiment_backlog": [
                {
                    "priority": exp.priority,
                    "config_patch": exp.config_patch,
                    "config_removals": exp.config_removals,
                    "hypothesis": exp.hypothesis,
                    "expected_improvement": exp.expected_improvement,
                    "success_criteria": exp.success_criteria,
                    "source_experts": exp.source_experts,
                    "confidence": exp.confidence,
                    "abort_conditions": exp.abort_conditions,
                }
                for exp in self.experiment_backlog
            ],
            "current_plan_iteration": self.current_plan_iteration,
            "baseline_rate": self.baseline_rate,
        }
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    async def _shutdown(self) -> None:
        """Clean shutdown."""
        print("\n🧹 Shutting down...")
        
        # Stop health servers
        print("   Stopping health servers...")
        await self.benchmark_agent.stop_health()
        await self.profiler_agent.stop_health()
        await self.coordinator_agent.stop_health()
        print("   ✓ Health servers stopped")
        
        # Print final summary
        self._print_summary()
        
        # Save final state
        self._save_state()
    
    def _print_summary(self) -> None:
        """Print optimization summary."""
        print("\n" + "="*80)
        print("  📊 OPTIMIZATION SUMMARY")
        print("="*80)
        
        if not self.convergence.history:
            print("\n   No iterations completed")
            return
        
        # Find best
        best_iter, best = self.convergence.find_best()
        
        print(f"\n📈 Iteration History ({len(self.convergence.history)} iterations):")
        print(f"{'Iter':>6} {'Throughput':>12} {'Latency':>10} {'Error':>8} {'Config':>30}")
        print("-" * 80)
        
        baseline = self.convergence.history[0]
        for r in self.convergence.history:
            marker = " ⭐" if r.iteration == best_iter else ""
            config_str = json.dumps(r.config)[:28]
            print(
                f"{r.iteration:>6} "
                f"{r.metrics.get('throughput', 0):>11.1f} "
                f"{r.metrics.get('latency_p99', 0):>9.0f} "
                f"{r.metrics.get('error_rate', 0):>7.1%} "
                f"{config_str:>30}{marker}"
            )
        
        print(f"\n🏆 Best Configuration (Iteration {best_iter}):")
        print(f"   Config: {json.dumps(best.config, indent=2)}")
        print(f"   Throughput: {best.metrics.get('throughput', 0):.1f} RPS")
        print(f"   Latency p99: {best.metrics.get('latency_p99', 0):.1f} ms")
        
        # Calculate improvement
        if len(self.convergence.history) > 1:
            baseline_tp = baseline.metrics.get('throughput', 0)
            best_tp = best.metrics.get('throughput', 0)
            if baseline_tp > 0:
                improvement = ((best_tp - baseline_tp) / baseline_tp) * 100
                print(f"\n📊 Improvement: {improvement:+.1f}% throughput")
        
        if self.convergence.converged:
            print(f"\n🎯 Converged: {self.convergence.reason}")
        elif self.shutdown_requested:
            print(f"\n⏹️  Stopped by user")
        
        print(f"\n💾 State saved to: {self.data_dir / 'optimization_state.json'}")


async def main():
    """CLI entry point for the Inference Engine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inference Forge - Continuous vLLM Optimization"
    )
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Model to optimize")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max iterations")
    parser.add_argument("--no-improvement-limit", type=int, default=3, 
                        help="Stop after N iterations without improvement")
    parser.add_argument("--force-iterations", action="store_true",
                        help="Run all max-iterations regardless of early convergence")
    
    args = parser.parse_args()
    
    # Check for API keys
    if not any([
        os.environ.get("MOONSHOT_API_KEY"),
        os.environ.get("GEMINI_API_KEY"),
        os.environ.get("ANTHROPIC_API_KEY"),
        os.environ.get("OPENAI_API_KEY"),
    ]):
        print("❌ ERROR: Set at least one LLM API key:")
        print("   - MOONSHOT_API_KEY")
        print("   - GEMINI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - OPENAI_API_KEY")
        return 1
    
    engine = InferenceEngine(
        model_name=args.model,
        port=args.port,
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        no_improvement_limit=args.no_improvement_limit,
        force_iterations=args.force_iterations,
    )
    
    convergence = await engine.run()
    
    return 0 if convergence.converged else 1


if __name__ == "__main__":
    import os
    exit(asyncio.run(main()))
