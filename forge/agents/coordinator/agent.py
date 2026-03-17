"""Agent 3: Coordinator.

The "Brain" agent that:
1. Discovers platform and registers SMEs
2. Receives profiling results from ProfilerAgent
3. Extracts signals from profiling data
4. Consults relevant SMEs (using multi-LLM consensus)
5. Synthesizes recommendations into ranked experiment plan
6. Hands off to BenchmarkAgent for next iteration
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from forge.agents.base import BaseAgent
from forge.core.events import (
    BenchmarkResult,
    ExperimentPlan,
    HardwareInfo,
    ProfilingResult,
    RankedExperiment,
    Task,
)
from forge.core.state import StateStore
from forge.sme_registry import SMERegistry

from .synthesis import ExperimentSynthesizer, SynthesisConfig


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent (Agent 3) - The brain of Inference Forge.
    
    Workflow:
    1. Discover platform → Register SMEs
    2. Receive profiling data from ProfilerAgent
    3. Extract signals (triage)
    4. Consult relevant SMEs (async, parallel)
    5. Synthesize recommendations → RankedExperiment list
    6. Hand off to BenchmarkAgent for next iteration
    
    Attributes:
        sme_registry: Registry for SME management
        synthesizer: Engine for ranking experiments
        platform_info: Detected hardware platform
        experiment_history: Past experiments for convergence detection
    """
    
    def __init__(
        self,
        state_store: StateStore,
        health_port: int = 8083,
        data_dir: str = "./data",
        synthesis_config: Optional[SynthesisConfig] = None,
    ):
        super().__init__(
            agent_id="coordinator",
            state_store=state_store,
            health_port=health_port
        )
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.sme_registry = SMERegistry()
        self.synthesizer = ExperimentSynthesizer(config=synthesis_config)
        
        # State
        self.platform_info: Optional[Dict[str, Any]] = None
        self.experiment_history: List[Dict[str, Any]] = []
        self.current_iteration: int = 0
        
        # Task queue
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
    
    async def get_next_task(self) -> Optional[Task]:
        """Get next coordination task from queue."""
        try:
            return await asyncio.wait_for(self._task_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def submit_task(self, task: Task) -> None:
        """Submit a coordination task."""
        self._task_queue.put_nowait(task)
    
    async def get_experiment_plan(self) -> Optional[ExperimentPlan]:
        """Get completed experiment plan (for handoff to BenchmarkAgent)."""
        try:
            return await asyncio.wait_for(self._result_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def execute(self, task: Task) -> Dict[str, Any]:
        """
        Execute coordination task.
        
        Expected task payload:
        {
            "action": "analyze",  # or "register_smes", "plan_experiments"
            "profiling_result": {...},  # From ProfilerAgent
            "current_config": {...},  # Current vLLM config
            "iteration": 1,
            "parent_experiment_id": "uuid",
        }
        """
        payload = task.payload
        action = payload.get("action", "analyze")
        
        if action == "register_smes":
            return await self._action_register_smes(task)
        
        elif action == "analyze":
            return await self._action_analyze(task)
        
        elif action == "plan_experiments":
            return await self._action_plan_experiments(task)
        
        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}"
            }
    
    async def _action_register_smes(self, task: Task) -> Dict[str, Any]:
        """Register SMEs for the platform."""
        # Discover platform if not already done
        if not self.platform_info:
            self.platform_info = self._discover_platform()
        
        # Discover and register SMEs
        self.sme_registry.discover_sme_classes()
        registered = self.sme_registry.register_all(self.platform_info)
        
        # Get consolidated data requirements for profiler
        profiler_requirements = self.sme_registry.get_profiler_requirements()
        
        return {
            "success": True,
            "platform": self.platform_info,
            "registered_smes": list(registered.keys()),
            "profiler_requirements": profiler_requirements,
            "message": f"Registered {len(registered)} SMEs for platform"
        }
    
    async def _action_analyze(self, task: Task) -> Dict[str, Any]:
        """Analyze profiling results and produce experiment plan."""
        payload = task.payload
        
        # Extract inputs
        profiling_result = payload.get("profiling_result", {})
        current_config = payload.get("current_config", {})
        iteration = payload.get("iteration", self.current_iteration + 1)
        parent_id = payload.get("parent_experiment_id")
        
        self.current_iteration = iteration
        
        print(f"\n{'='*70}")
        print(f"COORDINATOR ANALYSIS - Iteration {iteration}")
        print(f"{'='*70}")
        
        # Step 1: Extract signals from profiling data
        signals = self._extract_signals(profiling_result)
        print(f"\n📊 Signals Detected: {signals}")
        
        if not signals:
            print("⚠️  No signals detected - running baseline exploration")
            signals = ["baseline_exploration"]
        
        # Step 2: Extract profiling data for SMEs
        profiling_data = profiling_result.get("collected_data", {})
        benchmark_metrics = profiling_result.get("benchmark_metrics", {})
        
        # Step 3: Consult SMEs
        responses = await self.sme_registry.consult(
            signals, profiling_data, benchmark_metrics
        )
        
        if not responses:
            print("⚠️  No SME responses - cannot generate recommendations")
            return {
                "success": False,
                "error": "No SME responses",
                "signals": signals
            }
        
        # Step 4: Synthesize into experiment plan
        sme_ids = self.sme_registry.get_registered_sme_ids()
        
        experiment_plan = self.synthesizer.synthesize(
            sme_responses=responses,
            sme_ids=sme_ids,
            current_config=current_config,
            iteration=iteration,
            parent_experiment_id=parent_id
        )
        
        # Step 5: Store result for handoff
        await self._result_queue.put(experiment_plan)
        
        # Step 6: Save to disk
        plan_file = self._save_experiment_plan(experiment_plan)
        
        # Print summary
        self._print_summary(experiment_plan)
        
        return {
            "success": True,
            "experiment_plan": {
                "iteration": experiment_plan.iteration,
                "experiments_count": len(experiment_plan.experiments),
                "signals": experiment_plan.signals_detected,
                "smes": experiment_plan.smes_consulted,
            },
            "plan_file": str(plan_file),
            "message": f"Generated {len(experiment_plan.experiments)} ranked experiments"
        }
    
    async def _action_plan_experiments(self, task: Task) -> Dict[str, Any]:
        """Alternative action name for analyze."""
        return await self._action_analyze(task)
    
    def _discover_platform(self) -> Dict[str, Any]:
        """Detect hardware platform."""
        # In reality: run nvidia-smi, check /proc/cpuinfo, etc.
        # For now: simulate single GPU CUDA
        
        print("\n🔍 Discovering platform...")
        
        platform = {
            "type": "nvidia_cuda",
            "gpu_model": "A100",
            "gpu_count": 1,
            "cuda_version": "12.2",
            "driver_version": "535.104"
        }
        
        print(f"   Platform: {platform['type']}")
        print(f"   GPUs: {platform['gpu_count']}x {platform['gpu_model']}")
        
        return platform
    
    def _extract_signals(self, profiling_result: Dict[str, Any]) -> List[str]:
        """
        Extract bottleneck signals from profiling data.
        
        This is rule-based triage. Could be enhanced with LLM-based triage.
        """
        signals = []
        
        # Extract data from result
        collected = profiling_result.get("collected_data", {})
        ncu = collected.get("ncu_report", {})
        vllm = collected.get("vllm_logs", {})
        benchmark = profiling_result.get("benchmark_metrics", {})
        
        # Check if we have any actual data or just baseline
        has_profiler_data = bool(ncu or vllm)
        
        if not has_profiler_data:
            # No profiler data - use baseline exploration signals
            # These will trigger all SMEs to provide recommendations
            throughput = benchmark.get("throughput", 0)
            latency = benchmark.get("latency_p99", 0)
            
            # Always add baseline exploration if no profiler data
            signals.append("baseline_exploration")
            
            # Add throughput-based signals
            if throughput > 0:
                if throughput < 30:
                    signals.append("low_throughput")
                elif throughput < 50:
                    signals.append("moderate_throughput")
                
                # Latency-based signals
                if latency > 1000:
                    signals.append("high_latency")
                elif latency > 500:
                    signals.append("moderate_latency")
            
            return signals
        
        # GPU utilization check (from NCU)
        gpu_util = ncu.get("compute_utilization_percent", 100)
        if gpu_util < 50:
            signals.append("low_gpu_utilization")
        elif gpu_util < 70:
            signals.append("moderate_gpu_utilization")
        
        # Memory bandwidth check
        mem_bw = ncu.get("memory_bandwidth_utilization_percent", 0)
        if mem_bw > 80:
            signals.append("memory_bandwidth_bound")
        
        # Queue depth check
        queue_depth = vllm.get("scheduler_queue_depth", 0)
        if isinstance(queue_depth, list) and queue_depth:
            max_queue = max(queue_depth)
            if max_queue > 10:
                signals.append("queue_buildup")
        elif isinstance(queue_depth, (int, float)) and queue_depth > 10:
            signals.append("queue_buildup")
        
        # KV cache pressure
        kv_usage = vllm.get("kv_cache_usage_percent", 0)
        if kv_usage > 85:
            signals.append("kv_cache_pressure")
        
        # Latency checks from benchmark
        ttft_p99 = benchmark.get("ttft_p99", 0)
        if isinstance(ttft_p99, dict):
            ttft_p99 = ttft_p99.get("p99", 0)
        if ttft_p99 > 500:
            signals.append("high_ttft")
        
        tpot_p99 = benchmark.get("tpot_p99", 0)
        if isinstance(tpot_p99, dict):
            tpot_p99 = tpot_p99.get("p99", 0)
        if tpot_p99 > 100:
            signals.append("high_tpot")
        
        return signals
    
    def _save_experiment_plan(self, plan: ExperimentPlan) -> Path:
        """Save experiment plan to disk."""
        exp_dir = self.data_dir / "experiments" / str(plan.parent_experiment_id or "new")
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        plan_file = exp_dir / f"experiment_plan_iter{plan.iteration}.json"
        
        # Convert to dict for serialization
        plan_dict = {
            "iteration": plan.iteration,
            "parent_experiment_id": str(plan.parent_experiment_id) if plan.parent_experiment_id else None,
            "signals_detected": plan.signals_detected,
            "smes_consulted": plan.smes_consulted,
            "synthesis_reasoning": plan.synthesis_reasoning,
            "experiments": [
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
                for exp in plan.experiments
            ]
        }
        
        with open(plan_file, "w") as f:
            json.dump(plan_dict, f, indent=2)
        
        return plan_file
    
    def _print_summary(self, plan: ExperimentPlan):
        """Print experiment plan summary."""
        print(f"\n📋 EXPERIMENT PLAN - Iteration {plan.iteration}")
        print(f"{'='*70}")
        print(f"Signals: {', '.join(plan.signals_detected)}")
        print(f"SMEs Consulted: {', '.join(plan.smes_consulted)}")
        print(f"\nRanked Experiments ({len(plan.experiments)}):")
        
        for exp in plan.experiments:
            print(f"\n  #{exp.priority} [Confidence: {exp.confidence:.2f}]")
            print(f"    Config: {exp.config_patch}")
            print(f"    Expected: {exp.expected_improvement}")
            print(f"    Sources: {', '.join(exp.source_experts)}")
        
        print(f"\n{'='*70}")
        print("✅ Experiment plan ready for BenchmarkAgent")
        print(f"{'='*70}\n")
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get data for checkpointing."""
        return {
            "platform_info": self.platform_info,
            "current_iteration": self.current_iteration,
            "experiment_history": self.experiment_history,
        }
    
    def restore_from_checkpoint(self, data: Dict[str, Any]) -> None:
        """Restore from checkpoint data."""
        self.platform_info = data.get("platform_info")
        self.current_iteration = data.get("current_iteration", 0)
        self.experiment_history = data.get("experiment_history", [])
    
    def get_profiler_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Get data requirements for profiler (convenience method)."""
        if not self.sme_registry.registered_smes:
            # Auto-register if needed
            if not self.platform_info:
                self.platform_info = self._discover_platform()
            self.sme_registry.discover_sme_classes()
            self.sme_registry.register_all(self.platform_info)
        
        return self.sme_registry.get_profiler_requirements()
