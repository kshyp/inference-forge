"""Agent 3: Coordinator.

The "Brain" agent that:
1. Discovers platform and registers SMEs
2. Receives profiling results from ProfilerAgent
3. Makes ALL profiling data available to ALL SMEs in the profile directory
4. Consults ALL SMEs - each SME determines its own relevance by scanning data
5. Synthesizes recommendations into ranked experiment plan
6. Hands off to BenchmarkAgent for next iteration

NOTE: The Coordinator NO LONGER decides which SME gets triggered.
All profiling data is stored in experiments/data/profile/{experiment_id}/
and ALL SMEs have access to scan and analyze it themselves.
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
from forge.hardware import detect_hardware, HardwareCapabilities

from .synthesis import ExperimentSynthesizer, SynthesisConfig


class CoordinatorAgent(BaseAgent):
    """
    Coordinator Agent (Agent 3) - The brain of Inference Forge.
    
    Workflow:
    1. Discover platform → Register SMEs
    2. Receive profiling data from ProfilerAgent
    3. Store data in profile directory for ALL SMEs to access
    4. Consult ALL SMEs - each SME scans data to determine relevance
    5. Synthesize recommendations → RankedExperiment list
    6. Hand off to BenchmarkAgent for next iteration
    
    NOTE: This is a data-driven architecture. The Coordinator does NOT
    pre-filter SMEs. Each SME determines its own relevance by scanning
    the profiling data in the profile directory.
    
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
        
        # Profile directory where all profiling data is stored
        self.profile_dir = self.data_dir / "profile"
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
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
            "profile_dir": "./data/profile/exp_123"  # Path to profiling data
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
        profile_dir_override = payload.get("profile_dir")
        
        self.current_iteration = iteration
        
        print(f"\n{'='*70}")
        print(f"COORDINATOR ANALYSIS - Iteration {iteration}")
        print(f"{'='*70}")
        
        # Determine profile directory
        if profile_dir_override:
            profile_dir = Path(profile_dir_override)
        elif parent_id:
            profile_dir = self.profile_dir / str(parent_id)
        else:
            profile_dir = self.profile_dir / "default"
        
        # Ensure profile directory exists
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Profile directory: {profile_dir}")
        
        # Step 1: Extract profiling data for SMEs
        profiling_data = profiling_result.get("collected_data", {})
        benchmark_metrics = profiling_result.get("benchmark_metrics", {})
        
        # Step 2: Add hardware capabilities to profiling data for all SMEs
        if self.platform_info and "hardware_capabilities" in self.platform_info:
            profiling_data["hardware_capabilities"] = self.platform_info["hardware_capabilities"]
        
        # Step 3: Save profiling data to disk for SMEs to access directly
        self._save_profiling_data(profile_dir, profiling_data, benchmark_metrics)
        
        # Step 4: Consult ALL SMEs
        # Each SME will scan the data to determine if it's relevant
        responses, sme_ids = await self.sme_registry.consult(
            profile_dir, profiling_data, benchmark_metrics
        )
        
        if not responses:
            print("⚠️  No SME responses - cannot generate recommendations")
            return {
                "success": False,
                "error": "No SME responses",
                "profile_dir": str(profile_dir)
            }
        
        # Step 5: Synthesize into experiment plan
        experiment_plan = self.synthesizer.synthesize(
            sme_responses=responses,
            sme_ids=sme_ids,
            current_config=current_config,
            iteration=iteration,
            parent_experiment_id=parent_id
        )
        
        # Step 6: Store result for handoff
        await self._result_queue.put(experiment_plan)
        
        # Step 7: Save to disk
        plan_file = self._save_experiment_plan(experiment_plan)
        
        # Print summary
        self._print_summary(experiment_plan)
        
        return {
            "success": True,
            "experiment_plan": {
                "iteration": experiment_plan.iteration,
                "experiments_count": len(experiment_plan.experiments),
                "smes_consulted": experiment_plan.smes_consulted,
            },
            "plan_file": str(plan_file),
            "profile_dir": str(profile_dir),
            "message": f"Generated {len(experiment_plan.experiments)} ranked experiments"
        }
    
    async def _action_plan_experiments(self, task: Task) -> Dict[str, Any]:
        """Alternative action name for analyze."""
        return await self._action_analyze(task)
    
    def _discover_platform(self) -> Dict[str, Any]:
        """Detect hardware platform with full capabilities."""
        print("\n🔍 Discovering platform...")
        
        try:
            # Use hardware detector for comprehensive detection
            caps = detect_hardware()
            
            platform = {
                "type": "nvidia_cuda" if caps.gpu_count > 0 else "cpu",
                "gpu_model": caps.gpu_model,
                "gpu_count": caps.gpu_count,
                "cuda_version": caps.cuda_version,
                "driver_version": caps.driver_version,
                "hardware_capabilities": caps.to_dict(),
            }
            
            print(f"   Platform: {platform['type']}")
            print(f"   GPUs: {platform['gpu_count']}x {platform['gpu_model']}")
            print(f"   Architecture: {caps.gpu_architecture.value}")
            print(f"   VRAM: {caps.vram_gb:.1f} GB")
            print(f"   Native precisions: {', '.join(sorted(caps.native_precisions))}")
            print(f"   Quantization support: {', '.join(caps.recommended_weight_quant[:2])}")
            
        except Exception as e:
            print(f"   ⚠️ Hardware detection failed: {e}")
            print("   Using fallback platform detection")
            
            # Fallback to basic detection
            platform = {
                "type": "nvidia_cuda",
                "gpu_model": "A100",
                "gpu_count": 1,
                "cuda_version": "12.2",
                "driver_version": "535.104",
                "hardware_capabilities": {},
            }
            
            print(f"   Platform: {platform['type']}")
            print(f"   GPUs: {platform['gpu_count']}x {platform['gpu_model']}")
        
        return platform
    
    def _save_profiling_data(self, 
                             profile_dir: Path, 
                             profiling_data: Dict[str, Any],
                             benchmark_metrics: Dict[str, Any]) -> None:
        """Save profiling data to disk for SMEs to access directly."""
        # Save collected data
        data_file = profile_dir / "profiling_data.json"
        with open(data_file, "w") as f:
            json.dump(profiling_data, f, indent=2, default=str)
        
        # Save benchmark metrics
        metrics_file = profile_dir / "benchmark_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(benchmark_metrics, f, indent=2, default=str)
        
        # Also save individual data components if they exist
        if "vllm_logs" in profiling_data:
            vllm_file = profile_dir / "vllm_logs.json"
            with open(vllm_file, "w") as f:
                json.dump(profiling_data["vllm_logs"], f, indent=2, default=str)
        
        if "nsys_report" in profiling_data:
            nsys_file = profile_dir / "nsys_report.json"
            with open(nsys_file, "w") as f:
                json.dump(profiling_data["nsys_report"], f, indent=2, default=str)
        
        if "ncu_report" in profiling_data:
            ncu_file = profile_dir / "ncu_report.json"
            with open(ncu_file, "w") as f:
                json.dump(profiling_data["ncu_report"], f, indent=2, default=str)
        
        if "system_metrics_report" in profiling_data:
            sys_file = profile_dir / "system_metrics.json"
            with open(sys_file, "w") as f:
                json.dump(profiling_data["system_metrics_report"], f, indent=2, default=str)
        
        print(f"   💾 Saved profiling data to {profile_dir}")
    
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
