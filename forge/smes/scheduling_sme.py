"""Scheduling SME - Optimizes batching and prefill scheduling with AI brain."""

import json
from typing import Dict, Any, Optional

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from forge.llm import (
    IntelligencePool,
    Prompt,
    ConsensusEngine,
    ConsensusConfig,
    get_global_intelligence_pool,
)


class SchedulingSME(BaseSME):
    """
    Scheduling Expert with multi-LLM AI brain.
    
    Optimizes:
    - Batch size (max_num_seqs, max_num_batched_tokens)
    - Chunked prefill (enable_chunked_prefill, prefill_chunk_size)
    - Scheduler policy (scheduler_delay_factor, policy type)
    - Request prioritization
    
    Uses multi-LLM consensus to analyze scheduling inefficiencies and 
    recommend configuration changes.
    """
    
    # System prompt for scheduling analysis
    SYSTEM_PROMPT = """You are a vLLM scheduling optimization expert.

Your task: Analyze profiling data to identify scheduling inefficiencies and recommend configuration changes.

Key concepts:
- GPU utilization: % of GPU compute being used (low = wasted capacity)
- Queue depth: Number of requests waiting to be scheduled (high = backlog)
- Batch size: Number of sequences processed together (larger = more throughput)
- Chunked prefill: Splitting prefill computation into chunks (reduces TTFT)
- Scheduler starvation: When requests wait too long due to scheduling decisions

Respond with JSON only:
{
  "findings": {
    "primary_bottleneck": "string - one of: batch_underutilization, queue_starvation, prefill_interference, scheduler_starvation, memory_pressure",
    "diagnosis": "string - 2-3 sentence technical explanation of the root cause"
  },
  "suggestions": [
    {
      "priority": 1,
      "config_changes": {"param_name": "value"},
      "expected_improvement": "string e.g. +40% throughput, -30% TTFT p99",
      "confidence": 0.0-1.0,
      "rationale": "string explaining why this change helps"
    }
  ],
  "confidence": 0.0-1.0
}

Guidelines:
- Set priority 1 for the most impactful change
- Confidence should reflect certainty based on the data
- Config changes should be specific vLLM parameters
- Expected improvement should be quantitative where possible"""

    def __init__(self, intelligence_pool: Optional[IntelligencePool] = None):
        """
        Initialize SchedulingSME.
        
        Args:
            intelligence_pool: Optional custom pool. Uses global pool if not provided.
        """
        self.pool = intelligence_pool or get_global_intelligence_pool()
        
        # Consensus configuration for scheduling decisions
        self.consensus_config = ConsensusConfig(
            similarity_threshold=0.75,
            min_agreement_ratio=0.5,
            require_unanimous_for_p1=False,
            weight_by_quality=True,
        )
    
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Scheduling works on ALL platforms (universal).
        Batch sizing and prefill scheduling are fundamental to vLLM.
        """
        return RegistrationInfo(
            sme_id="scheduling",
            triggers=[
                "low_gpu_utilization",
                "queue_buildup",
                "high_ttft",
                "high_tpot_variance",
                "batch_underutilization",
                "prefill_decode_interference",
                "scheduler_starvation",
                # Baseline exploration triggers (when no profiler data available)
                "baseline_exploration",
                "low_throughput",
                "moderate_throughput",
                "high_latency",
                "moderate_latency",
            ],
            data_requirements=[
                # REQUIRED: vLLM debug logs - contains scheduler decisions, queue depth, batch sizes
                DataRequirement(
                    data_type="vllm_logs",
                    required=True,
                    extractors=[
                        # Scheduler config
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "enable_chunked_prefill",
                        "prefill_chunk_size",
                        "scheduler_delay_factor",
                        "num_scheduler_steps",
                        "batching_policy",
                        # Debug log metrics (requires --log-level=DEBUG)
                        "scheduler_queue_depth",
                        "scheduler_batch_size",
                        "scheduler_batch_tokens",
                        "kv_cache_usage_percent",
                        "gpu_memory_peak_mb",
                    ]
                ),
                # REQUIRED: Host CPU/memory stats (always available)
                DataRequirement(
                    data_type="vmstat_report",
                    required=True,
                    extractors=[
                        "cpu_user_pct",
                        "cpu_system_pct",
                        "cpu_wait_pct",
                        "memory_free_kb",
                        "context_switches_s",
                    ]
                ),
                # OPTIONAL: Per-CPU stats
                DataRequirement(
                    data_type="mpstat_report",
                    required=False,
                    extractors=[
                        "avg_usr_pct",
                        "avg_sys_pct",
                        "avg_iowait_pct",
                        "num_cpus",
                    ]
                ),
                # REQUIRED: Benchmark results for latency/throughput
                DataRequirement(
                    data_type="benchmark_metrics",
                    required=True,
                    extractors=[
                        "ttft_p50",
                        "ttft_p99",
                        "tpot_p50", 
                        "tpot_p99",
                        "throughput_rps",
                        "output_tokens_per_sec",
                    ]
                ),
                # REQUIRED: System metrics (always available, no root needed)
                DataRequirement(
                    data_type="system_metrics_report",
                    required=True,
                    extractors=[
                        "gpu_utilization_percent",
                        "memory_used_mb",
                        "memory_total_mb",
                        "temperature_celsius",
                        "power_draw_watts",
                    ]
                ),
                # REQUIRED: NCU to determine compute vs memory bound (optional, requires root)
                DataRequirement(
                    data_type="ncu_report",
                    required=False,  # Changed to optional - system metrics can substitute
                    extractors=[
                        "compute_utilization_percent",
                        "memory_bandwidth_utilization_percent",
                        "dram_throughput_gbps",
                    ]
                ),
                # OPTIONAL: NSys for GPU timeline and idle gaps
                DataRequirement(
                    data_type="nsys_report",
                    required=False,
                    extractors=[
                        "gpu_utilization_percent",
                        "gpu_idle_time_ms",
                        "kernel_gap_time_ms",
                    ]
                ),
            ]
        )
    
    async def analyze(self, profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze scheduling efficiency using multi-LLM consensus.
        
        Key signals analyzed:
        - Low GPU utilization + queue buildup → increase batch size
        - High TTFT → enable chunked prefill
        - Large TPOT variance → prefill interference, need chunking
        - Scheduler starvation → adjust delay factor
        
        Args:
            profiling_data: Collected profiler outputs (vllm_logs, ncu_report, etc.)
            benchmark_metrics: Benchmark results (TTFT, TPOT, throughput)
        
        Returns:
            SMEResponse with consensus-based findings and suggestions
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. Prepare structured prompt with all data
        prompt = self._build_analysis_prompt(profiling_data, benchmark_metrics)
        
        # 2. Call all available intelligence sources in parallel
        # Use deterministic mode (temperature=0) for reproducible results
        print(f"   [SchedulingSME] Calling LLM pool (sources: {len(self.pool.sources)})...")
        call_results = await self.pool.call_all(
            prompt,
            deterministic=self.consensus_config.temperature <= 0.0
        )
        
        print(f"   [SchedulingSME] Got {len(call_results)} call results")
        for i, cr in enumerate(call_results):
            status = "✓" if cr.success else "✗"
            error = f" ({cr.error})" if cr.error else ""
            print(f"      {status} {cr.source.provider}/{cr.source.model}{error}")
        
        if not call_results:
            print("   [SchedulingSME] ⚠️ No LLM intelligence sources available!")
            return SMEResponse(
                findings={"error": "No LLM intelligence sources available"},
                suggestions=[],
                confidence=0.0
            )
        
        # 3. Compute consensus across all model responses
        print(f"   [SchedulingSME] Computing consensus (min_agreement={self.consensus_config.min_agreement_ratio})...")
        consensus_result = ConsensusEngine.compute(
            call_results, 
            self.consensus_config
        )
        
        print(f"   [SchedulingSME] Consensus: {len(consensus_result.suggestions)} suggestions from {consensus_result.successful_models}/{consensus_result.total_models} models")
        
        # 4. Convert consensus to SMEResponse
        suggestions = []
        for cs in consensus_result.suggestions:
            print(f"      → Suggestion: {cs.config_changes} (confidence: {cs.weighted_confidence:.2f}, agreement: {cs.agreement_score:.2f})")
            suggestions.append(ExperimentSuggestion(
                config_changes=cs.config_changes,
                expected_improvement=cs.expected_improvement,
                confidence=cs.weighted_confidence,
                rationale=cs.rationale
            ))
        
        # 5. Extract primary finding from consensus
        primary_finding = self._extract_primary_finding(consensus_result)
        
        return SMEResponse(
            findings={
                "primary_bottleneck": primary_finding,
                "consensus_models": consensus_result.total_models,
                "successful_models": consensus_result.successful_models,
                "divergence_summary": consensus_result.divergence_report.summary,
            },
            suggestions=suggestions,
            confidence=consensus_result.suggestions[0].weighted_confidence 
                      if suggestions else 0.0
        )
    
    def _build_analysis_prompt(
        self, 
        profiling_data: Dict[str, Any],
        benchmark_metrics: Dict[str, Any]
    ) -> Prompt:
        """Build structured analysis prompt from profiling data."""
        
        # Extract vLLM config and runtime metrics
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        vllm_metrics = vllm_logs.get("metrics", {})
        
        # Extract NCU metrics
        ncu_report = profiling_data.get("ncu_report", {})
        
        # Build user prompt
        user_content = f"""Analyze this vLLM profiling data and recommend scheduling optimizations.

## Current Configuration
```json
{json.dumps(vllm_config, indent=2)}
```

## Runtime Metrics (from vLLM DEBUG logs)
```json
{json.dumps(vllm_metrics, indent=2)}
```

## GPU Metrics (from NCU)
```json
{json.dumps(ncu_report, indent=2)}
```

## Benchmark Results
```json
{json.dumps(benchmark_metrics, indent=2)}
```

## Analysis Task

1. Identify the PRIMARY scheduling bottleneck from the data
2. Recommend specific config_changes to improve performance
3. Estimate expected improvement (quantitative where possible)

Respond with JSON only (no markdown, no explanations outside JSON)."""

        return Prompt(
            system=self.SYSTEM_PROMPT,
            user=user_content
        )
    
    def _extract_primary_finding(self, consensus_result) -> str:
        """Extract primary bottleneck from consensus divergence report."""
        votes = consensus_result.divergence_report.bottleneck_votes
        
        if not votes:
            return "unknown"
        
        # Find most common bottleneck
        primary = max(votes.items(), key=lambda x: len(x[1]))
        return primary[0]
