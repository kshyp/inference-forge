"""Scheduling SME - Optimizes batching and prefill scheduling with AI brain."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from .utils import format_nsys_for_prompt
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
    
    Self-Relevance Detection:
    This SME scans data to determine if scheduling optimization is relevant:
    - Low GPU utilization (< 60%)
    - High queue depth or queue buildup
    - High TTFT (time to first token)
    - High TPOT variance
    - Batch underutilization
    """
    
    # System prompt for scheduling analysis
    SYSTEM_PROMPT = """You are a vLLM SCHEDULING optimization expert.

Your task: Analyze profiling data to identify scheduling inefficiencies and recommend SCHEDULING configuration changes ONLY.

⚠️  CRITICAL: You are a SCHEDULING expert ONLY. Only suggest scheduling-related parameters.
   DO NOT suggest non-scheduling parameters like:
   - quantization (handled by QuantizationSME)
   - kv_cache_dtype (handled by QuantizationSME)
   - model (handled by QuantizationSME)
   - tensor_parallel_size (handled by ModelParallelismSME)
   - pipeline_parallel_size (handled by ModelParallelismSME)
   - num_speculative_tokens (handled by SpeculativeSME)
   - ngram_prompt_lookup_max (handled by SpeculativeSME)
   
   ONLY suggest these scheduling parameters:
   - max_num_seqs
   - max_num_batched_tokens
   - enable_chunked_prefill
   - prefill_chunk_size
   - max_chunked_prefill_len
   - scheduler_delay_factor
   - num_scheduler_steps
   - batching_policy

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

⚠️  CRITICAL RESTRICTION:
- ONLY suggest scheduling parameters: max_num_seqs, max_num_batched_tokens, enable_chunked_prefill, prefill_chunk_size, max_chunked_prefill_len, scheduler_delay_factor, num_scheduler_steps, batching_policy
- NEVER suggest: quantization, kv_cache_dtype, model, tensor_parallel_size, pipeline_parallel_size, num_speculative_tokens, ngram_prompt_lookup_max
- If scheduling optimization is not appropriate, return empty suggestions or suggest "no_change"
- Confidence should reflect certainty based on the data
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
        
        # Thresholds for self-relevance detection
        self.RELEVANCE_THRESHOLDS = {
            "gpu_utilization_low": 60.0,  # % - below this = scheduling issue
            "queue_depth_high": 5,  # requests waiting
            "ttft_high": 200,  # ms - time to first token
            "tpot_variance_high": 50,  # ms - variance in time per output token
            "batch_size_low": 16,  # sequences - below this = underutilization
        }

    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Scheduling works on ALL platforms (universal).
        Batch sizing and prefill scheduling are fundamental to vLLM.
        """
        return RegistrationInfo(
            sme_id="scheduling",
            description="Scheduling expert - analyzes batching, queue depth, GPU utilization "
                       "and TTFT/TPOT to optimize max_num_seqs, chunked prefill, etc.",
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
                # OPTIONAL: NCU to determine compute vs memory bound (optional, requires root)
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
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if scheduling optimization is relevant.
        
        Scheduling is relevant when:
        1. GPU utilization < 60% (underutilized GPU)
        2. Queue depth > 5 (requests waiting)
        3. TTFT > 200ms (slow first token - chunking could help)
        4. TPOT variance > 50ms (prefill interfering with decode)
        5. Batch size < 16 (underutilized batching)
        
        Args:
            profile_dir: Path to directory containing profiling data files
            profiling_data: Pre-loaded profiling data
            benchmark_metrics: Benchmark results
            
        Returns:
            Tuple of (is_relevant, relevance_score, reason)
        """
        relevance_signals = []
        relevance_score = 0.0
        
        # Extract data
        system_metrics = profiling_data.get("system_metrics_report", {})
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        vllm_metrics = vllm_logs.get("metrics", {})
        ncu_report = profiling_data.get("ncu_report", {})
        
        # Check 1: GPU utilization (low = scheduling issue)
        gpu_util = system_metrics.get("gpu_utilization_percent", 100)
        if gpu_util < self.RELEVANCE_THRESHOLDS["gpu_utilization_low"]:
            relevance_signals.append(f"low_gpu_utilization ({gpu_util:.1f}%)")
            relevance_score += 0.35
        
        # Check 2: Queue depth (high = scheduling backlog)
        queue_depth = vllm_metrics.get("scheduler_queue_depth", 0)
        if isinstance(queue_depth, list) and queue_depth:
            queue_depth = max(queue_depth)
        if queue_depth > self.RELEVANCE_THRESHOLDS["queue_depth_high"]:
            relevance_signals.append(f"queue_buildup ({queue_depth} requests)")
            relevance_score += 0.25
        
        # Check 3: TTFT (high = prefill too slow, chunking could help)
        ttft_p99 = benchmark_metrics.get("ttft_p99", 0)
        if isinstance(ttft_p99, dict):
            ttft_p99 = ttft_p99.get("p99", 0)
        if ttft_p99 > self.RELEVANCE_THRESHOLDS["ttft_high"]:
            relevance_signals.append(f"high_ttft ({ttft_p99:.0f}ms p99)")
            relevance_score += 0.2
        
        # Check 4: Current batch settings (might be suboptimal)
        max_num_seqs = vllm_config.get("max_num_seqs", 256)
        enable_chunked_prefill = vllm_config.get("enable_chunked_prefill", False)
        
        if max_num_seqs > 128 and gpu_util < 70:
            relevance_signals.append(f"large_batch_low_utilization (max_num_seqs={max_num_seqs}, gpu={gpu_util:.1f}%)")
            relevance_score += 0.1
        
        if not enable_chunked_prefill and ttft_p99 > 150:
            relevance_signals.append(f"chunked_prefill_disabled_with_high_ttft")
            relevance_score += 0.1
        
        # Check 5: Batch utilization from metrics
        batch_size = vllm_metrics.get("scheduler_batch_size", 0)
        if isinstance(batch_size, list) and batch_size:
            avg_batch = sum(batch_size) / len(batch_size)
            if avg_batch < self.RELEVANCE_THRESHOLDS["batch_size_low"]:
                relevance_signals.append(f"batch_underutilization (avg={avg_batch:.1f})")
                relevance_score += 0.15
        
        # Determine relevance
        is_relevant = relevance_score >= 0.25  # At least 2 signals or strong signal needed
        
        if is_relevant:
            reason = f"Scheduling optimization relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = (f"No scheduling optimization signals detected "
                         f"(GPU util={gpu_util:.1f}%, queue_depth={queue_depth}, "
                         f"TTFT p99={ttft_p99:.0f}ms)")
        
        return is_relevant, min(relevance_score, 1.0), reason
    
    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze scheduling efficiency using multi-LLM consensus.
        
        Key signals analyzed:
        - Low GPU utilization + queue buildup → increase batch size
        - High TTFT → enable chunked prefill
        - Large TPOT variance → prefill interference, need chunking
        - Scheduler starvation → adjust delay factor
        
        Args:
            profile_dir: Path to directory containing profiling data
            profiling_data: Collected profiler outputs (vllm_logs, ncu_report, etc.)
            benchmark_metrics: Benchmark results (TTFT, TPOT, throughput)
        
        Returns:
            SMEResponse with consensus-based findings and suggestions
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # First, check relevance
        is_relevant, relevance_score, relevance_reason = self.scan_data(
            profile_dir, profiling_data, benchmark_metrics
        )
        
        if not is_relevant:
            return SMEResponse(
                findings={
                    "primary_bottleneck": "not_scheduling_related",
                    "scheduling_candidate": False,
                },
                suggestions=[],
                confidence=0.0,
                is_relevant=False,
                relevance_score=relevance_score,
                relevance_reason=relevance_reason
            )
        
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
                confidence=0.0,
                is_relevant=True,
                relevance_score=relevance_score,
                relevance_reason=relevance_reason
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
                      if suggestions else 0.0,
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_reason=relevance_reason
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
        
        # Extract NSYS analysis (enhanced with LLM insights)
        nsys_data = profiling_data.get("nsys_report", {})
        nsys_section = format_nsys_for_prompt(nsys_data, include_timeline=True)
        
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

{nsys_section}

## Benchmark Results
```json
{json.dumps(benchmark_metrics, indent=2)}
```

## Analysis Task

1. Identify the PRIMARY scheduling bottleneck from the data
2. Recommend specific config_changes to improve performance
3. Estimate expected improvement (quantitative where possible)

Use the NSYS timeline analysis to identify:
- Idle gaps between kernels (scheduling overhead)
- Prefill vs decode phase efficiency
- GPU utilization patterns

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
