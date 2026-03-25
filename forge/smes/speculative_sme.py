"""Speculative Decoding SME - Optimizes draft model speculation."""

from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo


class SpeculativeSME(BaseSME):
    """
    Speculative Decoding Expert
    
    Optimizes:
    - Draft model selection (n-gram, custom draft model)
    - Number of speculative tokens
    - Acceptance rate tuning
    - Verification overhead
    
    Target: Reduce decode phase latency by predicting future tokens.
    
    Self-Relevance Detection:
    This SME scans data to determine if speculative decoding is relevant:
    - High TPOT (time per output token) > 50ms
    - Decode phase dominates latency
    - Available GPU memory for draft model
    - Low speculative acceptance rate (if already enabled)
    """
    
    # Thresholds for self-relevance detection
    RELEVANCE_THRESHOLDS = {
        "tpot_high": 50.0,  # ms - TPOT above this suggests decode bottleneck
        "acceptance_rate_low": 0.6,  # 60% - below this needs tuning
        "draft_overhead_high": 20.0,  # ms - draft model too slow
        "available_memory_min": 2048,  # MB - minimum free memory for draft model
    }

    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Speculative decoding works on CUDA and ROCm.
        Requires sufficient GPU memory for draft model.
        """
        platform_type = platform_info.get("type", "")
        
        if platform_type not in ("nvidia_cuda", "amd_rocm"):
            return None
        
        return RegistrationInfo(
            sme_id="speculative",
            description="Speculative decoding expert - analyzes TPOT, decode phase latency "
                       "and acceptance rates to recommend draft models and speculation settings",
            data_requirements=[
                # REQUIRED: vLLM logs for speculative decoding config and stats
                DataRequirement(
                    data_type="vllm_logs",
                    required=True,
                    extractors=[
                        # Configuration
                        "speculative_model",
                        "num_speculative_tokens",
                        "speculative_disable_by_batch_size",
                        "speculative_acceptance_threshold",
                        # Stats (from debug logs)
                        "spec_acceptance_rate",         # Tokens accepted / tokens proposed
                        "spec_draft_latency_ms",        # Draft model inference time
                        "spec_verification_latency_ms", # Target model verification time
                        "spec_total_tokens_generated",
                        "spec_draft_tokens_accepted",
                        "spec_draft_tokens_rejected",
                    ]
                ),
                # REQUIRED: Benchmark metrics for decode phase
                DataRequirement(
                    data_type="benchmark_metrics",
                    required=True,
                    extractors=[
                        "tpot_p50",
                        "tpot_p99",
                        "decode_tokens_per_sec",
                        "itl_p50",  # Inter-token latency
                        "itl_p99",
                    ]
                ),
                # OPTIONAL: NCU for draft vs target model efficiency
                DataRequirement(
                    data_type="ncu_report",
                    required=False,
                    extractors=[
                        "draft_model_compute_utilization",
                        "target_model_compute_utilization",
                        "draft_model_memory_bandwidth",
                        "verification_overhead_percent",
                    ]
                ),
                # REQUIRED: vLLM memory status (need room for draft model)
                DataRequirement(
                    data_type="vllm_logs",
                    required=True,
                    extractors=[
                        "gpu_memory_utilization_actual",
                        "kv_cache_usage_percent",
                        "available_memory_for_draft_mb",
                    ]
                ),
                # OPTIONAL: NSys for timeline (draft + verification phases)
                DataRequirement(
                    data_type="nsys_report",
                    required=False,
                    extractors=[
                        "spec_draft_kernel_time_ms",
                        "spec_verify_kernel_time_ms",
                        "spec_gpu_idle_between_draft_verify_ms",
                    ]
                ),
            ]
        )
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if speculative decoding is relevant.
        
        Speculative decoding is relevant when:
        1. TPOT > 50ms (slow decode phase)
        2. Available GPU memory > 2GB (room for draft model)
        3. Low acceptance rate if already enabled (needs tuning)
        4. High draft overhead if already enabled (draft model too large)
        
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
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        vllm_metrics = vllm_logs.get("metrics", {})
        
        # Check 1: TPOT (high = decode bottleneck)
        tpot_p99 = benchmark_metrics.get("tpot_p99", 0)
        if isinstance(tpot_p99, dict):
            tpot_p99 = tpot_p99.get("p99", 0)
        if tpot_p99 > self.RELEVANCE_THRESHOLDS["tpot_high"]:
            relevance_signals.append(f"high_tpot ({tpot_p99:.1f}ms p99)")
            relevance_score += 0.4
        
        # Check 2: Available memory (need room for draft model)
        available_mem = vllm_metrics.get("available_memory_for_draft_mb", 0)
        gpu_mem_util = vllm_metrics.get("gpu_memory_utilization_actual", 100)
        if available_mem > self.RELEVANCE_THRESHOLDS["available_memory_min"] or gpu_mem_util < 80:
            relevance_signals.append(f"memory_available_for_draft ({available_mem}MB free)")
            relevance_score += 0.2
        else:
            # Not enough memory - speculative decoding won't work
            relevance_score -= 0.3
        
        # Check 3: Already enabled but poor performance
        spec_model = vllm_config.get("speculative_model", "")
        num_spec_tokens = vllm_config.get("num_speculative_tokens", 0)
        
        if spec_model or num_spec_tokens > 0:
            # Already enabled - check if needs tuning
            acceptance_rate = vllm_metrics.get("spec_acceptance_rate", 0)
            if acceptance_rate < self.RELEVANCE_THRESHOLDS["acceptance_rate_low"]:
                relevance_signals.append(f"low_acceptance_rate ({acceptance_rate:.1%})")
                relevance_score += 0.25
            
            draft_latency = vllm_metrics.get("spec_draft_latency_ms", 0)
            if draft_latency > self.RELEVANCE_THRESHOLDS["draft_overhead_high"]:
                relevance_signals.append(f"high_draft_overhead ({draft_latency:.1f}ms)")
                relevance_score += 0.15
        else:
            # Not enabled yet - candidate for enabling
            if tpot_p99 > self.RELEVANCE_THRESHOLDS["tpot_high"]:
                relevance_signals.append("speculative_not_enabled_but_high_tpot")
                relevance_score += 0.15
        
        # Determine relevance
        is_relevant = relevance_score >= 0.35
        
        if is_relevant:
            reason = f"Speculative decoding relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = (f"Speculative decoding not relevant (TPOT={tpot_p99:.1f}ms, "
                         f"available_mem={available_mem}MB)")
        
        return is_relevant, min(relevance_score, 1.0), reason
    
    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze speculative decoding efficiency using multi-LLM consensus.
        
        Key signals:
        - Low acceptance rate (< 60%): draft model too ambitious or mismatched
        - High draft latency: draft model too large
        - High TPOT without speculation: candidate for enabling
        - Low GPU memory: can't fit draft model
        
        Args:
            profile_dir: Path to directory containing profiling data
            profiling_data: Collected profiler outputs
            benchmark_metrics: Benchmark results
        
        Returns:
            SMEResponse with findings and suggestions
        """
        # Check relevance first
        is_relevant, relevance_score, relevance_reason = self.scan_data(
            profile_dir, profiling_data, benchmark_metrics
        )
        
        if not is_relevant:
            return SMEResponse(
                findings={
                    "primary_bottleneck": "not_speculative_related",
                    "speculative_candidate": False,
                },
                suggestions=[],
                confidence=0.0,
                is_relevant=False,
                relevance_score=relevance_score,
                relevance_reason=relevance_reason
            )
        
        # TODO: Implement full AI brain with multi-LLM consensus
        # For now, return rule-based suggestions
        suggestions = []
        findings = {}
        
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        vllm_metrics = vllm_logs.get("metrics", {})
        
        # Check if already enabled
        spec_model = vllm_config.get("speculative_model", "")
        num_spec_tokens = vllm_config.get("num_speculative_tokens", 0)
        
        if spec_model or num_spec_tokens > 0:
            # Already enabled - check for tuning opportunities
            acceptance_rate = vllm_metrics.get("spec_acceptance_rate", 0)
            draft_latency = vllm_metrics.get("spec_draft_latency_ms", 0)
            
            findings["speculative_enabled"] = True
            findings["current_acceptance_rate"] = acceptance_rate
            findings["current_draft_latency_ms"] = draft_latency
            
            if acceptance_rate < 0.5:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"num_speculative_tokens": max(1, num_spec_tokens // 2)},
                    expected_improvement="+10-20% acceptance rate by reducing speculation depth",
                    confidence=0.6,
                    rationale=f"Acceptance rate ({acceptance_rate:.1%}) too low. Reducing tokens may improve overall throughput."
                ))
            
            if draft_latency > 15:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"speculative_model": "ngram"},  # Suggest n-gram instead
                    expected_improvement="-50% draft latency by using n-gram instead of model-based",
                    confidence=0.5,
                    rationale="Current draft model too slow. N-gram speculation has lower overhead."
                ))
        else:
            # Not enabled - candidate for enabling
            findings["speculative_enabled"] = False
            
            tpot_p99 = benchmark_metrics.get("tpot_p99", 0)
            if isinstance(tpot_p99, dict):
                tpot_p99 = tpot_p99.get("p99", 0)
            
            if tpot_p99 > 50:
                suggestions.append(ExperimentSuggestion(
                    config_changes={
                        "num_speculative_tokens": 5,
                        "speculative_model": "ngram"
                    },
                    expected_improvement="-20-30% TPOT for long sequences",
                    confidence=0.5,
                    rationale=f"High TPOT ({tpot_p99:.1f}ms) suggests decode bottleneck. N-gram speculative decoding can reduce latency."
                ))
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.5 if suggestions else 0.0,
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_reason=relevance_reason
        )
