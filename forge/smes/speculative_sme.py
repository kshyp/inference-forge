"""Speculative Decoding SME - Optimizes draft model speculation."""

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from typing import Dict, Any, Optional


class SpeculativeSME(BaseSME):
    """
    Speculative Decoding Expert
    
    Optimizes:
    - Draft model selection (n-gram, custom draft model)
    - Number of speculative tokens
    - Acceptance rate tuning
    - Verification overhead
    
    Target: Reduce decode phase latency by predicting future tokens.
    """
    
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
            triggers=[
                "slow_decode_phase",
                "low_acceptance_rate",
                "high_draft_overhead",
                "decode_memory_available",
                "tpot_high",
            ],
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
                # REQUIRED: NCU for draft vs target model efficiency
                DataRequirement(
                    data_type="ncu_report",
                    required=True,
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
    
    async def analyze(self, profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze speculative decoding efficiency using multi-LLM consensus.
        
        Key signals:
        - Low acceptance rate (< 60%): draft model too ambitious or mismatched
        - High draft latency: draft model too large
        - High TPOT without speculation: candidate for enabling
        - Low GPU memory: can't fit draft model
        """
        # TODO: Implement AI brain with multi-LLM consensus
        findings = {}
        suggestions = []
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.0
        )
