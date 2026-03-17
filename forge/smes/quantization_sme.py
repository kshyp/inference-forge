"""Quantization SME - Recommends quantization strategies for vLLM."""

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from typing import Dict, Any, Optional


class QuantizationSME(BaseSME):
    """
    Quantization Expert
    
    Analyzes memory bandwidth pressure, model size, and compute patterns
    to recommend optimal quantization strategies (AWQ, GPTQ, FP8, etc.)
    """
    
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Register on CUDA and ROCm platforms.
        Quantization applies to most modern GPUs.
        """
        platform_type = platform_info.get("type", "")
        
        if platform_type == "nvidia_cuda":
            return RegistrationInfo(
                sme_id="quantization",
                triggers=[
                    "gpu_memory_high",
                    "memory_bandwidth_bound",
                    "kv_cache_pressure",
                    "model_size_large",
                    "quantization_candidate",
                    # Baseline exploration triggers
                    "baseline_exploration",
                ],
                data_requirements=[
                    # REQUIRED: NCU for memory bandwidth and compute analysis
                    DataRequirement(
                        data_type="ncu_report",
                        required=True,
                        extractors=[
                            "memory_bandwidth_utilization_percent",
                            "compute_utilization_percent",
                            "dram_throughput_gbps",
                            "l2_cache_hit_rate",
                        ]
                    ),
                    # REQUIRED: vLLM logs for memory breakdown and config
                    DataRequirement(
                        data_type="vllm_logs",
                        required=True,
                        extractors=[
                            "model_name",
                            "model_dtype",
                            "current_quantization",
                            "gpu_memory_utilization_setting",
                            "max_model_len",
                            "max_num_seqs",
                            # Memory breakdown (from vLLM startup logs)
                            "model_weights_memory_mb",
                            "kv_cache_memory_mb",
                            "kv_cache_usage_percent",
                            "activation_memory_mb",
                            "gpu_memory_peak_mb",
                            "oom_warnings",
                            "memory_reservation_warnings",
                        ]
                    ),
                    # OPTIONAL: NSys for memory timeline and pressure
                    DataRequirement(
                        data_type="nsys_report",
                        required=False,
                        extractors=[
                            "gpu_memory_timeline",
                            "cuda_malloc_time_ms",
                        ]
                    ),
                ]
            )
        
        elif platform_type == "amd_rocm":
            return RegistrationInfo(
                sme_id="quantization",
                triggers=[
                    "gpu_memory_high",
                    "memory_bandwidth_bound",
                    "kv_cache_pressure",
                ],
                data_requirements=[
                    DataRequirement(
                        data_type="rocprof_report",
                        required=True,
                        extractors=[
                            "memory_bandwidth_utilization",
                            "compute_utilization",
                        ]
                    ),
                    DataRequirement(
                        data_type="vllm_logs",
                        required=True,
                        extractors=[
                            "model_name",
                            "current_quantization",
                            "model_weights_memory_mb",
                            "kv_cache_usage_percent",
                            "oom_warnings",
                        ]
                    ),
                ]
            )
        
        # Not supported on other platforms (CPU, etc.)
        return None
    
    async def analyze(self, profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze quantization opportunities using multi-LLM consensus.
        
        Key signals:
        - Memory bandwidth bound (quantization reduces memory traffic)
        - High GPU memory usage (quantization reduces model size)
        - Current FP16/BF16 model (quantization candidate)
        - AWQ/GPTQ already applied (may recommend FP8 for further optimization)
        """
        # TODO: Implement AI brain with multi-LLM consensus
        # For now, return empty response
        findings = {}
        suggestions = []
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.0
        )
