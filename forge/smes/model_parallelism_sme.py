"""Model Parallelism SME - Tensor, Pipeline, and PD Disaggregation."""

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from typing import Dict, Any, Optional


class ModelParallelismSME(BaseSME):
    """
    Model Parallelism Expert
    
    Optimizes:
    - Tensor Parallelism (TP): Split layers across GPUs
    - Pipeline Parallelism (PP): Split stages across GPUs  
    - Disaggregated Prefill-Decode (PD): Separate GPUs for prefill vs decode
    
    Target: Scale to models/latency requirements that exceed single GPU capacity.
    """
    
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Only register if we have multiple GPUs.
        Single GPU setups cannot use TP/PP/PD disaggregation.
        """
        gpu_count = platform_info.get("gpu_count", 1)
        platform_type = platform_info.get("type", "")
        
        if gpu_count <= 1:
            # Log why we're not registering
            print(f"[ModelParallelismSME] Skipping registration: only {gpu_count} GPU(s) detected. "
                  f"Need 2+ GPUs for tensor/pipeline parallelism or PD disaggregation.")
            return None
        
        if platform_type == "nvidia_cuda":
            return RegistrationInfo(
                sme_id="model_parallelism",
                triggers=[
                    "multi_gpu",
                    "model_too_large_for_single_gpu",
                    "high_inter_gpu_traffic",
                    "pipeline_bottle_neck",
                    "pd_disaggregation_candidate",
                    "tp_pp_imbalance",
                ],
                data_requirements=[
                    # REQUIRED: NCU for inter-GPU communication analysis
                    DataRequirement(
                        data_type="ncu_report",
                        required=True,
                        extractors=[
                            "all_reduce_time_ms",           # TP communication overhead
                            "all_reduce_bytes_transferred",
                            "peer_to_peer_bandwidth_gbps",
                            "nvlink_utilization_percent",
                        ]
                    ),
                    # REQUIRED: NSys for timeline and pipeline stalls
                    DataRequirement(
                        data_type="nsys_report",
                        required=True,
                        extractors=[
                            "inter_gpu_transfer_mb",
                            "pipeline_bubble_percent",      # Idle time from PP imbalance
                            "tensor_parallel_efficiency",
                            "gpu_idle_waiting_for_others_ms",
                            "stage_imbalance_ratio",        # PP: longest stage / shortest stage
                        ]
                    ),
                    # REQUIRED: vLLM logs for parallel config
                    DataRequirement(
                        data_type="vllm_logs",
                        required=True,
                        extractors=[
                            "tensor_parallel_size",
                            "pipeline_parallel_size",
                            "gpu_memory_per_device_mb",
                            "distributed_init_method",
                            "num_layers_per_gpu",
                        ]
                    ),
                    # REQUIRED: Benchmark metrics per GPU (if available)
                    DataRequirement(
                        data_type="benchmark_metrics",
                        required=True,
                        extractors=[
                            "ttft_p99",                     # PD: prefill latency
                            "tpot_p99",                     # PD: decode latency
                            "throughput_rps",
                        ]
                    ),
                ]
            )
        
        elif platform_type == "amd_rocm":
            return RegistrationInfo(
                sme_id="model_parallelism",
                triggers=["multi_gpu", "high_inter_gpu_traffic"],
                data_requirements=[
                    DataRequirement(
                        data_type="rocprof_report",
                        required=True,
                        extractors=[
                            "inter_gpu_transfer_mb",
                            "collective_op_time_ms",
                        ]
                    ),
                    DataRequirement(
                        data_type="vllm_logs",
                        required=True,
                        extractors=[
                            "tensor_parallel_size",
                            "pipeline_parallel_size",
                        ]
                    ),
                ]
            )
        
        return None
    
    async def analyze(self, profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze model parallelism efficiency using multi-LLM consensus.
        
        Key signals:
        - High all-reduce time: TP overhead too high
        - Pipeline bubbles > 20%: stage imbalance
        - High prefill latency + available GPUs: PD disaggregation candidate
        - Low NVLink utilization: PCIe bottleneck
        """
        # TODO: Implement AI brain with multi-LLM consensus
        findings = {}
        suggestions = []
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.0
        )
