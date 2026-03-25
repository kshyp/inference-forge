"""Model Parallelism SME - Tensor, Pipeline, and PD Disaggregation."""

from pathlib import Path
from typing import Dict, Any, Optional, List

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo


class ModelParallelismSME(BaseSME):
    """
    Model Parallelism Expert
    
    Optimizes:
    - Tensor Parallelism (TP): Split layers across GPUs
    - Pipeline Parallelism (PP): Split stages across GPUs  
    - Disaggregated Prefill-Decode (PD): Separate GPUs for prefill vs decode
    
    Target: Scale to models/latency requirements that exceed single GPU capacity.
    
    Self-Relevance Detection:
    This SME scans data to determine if model parallelism is relevant:
    - Multiple GPUs available
    - High inter-GPU traffic (indicating TP is active but inefficient)
    - Pipeline bubbles (PP imbalance)
    - Model too large for single GPU
    - High TTFT (prefill) or TPOT (decode) with spare GPUs (PD candidate)
    - PD disaggregation already enabled but suboptimal
    """

    # Thresholds for self-relevance detection
    RELEVANCE_THRESHOLDS = {
        "inter_gpu_traffic_high": 100,  # MB - high traffic suggests TP issues
        "pipeline_bubble_high": 20.0,  # % - idle time from PP imbalance
        "tp_efficiency_low": 80.0,  # % - tensor parallelism efficiency
        "stage_imbalance_high": 2.0,  # ratio - longest/shortest stage
        "ttft_high": 500,  # ms - high prefill latency = PD candidate
        "tpot_high": 100,  # ms - high decode latency = PD candidate
        "pd_prefill_gpu_util_high": 90.0,  # % - prefill GPUs saturated
        "pd_decode_gpu_util_low": 50.0,  # % - decode GPUs underutilized
        "pd_latency_imbalance": 3.0,  # ratio - prefill/decode latency ratio
    }

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
                description="Model parallelism expert - analyzes multi-GPU setups for "
                           "tensor parallelism, pipeline parallelism, and prefill-decode disaggregation",
                data_requirements=[
                    # OPTIONAL: NCU for inter-GPU communication analysis
                    DataRequirement(
                        data_type="ncu_report",
                        required=False,
                        extractors=[
                            "all_reduce_time_ms",           # TP communication overhead
                            "all_reduce_bytes_transferred",
                            "peer_to_peer_bandwidth_gbps",
                            "nvlink_utilization_percent",
                        ]
                    ),
                    # OPTIONAL: NSys for timeline and pipeline stalls
                    DataRequirement(
                        data_type="nsys_report",
                        required=False,
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
                            # PD disaggregation config
                            "enable_pd_disaggregation",
                            "pd_ratio_prefill_decode",
                            "num_prefill_instances",
                            "num_decode_instances",
                            "pd_scheduler_policy",
                            "pd_transfer_backend",  # nccl, gloo, etc.
                        ]
                    ),
                    # REQUIRED: Per-GPU metrics for PD analysis
                    DataRequirement(
                        data_type="per_gpu_metrics",
                        required=False,
                        extractors=[
                            "gpu_utilization_by_role",  # prefill vs decode
                            "memory_by_role",
                            "queue_depth_by_role",
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
                            # PD-specific metrics
                            "prefill_queue_time_ms",
                            "decode_queue_time_ms",
                            "transfer_overhead_ms",         # KV cache transfer time
                        ]
                    ),
                    # REQUIRED: System metrics for all GPUs
                    DataRequirement(
                        data_type="system_metrics_report",
                        required=True,
                        extractors=[
                            "gpu_utilization_percent",
                            "memory_used_mb",
                            "memory_total_mb",
                        ]
                    ),
                ]
            )
        
        elif platform_type == "amd_rocm":
            return RegistrationInfo(
                sme_id="model_parallelism",
                description="Model parallelism expert for AMD ROCm multi-GPU setups",
                data_requirements=[
                    DataRequirement(
                        data_type="rocprof_report",
                        required=False,
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
                            "enable_pd_disaggregation",
                            "pd_ratio_prefill_decode",
                        ]
                    ),
                    DataRequirement(
                        data_type="benchmark_metrics",
                        required=True,
                        extractors=[
                            "ttft_p99",
                            "tpot_p99",
                            "throughput_rps",
                        ]
                    ),
                ]
            )
        
        return None
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if model parallelism optimization is relevant.
        
        Model parallelism is relevant when:
        1. Multiple GPUs available AND high inter-GPU traffic (TP optimization needed)
        2. Pipeline bubbles > 20% (PP imbalance)
        3. TP efficiency < 80%
        4. High TTFT with spare GPUs (PD disaggregation candidate)
        5. High TPOT with spare GPUs (PD disaggregation candidate)
        6. PD disaggregation already enabled but imbalanced
        7. High prefill latency but low decode latency (classic PD candidate)
        
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
        nsys_report = profiling_data.get("nsys_report", {})
        ncu_report = profiling_data.get("ncu_report", {})
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        system_metrics = profiling_data.get("system_metrics_report", {})
        per_gpu = profiling_data.get("per_gpu_metrics", {})
        
        # Get GPU count from system or config
        gpu_count = vllm_config.get("tensor_parallel_size", 1) * vllm_config.get("pipeline_parallel_size", 1)
        if gpu_count <= 1:
            return False, 0.0, f"Single GPU setup (TP={vllm_config.get('tensor_parallel_size', 1)}, PP={vllm_config.get('pipeline_parallel_size', 1)})"
        
        # Base relevance for multi-GPU
        relevance_score += 0.15
        
        # Check 1: Current TP/PP configuration
        tp_size = vllm_config.get("tensor_parallel_size", 1)
        pp_size = vllm_config.get("pipeline_parallel_size", 1)
        pd_enabled = vllm_config.get("enable_pd_disaggregation", False)
        
        if tp_size > 1 and not pd_enabled:
            # TP is active - check efficiency
            tp_efficiency = nsys_report.get("tensor_parallel_efficiency", 100)
            if tp_efficiency < self.RELEVANCE_THRESHOLDS["tp_efficiency_low"]:
                relevance_signals.append(f"low_tp_efficiency ({tp_efficiency:.1f}%)")
                relevance_score += 0.25
            
            inter_gpu_mb = nsys_report.get("inter_gpu_transfer_mb", 0)
            if inter_gpu_mb > self.RELEVANCE_THRESHOLDS["inter_gpu_traffic_high"]:
                relevance_signals.append(f"high_inter_gpu_traffic ({inter_gpu_mb}MB)")
                relevance_score += 0.15
            
            # All-reduce overhead
            all_reduce_time = ncu_report.get("all_reduce_time_ms", 0)
            if all_reduce_time > 5:  # More than 5ms is significant
                relevance_signals.append(f"high_all_reduce ({all_reduce_time:.1f}ms)")
                relevance_score += 0.15
        
        if pp_size > 1 and not pd_enabled:
            # PP is active - check for bubbles
            bubble_pct = nsys_report.get("pipeline_bubble_percent", 0)
            if bubble_pct > self.RELEVANCE_THRESHOLDS["pipeline_bubble_high"]:
                relevance_signals.append(f"pipeline_bubbles ({bubble_pct:.1f}%)")
                relevance_score += 0.25
            
            stage_imbalance = nsys_report.get("stage_imbalance_ratio", 1.0)
            if stage_imbalance > self.RELEVANCE_THRESHOLDS["stage_imbalance_high"]:
                relevance_signals.append(f"stage_imbalance ({stage_imbalance:.1f}x)")
                relevance_score += 0.15
        
        # Check 2: PD disaggregation analysis
        if pd_enabled:
            # PD is already enabled - check if it's working well
            prefill_queue = benchmark_metrics.get("prefill_queue_time_ms", 0)
            decode_queue = benchmark_metrics.get("decode_queue_time_ms", 0)
            transfer_overhead = benchmark_metrics.get("transfer_overhead_ms", 0)
            
            if transfer_overhead > 20:
                relevance_signals.append(f"high_pd_transfer_overhead ({transfer_overhead:.1f}ms)")
                relevance_score += 0.2
            
            # Check GPU utilization imbalance
            gpu_by_role = per_gpu.get("gpu_utilization_by_role", {})
            prefill_util = gpu_by_role.get("prefill", 0)
            decode_util = gpu_by_role.get("decode", 0)
            
            if prefill_util > self.RELEVANCE_THRESHOLDS["pd_prefill_gpu_util_high"]:
                relevance_signals.append(f"prefill_gpu_saturated ({prefill_util:.1f}%)")
                relevance_score += 0.15
            
            if decode_util < self.RELEVANCE_THRESHOLDS["pd_decode_gpu_util_low"]:
                relevance_signals.append(f"decode_gpu_underutil ({decode_util:.1f}%)")
                relevance_score += 0.15
            
            if relevance_signals:
                relevance_signals.append("pd_tuning_needed")
        else:
            # PD not enabled - check if it's a candidate
            ttft_p99 = benchmark_metrics.get("ttft_p99", 0)
            tpot_p99 = benchmark_metrics.get("tpot_p99", 0)
            
            if isinstance(ttft_p99, dict):
                ttft_p99 = ttft_p99.get("p99", 0)
            if isinstance(tpot_p99, dict):
                tpot_p99 = tpot_p99.get("p99", 0)
            
            # Classic PD candidate: high TTFT but low TPOT (prefill bottleneck)
            if ttft_p99 > self.RELEVANCE_THRESHOLDS["ttft_high"]:
                if tpot_p99 < self.RELEVANCE_THRESHOLDS["tpot_high"]:
                    relevance_signals.append(f"pd_candidate_prefill_bottleneck (TTFT={ttft_p99:.0f}ms, TPOT={tpot_p99:.0f}ms)")
                    relevance_score += 0.35
                else:
                    relevance_signals.append(f"high_ttft ({ttft_p99:.0f}ms)")
                    relevance_score += 0.15
            
            # Decode bottleneck case
            if tpot_p99 > self.RELEVANCE_THRESHOLDS["tpot_high"]:
                if ttft_p99 < self.RELEVANCE_THRESHOLDS["ttft_high"]:
                    relevance_signals.append(f"pd_candidate_decode_bottleneck (TPOT={tpot_p99:.0f}ms, TTFT={ttft_p99:.0f}ms)")
                    relevance_score += 0.25
        
        # Check 3: Memory pressure on single GPU (might need TP even if not configured)
        if tp_size == 1 and pp_size == 1 and not pd_enabled:
            mem_util = system_metrics.get("memory_utilization_percent", 0)
            if mem_util > 90:
                relevance_signals.append(f"high_memory_single_gpu ({mem_util:.1f}%)")
                relevance_score += 0.2
        
        # Determine relevance
        is_relevant = relevance_score >= 0.3
        
        if is_relevant:
            reason = f"Model parallelism relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = (f"Model parallelism not relevant (TP={tp_size}, PP={pp_size}, "
                         f"PD={pd_enabled}, no efficiency issues detected)")
        
        return is_relevant, min(relevance_score, 1.0), reason
    
    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze model parallelism efficiency using multi-LLM consensus.
        
        Key signals:
        - High all-reduce time: TP overhead too high
        - Pipeline bubbles > 20%: stage imbalance
        - High prefill latency + available GPUs: PD disaggregation candidate
        - Low NVLink utilization: PCIe bottleneck
        - PD enabled but imbalanced: tune PD ratio
        
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
                    "primary_bottleneck": "not_model_parallelism_related",
                    "model_parallelism_candidate": False,
                },
                suggestions=[],
                confidence=0.0,
                is_relevant=False,
                relevance_score=relevance_score,
                relevance_reason=relevance_reason
            )
        
        suggestions = []
        findings = {}
        
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        nsys_report = profiling_data.get("nsys_report", {})
        ncu_report = profiling_data.get("ncu_report", {})
        system_metrics = profiling_data.get("system_metrics_report", {})
        
        tp_size = vllm_config.get("tensor_parallel_size", 1)
        pp_size = vllm_config.get("pipeline_parallel_size", 1)
        pd_enabled = vllm_config.get("enable_pd_disaggregation", False)
        
        findings["current_tp_size"] = tp_size
        findings["current_pp_size"] = pp_size
        findings["pd_enabled"] = pd_enabled
        
        # Analyze TP issues
        if tp_size > 1 and not pd_enabled:
            tp_efficiency = nsys_report.get("tensor_parallel_efficiency", 100)
            all_reduce_time = ncu_report.get("all_reduce_time_ms", 0)
            
            if tp_efficiency < 80 or all_reduce_time > 5:
                findings["primary_bottleneck"] = "tensor_parallel_overhead"
                findings["tp_efficiency"] = tp_efficiency
                findings["all_reduce_time_ms"] = all_reduce_time
                
                # Suggest reducing TP if overhead is too high
                if tp_size > 2:
                    suggestions.append(ExperimentSuggestion(
                        config_changes={
                            "tensor_parallel_size": tp_size - 1,
                            "pipeline_parallel_size": 1
                        },
                        expected_improvement=f"+{100-tp_efficiency:.0f}% efficiency by reducing TP overhead",
                        confidence=0.5,
                        rationale=f"TP efficiency ({tp_efficiency:.1f}%) low with {all_reduce_time:.1f}ms all-reduce. Reducing TP may help."
                    ))
        
        # Analyze PP issues
        if pp_size > 1 and not pd_enabled:
            bubble_pct = nsys_report.get("pipeline_bubble_percent", 0)
            stage_imbalance = nsys_report.get("stage_imbalance_ratio", 1.0)
            
            if bubble_pct > 20:
                findings["primary_bottleneck"] = "pipeline_imbalance"
                findings["pipeline_bubble_percent"] = bubble_pct
                findings["stage_imbalance_ratio"] = stage_imbalance
                
                suggestions.append(ExperimentSuggestion(
                    config_changes={"num_scheduler_steps": 10},
                    expected_improvement=f"-{bubble_pct/2:.0f}% pipeline bubbles via more micro-batches",
                    confidence=0.5,
                    rationale=f"Pipeline bubbles ({bubble_pct:.1f}%) indicate stage imbalance. More scheduler steps may help."
                ))
        
        # Analyze PD disaggregation
        if pd_enabled:
            # PD already enabled - check for tuning opportunities
            per_gpu = profiling_data.get("per_gpu_metrics", {})
            gpu_by_role = per_gpu.get("gpu_utilization_by_role", {})
            prefill_util = gpu_by_role.get("prefill", 0)
            decode_util = gpu_by_role.get("decode", 0)
            
            current_ratio = vllm_config.get("pd_ratio_prefill_decode", "1:1")
            
            findings["current_pd_ratio"] = current_ratio
            findings["prefill_gpu_util"] = prefill_util
            findings["decode_gpu_util"] = decode_util
            
            # Check if ratio needs adjustment
            if prefill_util > 90 and decode_util < 50:
                # Need more prefill GPUs
                suggestions.append(ExperimentSuggestion(
                    config_changes={"pd_ratio_prefill_decode": "2:1"},
                    expected_improvement="-30% TTFT by allocating more GPUs to prefill",
                    confidence=0.6,
                    rationale=f"Prefill GPUs saturated ({prefill_util:.1f}%) while decode underutilized ({decode_util:.1f}%). Rebalance to 2:1."
                ))
            elif decode_util > 90 and prefill_util < 50:
                # Need more decode GPUs
                suggestions.append(ExperimentSuggestion(
                    config_changes={"pd_ratio_prefill_decode": "1:2"},
                    expected_improvement="-25% TPOT by allocating more GPUs to decode",
                    confidence=0.6,
                    rationale=f"Decode GPUs saturated ({decode_util:.1f}%) while prefill underutilized ({prefill_util:.1f}%). Rebalance to 1:2."
                ))
            
            # Check transfer overhead
            transfer_overhead = benchmark_metrics.get("transfer_overhead_ms", 0)
            if transfer_overhead > 20:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"pd_transfer_backend": "nccl"},
                    expected_improvement=f"-{transfer_overhead/2:.0f}ms KV cache transfer via optimized backend",
                    confidence=0.4,
                    rationale=f"High KV cache transfer overhead ({transfer_overhead:.1f}ms). NCCL backend may reduce latency."
                ))
        else:
            # PD not enabled - check if it should be
            ttft_p99 = benchmark_metrics.get("ttft_p99", 0)
            tpot_p99 = benchmark_metrics.get("tpot_p99", 0)
            
            if isinstance(ttft_p99, dict):
                ttft_p99 = ttft_p99.get("p99", 0)
            if isinstance(tpot_p99, dict):
                tpot_p99 = tpot_p99.get("p99", 0)
            
            # Get total GPU count
            total_gpus = tp_size * pp_size
            if total_gpus < 2:
                # Check if we can infer from system
                total_gpus = max(system_metrics.get("gpu_count", 1), 2)
            
            # Recommend PD for prefill bottleneck
            if ttft_p99 > 500 and tpot_p99 < 100 and total_gpus >= 2:
                findings["primary_bottleneck"] = "prefill_compute_bound"
                findings["pd_candidate"] = True
                
                suggestions.append(ExperimentSuggestion(
                    config_changes={
                        "enable_pd_disaggregation": True,
                        "pd_ratio_prefill_decode": "1:1",
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1
                    },
                    expected_improvement="-40% TTFT via dedicated prefill GPUs",
                    confidence=0.65,
                    rationale=f"High TTFT ({ttft_p99:.0f}ms) with low TPOT ({tpot_p99:.0f}ms) indicates prefill bottleneck. PD disaggregation with {total_gpus} GPUs recommended."
                ))
            
            # Recommend PD for decode bottleneck
            elif tpot_p99 > 100 and ttft_p99 < 500 and total_gpus >= 2:
                findings["primary_bottleneck"] = "decode_compute_bound"
                findings["pd_candidate"] = True
                
                suggestions.append(ExperimentSuggestion(
                    config_changes={
                        "enable_pd_disaggregation": True,
                        "pd_ratio_prefill_decode": "1:1",
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1
                    },
                    expected_improvement="-30% TPOT via dedicated decode GPUs",
                    confidence=0.6,
                    rationale=f"High TPOT ({tpot_p99:.0f}ms) with low TTFT ({ttft_p99:.0f}ms) indicates decode bottleneck. PD disaggregation recommended."
                ))
            
            # Memory pressure case - suggest TP
            mem_util = system_metrics.get("memory_utilization_percent", 0)
            if mem_util > 90 and tp_size == 1:
                suggestions.append(ExperimentSuggestion(
                    config_changes={
                        "tensor_parallel_size": min(2, total_gpus),
                        "pipeline_parallel_size": 1
                    },
                    expected_improvement="Distribute model across GPUs to reduce per-GPU memory",
                    confidence=0.5,
                    rationale=f"High memory utilization ({mem_util:.1f}%). Tensor parallelism can distribute model weights."
                ))
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.5 if suggestions else 0.0,
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_reason=relevance_reason
        )
