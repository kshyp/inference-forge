"""Memory Management SME - KV Cache Management, Prefix Caching, and Memory Pool Optimization."""

from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo


class MemoryManagementSME(BaseSME):
    """
    Memory Management Expert
    
    Optimizes:
    - Prefix Caching: Cache KV for common prompt prefixes (RAG, system prompts)
    - KV Cache Allocation: Block size, memory pool sizing
    - Memory Pool Management: gpu_memory_utilization, swap space
    - Eviction Policies: Preemption mode, swap vs recompute
    - Fragmentation: Block size tuning for memory efficiency
    
    Target: Maximize memory efficiency and throughput through smart caching
    and allocation strategies (distinct from quantization which reduces precision).
    
    Self-Relevance Detection:
    This SME scans data to determine if memory management is relevant:
    - High KV cache usage (> 70%)
    - Repeated prompt patterns (prefix caching opportunity)
    - Memory fragmentation indicators
    - OOM or near-OOM conditions
    - High swap usage (if enabled)
    - Long context workloads (prefix caching beneficial)
    """

    # Thresholds for self-relevance detection
    RELEVANCE_THRESHOLDS = {
        "kv_cache_usage_high": 70.0,  # %
        "kv_cache_usage_critical": 85.0,  # % - urgent attention needed
        "memory_fragmentation_high": 30.0,  # % - wasted memory due to fragmentation
        "swap_usage_high": 1000,  # MB - significant swapping happening
        "oom_warnings": 1,  # Any OOM warnings
        "prefix_similarity": 0.8,  # 80% of prompts share prefix
        "long_context_tokens": 2000,  # avg prompt tokens - prefix caching helps
        "repeated_sequences": 0.3,  # 30% of sequences have shared prefix
    }

    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Memory management is relevant on all platforms.
        """
        platform_type = platform_info.get("type", "")
        
        return RegistrationInfo(
            sme_id="memory_management",
            description="Memory management expert - analyzes KV cache efficiency, "
                       "prefix caching opportunities, memory pool tuning, and fragmentation",
            data_requirements=[
                # REQUIRED: vLLM logs for memory config and metrics
                DataRequirement(
                    data_type="vllm_logs",
                    required=True,
                    extractors=[
                        # Memory configuration
                        "gpu_memory_utilization",
                        "max_model_len",
                        "max_num_seqs",
                        "block_size",
                        "enable_prefix_caching",
                        "prefix_caching_hash_algo",
                        "swap_space_gb",
                        "preemption_mode",  # swap or recompute
                        # Memory breakdown
                        "model_weights_memory_mb",
                        "kv_cache_memory_mb",
                        "kv_cache_max_memory_mb",
                        "activation_memory_mb",
                        "gpu_memory_peak_mb",
                        # Runtime metrics
                        "kv_cache_usage_percent",
                        "num_blocks_total",
                        "num_blocks_allocated",
                        "num_blocks_free",
                        "num_blocks_swapped",
                        "memory_fragmentation_percent",
                        # Workload patterns
                        "avg_prompt_tokens",
                        "avg_output_tokens",
                        "prefix_hit_rate",  # if prefix caching enabled
                        "unique_prefixes",
                        "oom_warnings",
                        "oom_errors",
                    ]
                ),
                # REQUIRED: System metrics
                DataRequirement(
                    data_type="system_metrics_report",
                    required=True,
                    extractors=[
                        "gpu_memory_utilization_percent",
                        "memory_used_mb",
                        "memory_total_mb",
                        "memory_free_mb",
                    ]
                ),
                # OPTIONAL: Detailed swap metrics
                DataRequirement(
                    data_type="swap_metrics",
                    required=False,
                    extractors=[
                        "swap_in_blocks_per_sec",
                        "swap_out_blocks_per_sec",
                        "swap_latency_ms",
                        "swap_space_used_gb",
                    ]
                ),
                # OPTIONAL: Prefix analysis (computed from request logs)
                DataRequirement(
                    data_type="prefix_analysis",
                    required=False,
                    extractors=[
                        "common_prefixes",
                        "prefix_token_lengths",
                        "prefix_hit_rate_if_enabled",
                        "prefix_reuse_ratio",
                    ]
                ),
                # REQUIRED: Benchmark metrics
                DataRequirement(
                    data_type="benchmark_metrics",
                    required=True,
                    extractors=[
                        "throughput_rps",
                        "ttft_p99",
                        "tpot_p99",
                        "error_rate",
                    ]
                ),
            ]
        )
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if memory management is relevant.
        
        Memory management is relevant when:
        1. KV cache usage > 70% (tuning needed)
        2. KV cache usage > 85% (urgent - risk of OOM)
        3. High memory fragmentation (> 30%)
        4. Any OOM warnings or errors
        5. High swap usage (indicates memory pressure)
        6. Long context workloads (> 2000 tokens avg)
        7. Repeated prompt patterns (prefix caching opportunity)
        8. Prefix caching disabled but high reuse potential
        
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
        system_metrics = profiling_data.get("system_metrics_report", {})
        swap_metrics = profiling_data.get("swap_metrics", {})
        prefix_analysis = profiling_data.get("prefix_analysis", {})
        
        # Check 1: KV cache usage
        kv_usage = vllm_metrics.get("kv_cache_usage_percent", 0)
        if kv_usage > self.RELEVANCE_THRESHOLDS["kv_cache_usage_critical"]:
            relevance_signals.append(f"kv_cache_critical ({kv_usage:.1f}%)")
            relevance_score += 0.4
        elif kv_usage > self.RELEVANCE_THRESHOLDS["kv_cache_usage_high"]:
            relevance_signals.append(f"kv_cache_high ({kv_usage:.1f}%)")
            relevance_score += 0.25
        
        # Check 2: Memory fragmentation
        fragmentation = vllm_metrics.get("memory_fragmentation_percent", 0)
        if fragmentation > self.RELEVANCE_THRESHOLDS["memory_fragmentation_high"]:
            relevance_signals.append(f"high_fragmentation ({fragmentation:.1f}%)")
            relevance_score += 0.2
        
        # Check 3: OOM conditions
        oom_warnings = vllm_metrics.get("oom_warnings", 0)
        oom_errors = vllm_metrics.get("oom_errors", 0)
        if oom_errors > 0:
            relevance_signals.append(f"oom_errors ({oom_errors})")
            relevance_score += 0.5  # Critical
        elif oom_warnings > 0:
            relevance_signals.append(f"oom_warnings ({oom_warnings})")
            relevance_score += 0.3
        
        # Check 4: Swap usage
        swap_used = swap_metrics.get("swap_space_used_gb", 0)
        if swap_used > 0:
            relevance_signals.append(f"swap_in_use ({swap_used:.1f}GB)")
            relevance_score += 0.15
        
        # Check 5: Prefix caching opportunity
        prefix_caching_enabled = vllm_config.get("enable_prefix_caching", False)
        avg_prompt_tokens = vllm_metrics.get("avg_prompt_tokens", 0)
        
        if not prefix_caching_enabled:
            # Check if it would be beneficial
            prefix_reuse = prefix_analysis.get("prefix_reuse_ratio", 0)
            unique_prefixes = prefix_analysis.get("unique_prefixes", 0)
            
            if avg_prompt_tokens > self.RELEVANCE_THRESHOLDS["long_context_tokens"]:
                relevance_signals.append(f"long_context ({avg_prompt_tokens:.0f} avg tokens)")
                relevance_score += 0.1
            
            if prefix_reuse > self.RELEVANCE_THRESHOLDS["repeated_sequences"]:
                relevance_signals.append(f"prefix_reuse_opportunity ({prefix_reuse:.1%} reuse)")
                relevance_score += 0.2
            
            if unique_prefixes > 0 and unique_prefixes < 10:
                relevance_signals.append(f"few_unique_prefixes ({unique_prefixes})")
                relevance_score += 0.15
        else:
            # Prefix caching enabled - check if it's working
            hit_rate = vllm_metrics.get("prefix_hit_rate", 0)
            if hit_rate < 0.3:
                relevance_signals.append(f"low_prefix_hit_rate ({hit_rate:.1%})")
                relevance_score += 0.1
        
        # Check 6: Block size optimization
        block_size = vllm_config.get("block_size", 16)
        num_blocks = vllm_metrics.get("num_blocks_total", 0)
        
        if num_blocks > 0:
            # Larger blocks for long sequences, smaller for short
            if avg_prompt_tokens > 4000 and block_size < 32:
                relevance_signals.append(f"small_blocks_for_long_context ({block_size}, {avg_prompt_tokens:.0f} tokens)")
                relevance_score += 0.1
            elif avg_prompt_tokens < 500 and block_size > 16:
                relevance_signals.append(f"large_blocks_for_short_context ({block_size}, {avg_prompt_tokens:.0f} tokens)")
                relevance_score += 0.1
        
        # Check 7: Memory pool sizing
        gpu_mem_util_setting = vllm_config.get("gpu_memory_utilization", 0.9)
        actual_mem_util = system_metrics.get("gpu_memory_utilization_percent", 0)
        
        if actual_mem_util < 70 and gpu_mem_util_setting > 0.85:
            relevance_signals.append(f"memory_pool_overallocated ({gpu_mem_util_setting:.0%} setting, {actual_mem_util:.0%} used)")
            relevance_score += 0.1
        
        # Determine relevance
        is_relevant = relevance_score >= 0.25
        
        if is_relevant:
            reason = f"Memory management relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = (f"Memory management not relevant (KV={kv_usage:.1f}%, "
                         f"fragmentation={fragmentation:.1f}%, prefix_caching={prefix_caching_enabled})")
        
        return is_relevant, min(relevance_score, 1.0), reason
    
    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze memory management opportunities.
        
        Key signals:
        - High KV cache usage → Tune block size or increase memory pool
        - Memory fragmentation → Adjust block size
        - OOM warnings → Reduce max_num_seqs or enable swap
        - Long repeated prefixes → Enable prefix caching
        - High swap usage → Tune swap space or reduce batch size
        
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
                    "primary_bottleneck": "not_memory_management_related",
                    "memory_management_candidate": False,
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
        vllm_metrics = vllm_logs.get("metrics", {})
        swap_metrics = profiling_data.get("swap_metrics", {})
        prefix_analysis = profiling_data.get("prefix_analysis", {})
        
        # Current configuration
        kv_usage = vllm_metrics.get("kv_cache_usage_percent", 0)
        block_size = vllm_config.get("block_size", 16)
        prefix_caching_enabled = vllm_config.get("enable_prefix_caching", False)
        swap_space = vllm_config.get("swap_space_gb", 0)
        preemption_mode = vllm_config.get("preemption_mode", "recompute")
        gpu_mem_util = vllm_config.get("gpu_memory_utilization", 0.9)
        max_num_seqs = vllm_config.get("max_num_seqs", 256)
        
        findings["kv_cache_usage_percent"] = kv_usage
        findings["block_size"] = block_size
        findings["prefix_caching_enabled"] = prefix_caching_enabled
        findings["swap_space_gb"] = swap_space
        
        # Analysis 1: KV Cache Pressure
        if kv_usage > 85:
            findings["primary_bottleneck"] = "kv_cache_pressure"
            
            # Suggest reducing max sequences first
            if max_num_seqs > 128:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"max_num_seqs": max(64, max_num_seqs // 2)},
                    expected_improvement="Reduce memory pressure by limiting concurrent sequences",
                    confidence=0.6,
                    rationale=f"KV cache at {kv_usage:.1f}% - reducing max_num_seqs from {max_num_seqs} to relieve pressure."
                ))
            
            # Suggest swap space if not enabled
            if swap_space == 0:
                suggestions.append(ExperimentSuggestion(
                    config_changes={
                        "swap_space_gb": 4,
                        "preemption_mode": "swap"
                    },
                    expected_improvement="Enable swapping to handle overflow without OOM",
                    confidence=0.5,
                    rationale=f"KV cache critical at {kv_usage:.1f}%. Swap space prevents OOM by offloading to CPU."
                ))
        
        # Analysis 2: Prefix Caching
        avg_prompt_tokens = vllm_metrics.get("avg_prompt_tokens", 0)
        prefix_reuse = prefix_analysis.get("prefix_reuse_ratio", 0)
        unique_prefixes = prefix_analysis.get("unique_prefixes", 0)
        
        if not prefix_caching_enabled:
            # Check if prefix caching would help
            should_enable = False
            reason_parts = []
            
            if prefix_reuse > 0.3:
                should_enable = True
                reason_parts.append(f"{prefix_reuse:.0%} prefix reuse detected")
            
            if unique_prefixes > 0 and unique_prefixes < 10 and avg_prompt_tokens > 1000:
                should_enable = True
                reason_parts.append(f"only {unique_prefixes} unique prefixes with long contexts")
            
            if avg_prompt_tokens > 2000:
                should_enable = True
                reason_parts.append(f"long context workload ({avg_prompt_tokens:.0f} tokens)")
            
            if should_enable:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"enable_prefix_caching": True},
                    expected_improvement=f"-20-50% TTFT for repeated prefixes, -{min(30, int(prefix_reuse * 40))}% KV memory",
                    confidence=0.65,
                    rationale=f"Enable prefix caching: {'; '.join(reason_parts)}. Avoids recomputing shared prompt prefixes."
                ))
        else:
            # Prefix caching enabled - check if working well
            hit_rate = vllm_metrics.get("prefix_hit_rate", 0)
            
            if hit_rate < 0.2:
                findings["prefix_caching_hit_rate"] = hit_rate
                # Might not be beneficial - workload doesn't have shared prefixes
                # Or hash algorithm needs tuning
                suggestions.append(ExperimentSuggestion(
                    config_changes={"enable_prefix_caching": False},
                    expected_improvement="Remove prefix caching overhead (low hit rate)",
                    confidence=0.3,
                    rationale=f"Prefix caching hit rate only {hit_rate:.1%} - overhead not justified for this workload."
                ))
        
        # Analysis 3: Block Size Optimization
        fragmentation = vllm_metrics.get("memory_fragmentation_percent", 0)
        num_blocks = vllm_metrics.get("num_blocks_total", 0)
        
        if fragmentation > 30:
            findings["primary_bottleneck"] = "memory_fragmentation"
            findings["fragmentation_percent"] = fragmentation
            
            # Recommend block size change
            if avg_prompt_tokens > 4000 and block_size < 32:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"block_size": 32},
                    expected_improvement=f"-{fragmentation/2:.0f}% fragmentation for long context workloads",
                    confidence=0.5,
                    rationale=f"High fragmentation ({fragmentation:.1f}%) with {avg_prompt_tokens:.0f} avg tokens. Larger blocks (32) reduce overhead."
                ))
            elif avg_prompt_tokens < 1000 and block_size > 16:
                suggestions.append(ExperimentSuggestion(
                    config_changes={"block_size": 16},
                    expected_improvement=f"-{fragmentation/2:.0f}% fragmentation for short context workloads",
                    confidence=0.5,
                    rationale=f"High fragmentation ({fragmentation:.1f}%) with {avg_prompt_tokens:.0f} avg tokens. Smaller blocks (16) improve granularity."
                ))
        
        # Analysis 4: Swap Tuning
        swap_used = swap_metrics.get("swap_space_used_gb", 0)
        swap_latency = swap_metrics.get("swap_latency_ms", 0)
        
        if swap_used > 0:
            findings["swap_in_use"] = True
            findings["swap_used_gb"] = swap_used
            findings["swap_latency_ms"] = swap_latency
            
            if swap_latency > 50:
                # Swap is causing latency issues
                if preemption_mode == "swap":
                    suggestions.append(ExperimentSuggestion(
                        config_changes={"preemption_mode": "recompute"},
                        expected_improvement=f"-{swap_latency/2:.0f}ms latency by recomputing instead of swapping",
                        confidence=0.4,
                        rationale=f"Swap latency high ({swap_latency:.1f}ms). Recompute may be faster for this workload."
                    ))
            
            if swap_space > 0 and swap_used > swap_space * 0.9:
                # Swap space nearly exhausted
                suggestions.append(ExperimentSuggestion(
                    config_changes={"swap_space_gb": swap_space + 4},
                    expected_improvement="Prevent swap exhaustion with larger swap space",
                    confidence=0.5,
                    rationale=f"Swap space nearly full ({swap_used:.1f}/{swap_space:.1f}GB). Increase to handle peak loads."
                ))
        
        # Analysis 5: Memory Pool Sizing
        system_mem_util = vllm_logs.get("system_metrics_report", {}).get("gpu_memory_utilization_percent", 0)
        
        if system_mem_util < 60 and gpu_mem_util > 0.85:
            # Over-allocated memory pool
            suggestions.append(ExperimentSuggestion(
                config_changes={"gpu_memory_utilization": 0.75},
                expected_improvement="Reduce memory waste from over-allocation",
                confidence=0.4,
                rationale=f"Memory pool allocated at {gpu_mem_util:.0%} but only {system_mem_util:.0%} used. Reduce to 75%."
            ))
        elif system_mem_util > 95 and gpu_mem_util < 0.95:
            # Under-allocated, hitting limit
            suggestions.append(ExperimentSuggestion(
                config_changes={"gpu_memory_utilization": min(0.95, gpu_mem_util + 0.05)},
                expected_improvement="Allow more GPU memory for KV cache",
                confidence=0.4,
                rationale=f"Memory fully utilized ({system_mem_util:.0f}%). Increase memory pool to prevent OOM."
            ))
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.5 if suggestions else 0.0,
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_reason=relevance_reason
        )
