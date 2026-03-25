"""CPU Optimizations SME - Optimizes vLLM for CPU inference."""

from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo


class CPUOptimizationsSME(BaseSME):
    """
    CPU Optimizations Expert
    
    Optimizes vLLM for CPU-only inference (no GPU available).
    Focuses on:
    - Thread/core affinity and parallelism
    - Memory bandwidth optimization (NUMA awareness)
    - Quantization strategies for CPU (INT8, INT4)
    - Batch size tuning for CPU cache efficiency
    - Intel/AMD-specific optimizations (MKL, OpenBLAS)
    
    Self-Relevance Detection:
    This SME scans data to determine if CPU optimization is relevant:
    - Platform is CPU-only
    - Low IPC (instructions per cycle)
    - High NUMA remote access
    - Low vectorization (AVX-512/AVX2)
    - High cache miss rate
    """

    # Thresholds for self-relevance detection
    RELEVANCE_THRESHOLDS = {
        "ipc_low": 1.0,  # instructions per cycle - below this is memory bound
        "cache_miss_rate_l3_high": 30.0,  # % - high L3 misses
        "numa_remote_high": 30.0,  # % - remote NUMA access
        "vectorization_low": 50.0,  # % - not using SIMD effectively
        "cpu_utilization_low": 70.0,  # % - underutilized CPU
    }

    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Only register on CPU platforms (no GPU).
        """
        platform_type = platform_info.get("type", "")
        
        if platform_type != "cpu":
            print(f"[CPUOptimizationsSME] Skipping: platform is '{platform_type}', "
                  f"not CPU. This SME only applies to CPU-only inference.")
            return None
        
        cpu_info = platform_info.get("cpu_info", {})
        cpu_count = cpu_info.get("cpu_count", 1)
        
        return RegistrationInfo(
            sme_id="cpu_optimizations",
            description="CPU optimization expert - analyzes CPU inference performance "
                       "for thread affinity, NUMA, cache efficiency, and quantization",
            data_requirements=[
                # OPTIONAL: CPU profiler (perf, vtune, or similar)
                DataRequirement(
                    data_type="cpu_profile_report",
                    required=False,
                    extractors=[
                        "cpu_utilization_percent",
                        "instructions_per_cycle",
                        "cache_miss_rate_l1",
                        "cache_miss_rate_l3",
                        "memory_bandwidth_gbps",
                        "numa_local_access_percent",
                        "numa_remote_access_percent",
                        "vectorization_percent",  # AVX-512, AVX2 usage
                    ]
                ),
                # REQUIRED: System metrics
                DataRequirement(
                    data_type="system_metrics",
                    required=True,
                    extractors=[
                        "cpu_freq_ghz",
                        "cpu_temp_celsius",
                        "memory_used_gb",
                        "memory_available_gb",
                        "swap_usage_gb",
                        "context_switches_per_sec",
                    ]
                ),
                # REQUIRED: vLLM logs for CPU config
                DataRequirement(
                    data_type="vllm_logs",
                    required=True,
                    extractors=[
                        "device_type",  # Should be "cpu"
                        "num_threads",
                        "num_cpus",
                        "cpu_quantization",  # int8, int4
                        "blas_library",  # MKL, OpenBLAS, etc.
                        "memory_pool_size_mb",
                    ]
                ),
                # REQUIRED: Benchmark metrics
                DataRequirement(
                    data_type="benchmark_metrics",
                    required=True,
                    extractors=[
                        "ttft_p50",
                        "ttft_p99",
                        "tpot_p50",
                        "tpot_p99",
                        "throughput_rps",
                        "latency_ms_per_token",
                    ]
                ),
                # OPTIONAL: NUMA topology
                DataRequirement(
                    data_type="numa_topology",
                    required=False,
                    extractors=[
                        "numa_nodes",
                        "cpus_per_numa_node",
                        "memory_per_numa_node_gb",
                        "inter_node_latency_ns",
                    ]
                ),
            ]
        )
    
    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if CPU optimization is relevant.
        
        CPU optimization is relevant when:
        1. Platform is CPU-only
        2. Low IPC (< 1.0) - memory bound
        3. High NUMA remote access (> 30%)
        4. Low vectorization (< 50%)
        5. High cache miss rate (> 30%)
        
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
        cpu_profile = profiling_data.get("cpu_profile_report", {})
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        
        # Check 1: Platform is CPU
        device_type = vllm_config.get("device_type", "cuda").lower()
        if device_type != "cpu":
            # Not a CPU platform - definitely not relevant
            return False, 0.0, f"Not CPU platform (device_type={device_type})"
        
        # Platform is CPU - base relevance
        relevance_score += 0.3
        
        # Check 2: Low IPC (memory bound)
        ipc = cpu_profile.get("instructions_per_cycle", 2.0)
        if ipc < self.RELEVANCE_THRESHOLDS["ipc_low"]:
            relevance_signals.append(f"low_ipc ({ipc:.2f})")
            relevance_score += 0.2
        
        # Check 3: NUMA remote access
        numa_remote = cpu_profile.get("numa_remote_access_percent", 0)
        if numa_remote > self.RELEVANCE_THRESHOLDS["numa_remote_high"]:
            relevance_signals.append(f"high_numa_remote ({numa_remote:.1f}%)")
            relevance_score += 0.15
        
        # Check 4: Cache miss rate
        l3_miss = cpu_profile.get("cache_miss_rate_l3", 0)
        if l3_miss > self.RELEVANCE_THRESHOLDS["cache_miss_rate_l3_high"]:
            relevance_signals.append(f"high_l3_miss ({l3_miss:.1f}%)")
            relevance_score += 0.15
        
        # Check 5: Vectorization
        vec_pct = cpu_profile.get("vectorization_percent", 100)
        if vec_pct < self.RELEVANCE_THRESHOLDS["vectorization_low"]:
            relevance_signals.append(f"low_vectorization ({vec_pct:.1f}%)")
            relevance_score += 0.1
        
        # Check 6: CPU utilization
        cpu_util = cpu_profile.get("cpu_utilization_percent", 100)
        if cpu_util < self.RELEVANCE_THRESHOLDS["cpu_utilization_low"]:
            relevance_signals.append(f"low_cpu_util ({cpu_util:.1f}%)")
            relevance_score += 0.1
        
        # Determine relevance
        is_relevant = relevance_score >= 0.4  # CPU platform + at least one signal
        
        if is_relevant:
            reason = f"CPU optimization relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = "CPU platform but no optimization signals detected"
        
        return is_relevant, min(relevance_score, 1.0), reason
    
    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze CPU inference efficiency using multi-LLM consensus.
        
        Key signals:
        - Low IPC (< 1.0): Memory bound, need better cache utilization
        - High NUMA remote access: Poor memory affinity
        - Low vectorization: Not using AVX-512/AVX2 effectively
        - High cache miss rate: Batch size too large for cache
        
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
                    "primary_bottleneck": "not_cpu_optimization_related",
                    "cpu_optimization_candidate": False,
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
        
        cpu_profile = profiling_data.get("cpu_profile_report", {})
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        
        # Analyze current state
        ipc = cpu_profile.get("instructions_per_cycle", 2.0)
        numa_remote = cpu_profile.get("numa_remote_access_percent", 0)
        vec_pct = cpu_profile.get("vectorization_percent", 100)
        
        findings["ipc"] = ipc
        findings["numa_remote_access"] = numa_remote
        findings["vectorization_percent"] = vec_pct
        
        # Suggest optimizations based on findings
        if ipc < 1.0:
            suggestions.append(ExperimentSuggestion(
                config_changes={"num_threads": "min(physical_cores, 16)"},
                expected_improvement="+15-25% throughput by reducing thread contention",
                confidence=0.5,
                rationale=f"Low IPC ({ipc:.2f}) suggests memory bottleneck. Reducing threads may improve cache locality."
            ))
        
        if numa_remote > 30:
            suggestions.append(ExperimentSuggestion(
                config_changes={"numa_aware_scheduling": True},
                expected_improvement="+10-20% memory bandwidth by using local NUMA nodes",
                confidence=0.6,
                rationale=f"High NUMA remote access ({numa_remote:.1f}%). NUMA-aware scheduling will improve memory latency."
            ))
        
        if vec_pct < 50:
            suggestions.append(ExperimentSuggestion(
                config_changes={"blas_library": "mkl"},  # MKL has better vectorization
                expected_improvement="+20-30% compute efficiency via better vectorization",
                confidence=0.4,
                rationale=f"Low vectorization ({vec_pct:.1f}%). Intel MKL provides optimized AVX-512 kernels."
            ))
        
        return SMEResponse(
            findings=findings,
            suggestions=suggestions,
            confidence=0.5 if suggestions else 0.0,
            is_relevant=True,
            relevance_score=relevance_score,
            relevance_reason=relevance_reason
        )
