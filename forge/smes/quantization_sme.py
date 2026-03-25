"""Quantization SME - Recommends quantization strategies for vLLM with multi-LLM consensus."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .base import BaseSME, DataRequirement, ExperimentSuggestion, SMEResponse, RegistrationInfo
from .utils import format_nsys_for_prompt, infer_bottleneck_from_nsys
from .quantization.hf_model_finder import find_best_quantized_model
from forge.llm import (
    IntelligencePool,
    Prompt,
    ConsensusEngine,
    ConsensusConfig,
    get_global_intelligence_pool,
)


@dataclass
class QuantizationRecommendation:
    """A quantization recommendation with full context."""
    target: str  # "weights", "kv_cache", "both"
    weight_method: Optional[str]  # "fp8", "awq", "gptq", "int8", None
    weight_bits: Optional[int]  # 4, 8, 16
    kv_cache_dtype: Optional[str]  # "fp8", None
    source: str  # "pre_quantized_hf", "on_the_fly"
    expected_memory_reduction: str
    expected_throughput_improvement: str
    rationale: str


class QuantizationSME(BaseSME):
    """
    Quantization Expert with multi-LLM AI brain.
    
    Analyzes memory bandwidth pressure, model size, KV cache pressure,
    and hardware capabilities to recommend optimal quantization strategies:
    
    - FP8: Best on Hopper/Ada GPUs with native FP8 Tensor Cores
    - AWQ: Activation-aware weight quantization, good accuracy
    - GPTQ: Layer-wise quantization, faster than AWQ
    - INT8: General-purpose, works on all Tensor Core GPUs
    - KV Cache FP8: Reduces KV cache memory by 50%
    
    Self-Relevance Detection:
    This SME scans data to determine if quantization is relevant:
    - Memory bandwidth utilization > 60%
    - KV cache usage > 70%
    - GPU memory utilization > 80%
    - Model is FP16/BF16 (quantization candidate)
    - Hardware supports FP8 or INT4/INT8 quantization
    """
    
    SYSTEM_PROMPT = """You are a vLLM QUANTIZATION optimization expert.

Your task: Analyze profiling data and hardware capabilities to recommend optimal QUANTIZATION strategies ONLY.

⚠️  CRITICAL: You are a QUANTIZATION expert ONLY. Only suggest quantization-related parameters.
   DO NOT suggest scheduling parameters like:
   - max_num_seqs
   - max_num_batched_tokens  
   - enable_chunked_prefill
   - prefill_chunk_size
   - max_chunked_prefill_len
   - scheduler_delay_factor
   - num_scheduler_steps
   
   Those are handled by the Scheduling SME, not you.

Key concepts:
- Memory bandwidth bound: When HBM bandwidth is saturated (>70%), quantization reduces memory traffic
- Model weights memory: Large models (13B+) benefit more from weight quantization  
- KV cache pressure: Quantization doubles effective KV cache capacity
- Hardware native support: FP8 only on Hopper (H100) and Ada (RTX 4090/L40S)

Quantization methods:
- FP8: Hardware-accelerated on Hopper/Ada, minimal accuracy loss, fastest
- AWQ: 4-bit, protects salient weights, better accuracy than GPTQ
- GPTQ: 4-bit, layer-wise, faster inference than AWQ
- INT8: 8-bit, good balance, works on all modern GPUs

KV cache quantization:
- FP8 KV cache: Supported on Hopper/Ada, 50% memory reduction
- Config: --kv-cache-dtype fp8 (independent of weight quantization)

Decision framework:
1. Check hardware_capabilities for FP8 support
2. If memory_bandwidth_bound AND supports_fp8 → FP8 weights + KV cache
3. If kv_cache_pressure AND supports_fp8 → FP8 KV cache only
4. If gpu_memory_high AND NOT supports_fp8 → AWQ/GPTQ 4-bit
5. If already_quantized → Check for KV cache quantization opportunity

Respond with JSON only:
{
  "findings": {
    "primary_bottleneck": "memory_bandwidth_bound|kv_cache_pressure|gpu_memory_high|compute_bound|already_optimized",
    "quantization_candidate": true|false,
    "recommended_target": "weights|kv_cache|both",
    "hardware_supports_fp8": true|false,
    "diagnosis": "2-3 sentence technical explanation"
  },
  "suggestions": [
    {
      "priority": 1,
      "target": "weights|kv_cache|both",
      "config_changes": {
        "quantization": "fp8|awq|gptq|int8",
        "kv_cache_dtype": "fp8"
      },
      "expected_improvement": "e.g., -50% memory, +40% throughput",
      "confidence": 0.0-1.0,
      "rationale": "Why this recommendation"
    }
  ],
  "confidence": 0.0-1.0
}

⚠️  CRITICAL RESTRICTION: 
- ONLY suggest these parameters: "quantization", "kv_cache_dtype", "model" (for quantized model variant)
- NEVER suggest: max_num_seqs, max_num_batched_tokens, enable_chunked_prefill, prefill_chunk_size, etc.
- If quantization is not appropriate, return empty suggestions or suggest "no_change"
- DO NOT include placeholder text like "model_id" - we will search for models separately
- The system will automatically search for pre-quantized models on HuggingFace"""

    def __init__(self, intelligence_pool: Optional[IntelligencePool] = None):
        """
        Initialize QuantizationSME.
        
        Args:
            intelligence_pool: Optional custom pool. Uses global pool if not provided.
        """
        self.pool = intelligence_pool or get_global_intelligence_pool()
        
        # Consensus configuration for quantization decisions
        # Higher threshold since quantization choices have lasting impact
        self.consensus_config = ConsensusConfig(
            similarity_threshold=0.8,
            min_agreement_ratio=0.5,
            require_unanimous_for_p1=False,
            weight_by_quality=True,
        )
        
        # Store hardware capabilities from registration
        self._hardware_capabilities: Optional[Dict[str, Any]] = None
        
        # Thresholds for self-relevance detection
        self.RELEVANCE_THRESHOLDS = {
            "memory_bandwidth_utilization": 60.0,  # %
            "kv_cache_usage": 70.0,  # %
            "gpu_memory_utilization": 80.0,  # %
            "compute_utilization": 50.0,  # % - low compute with high mem = bandwidth bound
        }

    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Register on CUDA and ROCm platforms.
        Store hardware capabilities for later use.
        """
        platform_type = platform_info.get("type", "")
        
        # Store hardware capabilities for analyze phase
        self._hardware_capabilities = platform_info.get("hardware_capabilities", {})
        
        if platform_type == "nvidia_cuda":
            return RegistrationInfo(
                sme_id="quantization",
                description="Quantization expert - analyzes memory bandwidth, KV cache pressure, "
                           "and hardware capabilities to recommend FP8/AWQ/GPTQ/INT8 quantization",
                data_requirements=[
                    # REQUIRED: System metrics for memory/bandwidth analysis
                    DataRequirement(
                        data_type="system_metrics_report",
                        required=True,
                        extractors=[
                            "gpu_utilization_percent",
                            "memory_used_mb",
                            "memory_total_mb",
                            "memory_utilization_percent",
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
                            "model_weights_memory_mb",
                            "kv_cache_memory_mb",
                            "kv_cache_usage_percent",
                            "activation_memory_mb",
                            "gpu_memory_peak_mb",
                            "oom_warnings",
                        ]
                    ),
                    # OPTIONAL: NCU for compute/bandwidth analysis
                    DataRequirement(
                        data_type="ncu_report",
                        required=False,
                        extractors=[
                            "memory_bandwidth_utilization_percent",
                            "compute_utilization_percent",
                            "dram_throughput_gbps",
                        ]
                    ),
                    # OPTIONAL: NSys for timeline analysis
                    DataRequirement(
                        data_type="nsys_report",
                        required=False,
                        extractors=[
                            "memory_bandwidth_gbps",
                            "hbm_utilization_percent",
                        ]
                    ),
                    # REQUIRED: Benchmark metrics
                    DataRequirement(
                        data_type="benchmark_metrics",
                        required=True,
                        extractors=[
                            "throughput_rps",
                            "output_tokens_per_sec",
                        ]
                    ),
                ]
            )
        
        elif platform_type == "amd_rocm":
            # AMD ROCm has different quantization support
            return RegistrationInfo(
                sme_id="quantization",
                description="Quantization expert for AMD ROCm platforms",
                data_requirements=[
                    DataRequirement(
                        data_type="system_metrics_report",
                        required=True,
                        extractors=[
                            "gpu_utilization_percent",
                            "memory_used_mb",
                            "memory_total_mb",
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
                        ]
                    ),
                ]
            )
        
        # Not supported on other platforms
        return None

    def scan_data(self, 
                  profile_dir: Path,
                  profiling_data: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any]) -> tuple[bool, float, str]:
        """
        Scan profiling data to determine if quantization is relevant.
        
        Quantization is relevant when:
        1. Memory bandwidth utilization > 60% (quantization reduces memory traffic)
        2. KV cache usage > 70% (FP8 KV cache doubles capacity)
        3. GPU memory utilization > 80% (weight quantization reduces memory)
        4. Model is FP16/BF16 (candidate for quantization)
        5. Compute utilization < 50% with high memory bandwidth (memory bound)
        
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
        hw_caps = profiling_data.get("hardware_capabilities") or self._hardware_capabilities or {}
        
        # Check 1: Memory bandwidth utilization
        mem_bw_util = max(
            ncu_report.get("memory_bandwidth_utilization_percent", 0),
            system_metrics.get("memory_utilization_percent", 0)
        )
        if mem_bw_util > self.RELEVANCE_THRESHOLDS["memory_bandwidth_utilization"]:
            relevance_signals.append(f"memory_bandwidth_high ({mem_bw_util:.1f}%)")
            relevance_score += 0.3
        
        # Check 2: KV cache usage
        kv_usage = vllm_metrics.get("kv_cache_usage_percent", 0)
        if kv_usage > self.RELEVANCE_THRESHOLDS["kv_cache_usage"]:
            relevance_signals.append(f"kv_cache_pressure ({kv_usage:.1f}%)")
            relevance_score += 0.25
        
        # Check 3: GPU memory utilization
        gpu_mem_util = system_metrics.get("memory_utilization_percent", 0)
        if gpu_mem_util > self.RELEVANCE_THRESHOLDS["gpu_memory_utilization"]:
            relevance_signals.append(f"gpu_memory_high ({gpu_mem_util:.1f}%)")
            relevance_score += 0.2
        
        # Check 4: Model dtype (is it a quantization candidate?)
        model_dtype = vllm_config.get("model_dtype", "fp16").lower()
        current_quant = vllm_config.get("quantization", "none").lower()
        if model_dtype in ["fp16", "bf16", "float16", "bfloat16"] and current_quant in ["none", ""]:
            relevance_signals.append(f"model_is_quantization_candidate ({model_dtype})")
            relevance_score += 0.15
        
        # Check 5: Compute vs Memory bound (low compute + high memory = bandwidth bound)
        compute_util = ncu_report.get("compute_utilization_percent", 100)
        if compute_util < self.RELEVANCE_THRESHOLDS["compute_utilization"] and mem_bw_util > 60:
            relevance_signals.append(f"likely_memory_bound (compute={compute_util:.1f}%, mem_bw={mem_bw_util:.1f}%)")
            relevance_score += 0.1
        
        # Check 6: Hardware supports quantization
        supports_fp8 = hw_caps.get("supports_fp8", False)
        recommended = hw_caps.get("recommended_weight_quant", [])
        if supports_fp8 or recommended:
            relevance_signals.append(f"hardware_supports_quantization (fp8={supports_fp8})")
            relevance_score += 0.1
        
        # Determine relevance
        is_relevant = relevance_score >= 0.3  # At least 2-3 signals needed
        
        if is_relevant:
            reason = f"Quantization relevant: {'; '.join(relevance_signals)}"
        else:
            if relevance_signals:
                reason = f"Weak relevance ({relevance_score:.2f}): {'; '.join(relevance_signals)}"
            else:
                reason = "No quantization signals detected (low memory pressure, already quantized, or compute bound)"
        
        return is_relevant, min(relevance_score, 1.0), reason

    async def analyze(self,
                      profile_dir: Path,
                      profiling_data: Dict[str, Any],
                      benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """
        Analyze quantization opportunities using multi-LLM consensus.
        
        Key signals analyzed:
        - Memory bandwidth bound → FP8 weights (if hardware supports)
        - KV cache pressure → FP8 KV cache (if hardware supports)
        - High GPU memory → AWQ/GPTQ 4-bit
        - Current dtype (FP16/BF16) → Quantization candidate
        
        Args:
            profile_dir: Path to directory containing profiling data
            profiling_data: Collected profiler outputs + hardware_capabilities
            benchmark_metrics: Benchmark results
        
        Returns:
            SMEResponse with consensus-based findings and suggestions
        """
        # First, check relevance
        is_relevant, relevance_score, relevance_reason = self.scan_data(
            profile_dir, profiling_data, benchmark_metrics
        )
        
        if not is_relevant:
            return SMEResponse(
                findings={
                    "primary_bottleneck": "not_quantization_related",
                    "quantization_candidate": False,
                },
                suggestions=[],
                confidence=0.0,
                is_relevant=False,
                relevance_score=relevance_score,
                relevance_reason=relevance_reason
            )
        
        # Get hardware capabilities
        hw_caps = profiling_data.get("hardware_capabilities") or self._hardware_capabilities or {}
        
        # 1. Prepare structured prompt with all data
        prompt = self._build_analysis_prompt(profiling_data, benchmark_metrics, hw_caps)
        
        # 2. Call all available intelligence sources in parallel
        print(f"   [QuantizationSME] Calling LLM pool (sources: {len(self.pool.sources)})...")
        call_results = await self.pool.call_all(
            prompt,
            deterministic=self.consensus_config.temperature <= 0.0
        )
        
        print(f"   [QuantizationSME] Got {len(call_results)} call results")
        for i, cr in enumerate(call_results):
            status = "✓" if cr.success else "✗"
            error = f" ({cr.error})" if cr.error else ""
            print(f"      {status} {cr.source.provider}/{cr.source.model}{error}")
        
        if not call_results:
            print("   [QuantizationSME] ⚠️ No LLM intelligence sources available!")
            return self._create_fallback_response(hw_caps)
        
        # 3. Compute consensus across all model responses
        print(f"   [QuantizationSME] Computing consensus...")
        consensus_result = ConsensusEngine.compute(
            call_results, 
            self.consensus_config
        )
        
        print(f"   [QuantizationSME] Consensus: {len(consensus_result.suggestions)} suggestions")
        
        # 4. Enhance suggestions with HF model search and KV cache recommendations
        suggestions = []
        for cs in consensus_result.suggestions:
            print(f"      → {cs.config_changes} (confidence: {cs.weighted_confidence:.2f})")
            
            # Search for pre-quantized model on HF if weight quantization recommended
            config_changes = dict(cs.config_changes)
            
            # Clean up placeholder values from LLM output
            self._clean_placeholder_values(config_changes)
            
            if "quantization" in config_changes:
                quant_method = config_changes["quantization"]
                vllm_logs = profiling_data.get("vllm_logs", {})
                vllm_config = vllm_logs.get("config", {})
                base_model = vllm_config.get("model_name", "")
                
                print(f"   [QuantizationSME] Base model from logs: '{base_model}'")
                
                if base_model:
                    # Try to find pre-quantized model
                    bits = 4 if quant_method in ("awq", "gptq") else 8
                    print(f"   [QuantizationSME] Searching HF for {quant_method} {bits}-bit model...")
                    
                    try:
                        quantized_model = await find_best_quantized_model(
                            base_model, quant_method, bits
                        )
                        if quantized_model:
                            config_changes["model"] = quantized_model
                            config_changes["model_source"] = "pre_quantized_hf"
                            print(f"      ✓ Found quantized model: {quantized_model}")
                        else:
                            # For GPTQ/AWQ, we NEED a pre-quantized model - vLLM can't quantize on-the-fly
                            if quant_method in ("gptq", "awq"):
                                print(f"      ⚠️ No pre-quantized {quant_method} model found on HF")
                                print(f"         Skipping {quant_method} recommendation (requires pre-quantized model)")
                                # Skip this suggestion - can't be executed without pre-quantized model
                                continue
                            else:
                                # FP8 can be done on-the-fly
                                config_changes["model_source"] = "on_the_fly"
                                print(f"      ℹ️ No pre-quantized model found, will use on-the-fly {quant_method}")
                    except Exception as e:
                        print(f"      ⚠️ HF search failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # For GPTQ/AWQ, skip if search fails
                        if quant_method in ("gptq", "awq"):
                            print(f"         Skipping {quant_method} due to search failure")
                            continue
                        config_changes["model_source"] = "on_the_fly"
                else:
                    print(f"   [QuantizationSME] ⚠️ No base_model found in vllm_logs, cannot search HF")
                    print(f"      vllm_config keys: {list(vllm_config.keys())}")
                    # For GPTQ/AWQ, skip if we can't find the base model
                    if quant_method in ("gptq", "awq"):
                        print(f"      Skipping {quant_method} - cannot find pre-quantized model without base model name")
                        continue
            
            print(f"   [QuantizationSME] Final config_changes: {config_changes}")
            suggestions.append(ExperimentSuggestion(
                config_changes=config_changes,
                expected_improvement=cs.expected_improvement,
                confidence=cs.weighted_confidence,
                rationale=cs.rationale
            ))
        
        # 5. Extract findings from consensus
        primary_finding = self._extract_primary_finding(consensus_result)
        
        return SMEResponse(
            findings={
                "primary_bottleneck": primary_finding,
                "quantization_candidate": self._is_quantization_candidate(profiling_data),
                "hardware_supports_fp8": hw_caps.get("supports_fp8", False),
                "recommended_target": self._extract_recommended_target(consensus_result),
                "consensus_models": consensus_result.total_models,
                "successful_models": consensus_result.successful_models,
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
        benchmark_metrics: Dict[str, Any],
        hw_caps: Dict[str, Any]
    ) -> Prompt:
        """Build structured analysis prompt from profiling data."""
        
        # Extract vLLM config and runtime metrics
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        vllm_metrics = vllm_logs.get("metrics", {})
        
        # Extract system metrics
        system_metrics = profiling_data.get("system_metrics_report", {})
        
        # Extract NCU metrics
        ncu_report = profiling_data.get("ncu_report", {})
        
        # Extract NSYS analysis
        nsys_data = profiling_data.get("nsys_report", {})
        nsys_section = format_nsys_for_prompt(nsys_data, include_timeline=False)
        
        # Build user prompt
        user_content = f"""Analyze this vLLM profiling data and recommend quantization strategies.

## Hardware Capabilities
```json
{json.dumps(hw_caps, indent=2)}
```

## Current Configuration
```json
{json.dumps(vllm_config, indent=2)}
```

## System Metrics
```json
{json.dumps(system_metrics, indent=2)}
```

## vLLM Runtime Metrics
```json
{json.dumps(vllm_metrics, indent=2)}
```

## NCU Metrics
```json
{json.dumps(ncu_report, indent=2)}
```

{nsys_section}

## Benchmark Results
```json
{json.dumps(benchmark_metrics, indent=2)}
```

## Analysis Task

Based on the hardware capabilities and profiling data:

1. Check if hardware supports FP8 (Hopper/Ada GPUs)
2. Determine if workload is memory bandwidth bound
3. Check KV cache utilization
4. Identify current model dtype (FP16/BF16 = quantization candidate)
5. Recommend specific quantization strategy

Consider:
- If supports_fp8 AND memory bandwidth > 70%: Recommend FP8 weights + KV cache
- If supports_fp8 AND kv_cache_usage > 80%: Recommend FP8 KV cache only
- If NOT supports_fp8 AND memory pressure: Recommend AWQ/GPTQ 4-bit
- If already quantized: Check for KV cache quantization opportunity

Respond with JSON only (no markdown)."""

        return Prompt(
            system=self.SYSTEM_PROMPT,
            user=user_content
        )
    
    def _clean_placeholder_values(self, config_changes: Dict[str, Any]) -> None:
        """Remove placeholder values that the LLM might output.
        
        The LLM sometimes outputs placeholder text like:
        - "HF model ID if pre-quantized"
        - "Use GPTQ-quantized variant of current model"
        
        These should be removed - we'll either find a real model on HF
        or use on-the-fly quantization.
        """
        # Patterns that indicate placeholder values
        placeholder_patterns = [
            "HF model ID",
            "if pre-quantized",
            "Use ",
            "variant of current model",
            "GPTQ-quantized",
            "AWQ-quantized",
            "FP8-quantized",
        ]
        
        # Check model_id field (from LLM output template)
        if "model_id" in config_changes:
            value = str(config_changes["model_id"])
            if any(pattern in value for pattern in placeholder_patterns):
                print(f"      ℹ️ Removing placeholder model_id: {value}")
                del config_changes["model_id"]
        
        # Check model field
        if "model" in config_changes:
            value = str(config_changes["model"])
            if any(pattern in value for pattern in placeholder_patterns):
                print(f"      ℹ️ Removing placeholder model: {value}")
                del config_changes["model"]
    
    def _create_fallback_response(self, hw_caps: Dict[str, Any]) -> SMEResponse:
        """Create fallback response when LLM is unavailable."""
        suggestions = []
        
        # Use rule-based logic when LLM is unavailable
        supports_fp8 = hw_caps.get("supports_fp8", False)
        recommended = hw_caps.get("recommended_weight_quant", [])
        
        if supports_fp8 and "fp8" in recommended:
            suggestions.append(ExperimentSuggestion(
                config_changes={"quantization": "fp8"},
                expected_improvement="-50% memory bandwidth, +30% throughput on Hopper/Ada",
                confidence=0.7,
                rationale="Hardware supports FP8. Recommend FP8 quantization for memory bandwidth reduction."
            ))
        elif recommended:
            method = recommended[0]
            suggestions.append(ExperimentSuggestion(
                config_changes={"quantization": method},
                expected_improvement="-50% model memory, +20% throughput",
                confidence=0.6,
                rationale=f"Recommend {method.upper()} quantization to reduce memory pressure."
            ))
        
        return SMEResponse(
            findings={
                "primary_bottleneck": "unknown",
                "quantization_candidate": True,
                "hardware_supports_fp8": supports_fp8,
                "fallback": True,
            },
            suggestions=suggestions,
            confidence=0.5,
            is_relevant=True,
            relevance_score=0.5,
            relevance_reason="Fallback mode - using rule-based recommendations"
        )
    
    def _extract_primary_finding(self, consensus_result) -> str:
        """Extract primary bottleneck from consensus divergence report."""
        votes = consensus_result.divergence_report.bottleneck_votes
        
        if not votes:
            return "unknown"
        
        primary = max(votes.items(), key=lambda x: len(x[1]))
        return primary[0]
    
    def _extract_recommended_target(self, consensus_result) -> str:
        """Extract recommended quantization target from consensus."""
        for suggestion in consensus_result.suggestions:
            target = suggestion.config_changes.get("target")
            if target:
                return target
            # Infer from config changes
            has_weights = "quantization" in suggestion.config_changes
            has_kv = "kv_cache_dtype" in suggestion.config_changes
            if has_weights and has_kv:
                return "both"
            elif has_weights:
                return "weights"
            elif has_kv:
                return "kv_cache"
        return "unknown"
    
    def _is_quantization_candidate(self, profiling_data: Dict[str, Any]) -> bool:
        """Determine if current config is a quantization candidate."""
        vllm_logs = profiling_data.get("vllm_logs", {})
        vllm_config = vllm_logs.get("config", {})
        
        # Check current dtype
        current_dtype = vllm_config.get("model_dtype", "fp16").lower()
        if current_dtype in ["fp16", "bf16", "float16", "bfloat16"]:
            return True
        
        # Check current quantization
        current_quant = vllm_config.get("quantization", "none").lower()
        if current_quant in ["none", ""]:
            return True
            
        return False
