"""VLLM Log Collector - Parses vLLM server logs.

This is the simplest profiler - no special hardware or permissions needed.
Extracts configuration and runtime metrics from vLLM log files.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseProfiler,
    DataQualityError,
    ExtractorError,
    ProfilingContext,
    RawProfilerOutput,
)


class VLLMLogCollector(BaseProfiler):
    """Collects and parses vLLM server logs.
    
    Extracts:
    - Configuration settings (max_num_seqs, quantization, etc.)
    - Runtime metrics (KV cache usage, scheduler queue depth)
    - Performance statistics
    - Error messages
    
    No special permissions or hardware required.
    """
    
    DATA_TYPE = "vllm_logs"
    DEPENDENCIES = []  # Can collect anytime
    ESTIMATED_DURATION_SECONDS = 5
    
    # Regex patterns for log parsing
    # These are matched in order, first match wins
    LOG_PATTERNS = {
        # Configuration patterns
        "max_num_seqs": r"max_num_seqs[=:]\s*(\d+)",
        "max_model_len": r"max_model_len[=:]\s*(\d+)",
        "tensor_parallel_size": r"tensor[_-]?parallel[_-]?size[=:]\s*(\d+)",
        "pipeline_parallel_size": r"pipeline[_-]?parallel[_-]?size[=:]\s*(\d+)",
        "quantization": r"quantization[=:]\s*(\w+)",
        "dtype": r"dtype[=:]\s*(\w+)",
        "gpu_memory_utilization": r"gpu_memory_utilization[=:]\s*(0?\.\d+|1\.0?)",
        "load_format": r"load_format[=:]\s*(\w+)",
        
        # Scheduler patterns
        "scheduler_policy": r"scheduler[_-]?policy[=:]\s*(\w+)",
        "enable_chunked_prefill": r"enable_chunked_prefill[=:]\s*(True|False)",
        "max_num_batched_tokens": r"max_num_batched_tokens[=:]\s*(\d+)",
        
        # KV cache patterns (vLLM periodic stats)
        "kv_cache_usage_percent": r"GPU KV cache usage[:\s]+(\d+\.?\d*)\s*%",
        "kv_cache_tokens": r"KV cache tokens[:\s]+(\d+)",
        
        # Queue patterns
        "scheduler_queue_depth": r"(Waiting|Queued)[:\s]+(\d+)",
        "running_requests": r"Running[:\s]+(\d+)",
        "swapped_requests": r"Swapped[:\s]+(\d+)",
        
        # Performance patterns
        "tokens_per_second": r"(?:tokens|throughput)[:\s]+(\d+\.?\d*)\s*/?\s*s",
        "generation_time": r"generation time[:\s]+(\d+\.?\d*)\s*(?:ms|s)",
        
        # Model info
        "model_name": r'Loading model[:\s]+(["\']?)([\w\-/\.]+)\1',
        "model_architecture": r"model architecture[:\s]+(\w+)",
        "model_parameters": r"(\d+\.?\d*)\s*[BM]\s*parameters",
        
        # Startup info
        "vllm_version": r"vLLM version[:\s]+([\d\.]+)",
        "cuda_version": r"CUDA version[:\s]+([\d\.]+)",
        
        # DEBUG level patterns (vLLM --log-level DEBUG)
        # Attention/Transformer debug info
        "attention_backend": r"Using\s+(\w+)\s+attention backend",
        "attention_sliding_window": r"sliding_window\s*=\s*(\d+)",
        "num_attention_heads": r"num_attention_heads[=:]\s*(\d+)",
        "num_key_value_heads": r"num_key_value_heads[=:]\s*(\d+)",
        "head_dim": r"head_dim\s*=\s*(\d+)",
        
        # Memory allocation debug
        "gpu_memory_allocated_gb": r"GPU memory allocated[:\s]+(\d+\.?\d*)\s*[Gg][Bb]?",
        "gpu_memory_reserved_gb": r"GPU memory reserved[:\s]+(\d+\.?\d*)\s*[Gg][Bb]?",
        "kv_cache_size_gb": r"KV cache size[:\s]+(\d+\.?\d*)\s*[Gg][Bb]?",
        "gpu_blocks": r"num_gpu_blocks[=:]\s*(\d+)",
        "cpu_blocks": r"num_cpu_blocks[=:]\s*(\d+)",
        "block_size": r"block_size[=:]\s*(\d+)",
        
        # Engine/Worker debug
        "num_workers": r"num_workers[=:]\s*(\d+)",
        "device_count": r"device_count[=:]\s*(\d+)",
        "distributed_init_method": r"distributed_init_method[=:]\s*(\S+)",
        
        # Request processing debug
        "prompt_tokens": r"prompt_tokens[=:]\s*(\d+)",
        "completion_tokens": r"completion_tokens[=:]\s*(\d+)",
        "total_tokens": r"total_tokens[=:]\s*(\d+)",
        "prefill_tokens": r"prefill_tokens[=:]\s*(\d+)",
        "decode_tokens": r"decode_tokens[=:]\s*(\d+)",
        "request_arrival_time": r"arrival_time[:\s]+([\d\.]+)",
        "request_last_token_time": r"last_token_time[:\s]+([\d\.]+)",
        "scheduling_latency_ms": r"scheduling_latency[:\s]+(\d+\.?\d*)\s*ms",
        
        # Sampling debug
        "sampling_temperature": r"temperature[=:]\s*(\d+\.?\d*)",
        "sampling_top_p": r"top_p[=:]\s*(\d+\.?\d*)",
        "sampling_top_k": r"top_k[=:]\s*(\d+)",
        "sampling_repetition_penalty": r"repetition_penalty[=:]\s*(\d+\.?\d*)",
        
        # Speculative decoding debug
        "speculative_draft_model": r"Draft model[:\s]+([\w\-/\.]+)",
        "speculative_num_steps": r"num_speculative_tokens[=:]\s*(\d+)",
        "speculative_acceptance_rate": r"acceptance_rate[:\s]+(\d+\.?\d*)",
        "speculative_drafted_tokens": r"drafted_tokens[:\s]+(\d+)",
        "speculative_verified_tokens": r"verified_tokens[:\s]+(\d+)",
        
        # CUDA/GPU debug
        "cuda_device": r"CUDA device[:\s]+(\w+)",
        "cuda_visible_devices": r"CUDA_VISIBLE_DEVICES[=:]\s*(\S+)",
        "p2p_bandwidth": r"P2P bandwidth[:\s]+(\d+\.?\d*)\s*GB/s",
        
        # Error patterns (WARNING/ERROR level)
        "out_of_memory_error": r"(CUDA out of memory|OutOfMemoryError)",
        "memory_allocation_failure": r"Failed to allocate memory",
        "cuda_error": r"CUDA error[:\s]+(\w+[:\s]+[^\n]+)",
        "nccl_error": r"NCCL error[:\s]+([^\n]+)",
    }
    
    async def run(self, context: ProfilingContext) -> RawProfilerOutput:
        """Read vLLM log file.
        
        Args:
            context: Profiling context with vllm_log_path
            
        Returns:
            RawProfilerOutput with log content
            
        Raises:
            FileNotFoundError: If log file doesn't exist
        """
        log_path = context.vllm_log_path
        
        if not log_path:
            raise FileNotFoundError("No vllm_log_path provided in context")
        
        if not Path(log_path).exists():
            raise FileNotFoundError(f"vLLM log file not found: {log_path}")
        
        # Read log file
        log_content = await self._read_log_file(log_path)
        
        return RawProfilerOutput(
            report_path=log_path,
            log_content=log_content,
            metadata={
                "log_size_bytes": len(log_content.encode('utf-8')),
                "line_count": len(log_content.splitlines()),
            }
        )
    
    async def _read_log_file(self, log_path: Path) -> str:
        """Read log file asynchronously.
        
        Uses asyncio to avoid blocking the event loop for large files.
        """
        loop = asyncio.get_event_loop()
        
        def _read():
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        return await loop.run_in_executor(None, _read)
    
    async def extract(
        self,
        raw_output: RawProfilerOutput,
        extractors: List[str]
    ) -> Dict[str, Any]:
        """Parse log content and extract requested metrics.
        
        Args:
            raw_output: Output from run() with log_content
            extractors: List of metric names to extract
            
        Returns:
            Dict mapping extractor name to extracted value
            
        Example:
            extractors=["max_num_seqs", "quantization"]
            Returns: {"max_num_seqs": 64, "quantization": "awq"}
        """
        results = {}
        log_content = raw_output.log_content
        
        if not log_content:
            raise DataQualityError("No log content to extract from")
        
        for extractor in extractors:
            try:
                value = self._extract_single(log_content, extractor)
                results[extractor] = value
            except ExtractorError as e:
                # Log warning but don't fail - some extractors may be optional
                results[extractor] = None
                results[f"{extractor}_error"] = str(e)
        
        return results
    
    def _extract_single(self, log_content: str, extractor: str) -> Any:
        """Extract a single metric from log content.
        
        Args:
            log_content: Full log content
            extractor: Name of the metric to extract
            
        Returns:
            Extracted value (type depends on extractor)
            
        Raises:
            ExtractorError: If extractor not found or pattern doesn't match
        """
        if extractor not in self.LOG_PATTERNS:
            # Try to extract as a generic key=value pattern
            pattern = rf"{extractor}[=:]\s*([^\s,;]+)"
        else:
            pattern = self.LOG_PATTERNS[extractor]
        
        matches = re.findall(pattern, log_content, re.IGNORECASE)
        
        if not matches:
            raise ExtractorError(f"Pattern for '{extractor}' not found in logs")
        
        # Handle different match formats
        value = matches[-1]  # Take last match (most recent)
        
        # Handle tuple matches (groups)
        if isinstance(value, tuple):
            # If pattern has multiple groups, take the last non-empty one
            value = next((v for v in reversed(value) if v), value[0])
        
        # Type conversion
        return self._convert_type(extractor, value)
    
    def _convert_type(self, extractor: str, value: str) -> Any:
        """Convert extracted string value to appropriate type.
        
        Args:
            extractor: Name of the extractor (for type hints)
            value: String value from regex match
            
        Returns:
            Converted value (int, float, bool, or str)
        """
        # Boolean extractors
        if extractor.startswith("enable_"):
            return value.lower() in ("true", "1", "yes", "on")
        
        # Percentage extractors
        if "percent" in extractor or extractor.endswith("_pct"):
            try:
                return float(value)
            except ValueError:
                return value
        
        # Try int first
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def is_available(self) -> bool:
        """Check if log collection is available.
        
        Always returns True since we just need to read files.
        """
        return True
    
    def get_all_extracted_metrics(self, log_content: str) -> Dict[str, Any]:
        """Extract all known metrics from log content.
        
        This is a convenience method for debugging/analysis.
        
        Args:
            log_content: Full log content
            
        Returns:
            Dict of all extractable metrics
        """
        results = {}
        
        for extractor in self.LOG_PATTERNS.keys():
            try:
                value = self._extract_single(log_content, extractor)
                results[extractor] = value
            except ExtractorError:
                pass  # Skip metrics not found
        
        return results
