"""Comprehensive vLLM configuration space definition.

Defines all tunable vLLM parameters and their valid values.
Optimized for single-GPU systems (no model parallelism).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
import itertools
import json


@dataclass
class ConfigOption:
    """A single configuration dimension."""
    name: str
    values: List[Any]
    # If set, this option is only valid when the conditional field has the specified value
    conditional_on: Optional[Dict[str, Any]] = None
    
    def is_valid_for(self, config: Dict[str, Any]) -> bool:
        """Check if this option is valid given current config."""
        if self.conditional_on is None:
            return True
        for key, value in self.conditional_on.items():
            if config.get(key) != value:
                return False
        return True


class VLLMConfigSpace:
    """
    Comprehensive vLLM configuration space for single-GPU systems.
    
    Covers all major tunable parameters (no model parallelism):
    - Scheduling: max_num_seqs, max_num_batched_tokens
    - Chunked prefill: enable/disable, chunk size
    - Note: num_scheduler_steps removed (not available in vLLM 0.16.0)
    - Quantization: fp8, int8_awq, int4_awq
    - Speculative decoding: n-gram based
    - Attention backend
    """
    
    def __init__(self):
        self._build_dimensions()
    
    def _build_dimensions(self):
        """Define all configuration dimensions."""
        
        # Core scheduling (always applicable)
        self.scheduling_options = [
            ConfigOption("max_num_seqs", [64, 128, 256, 512, 1024]),
            ConfigOption("max_num_batched_tokens", [1024, 2048, 4096, 8192]),
        ]
        
        # Chunked prefill options
        self.chunked_prefill_options = [
            ConfigOption("enable_chunked_prefill", [False, True]),
            ConfigOption("max_chunked_prefill_len", [512, 1024, 2048], 
                        conditional_on={"enable_chunked_prefill": True}),
        ]
        
        # Note: num_scheduler_steps removed - not available in vLLM 0.16.0 API
        self.scheduler_options = []
        
        # Quantization options
        self.quantization_options = [
            ConfigOption("quantization", [None, "fp8"]),
        ]
        
        # Speculative decoding (n-gram based as minimum)
        self.speculative_options = [
            ConfigOption("speculative_model", [None, "ngram"]),
            ConfigOption("num_lookahead_slots", [1, 2, 4], 
                        conditional_on={"speculative_model": "ngram"}),
            ConfigOption("ngram_prompt_lookup_max", [2, 4, 8],
                        conditional_on={"speculative_model": "ngram"}),
            ConfigOption("ngram_prompt_lookup_min", [1, 2],
                        conditional_on={"speculative_model": "ngram"}),
        ]
        
        # Attention backend
        self.attention_options = [
            ConfigOption("attention_backend", [None, "flash_attn", "flashinfer"]),
        ]
        
        # Combine all options
        self.all_options = (
            self.scheduling_options +
            self.chunked_prefill_options +
            self.scheduler_options +
            self.quantization_options +
            self.speculative_options +
            self.attention_options
        )
    
    def enumerate_configs(self, max_configs: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate all valid configurations.
        
        For conditional options, only include when condition is met.
        This generates a comprehensive but filtered config space.
        """
        configs = []
        
        # Generate base combinations (non-conditional options)
        base_options = [opt for opt in self.all_options if opt.conditional_on is None]
        base_values = [opt.values for opt in base_options]
        base_names = [opt.name for opt in base_options]
        
        for combo in itertools.product(*base_values):
            base_config = dict(zip(base_names, combo))
            
            # Add conditional options that are valid for this base config
            conditional_configs = [base_config.copy()]
            
            for opt in self.all_options:
                if opt.conditional_on is not None and opt.is_valid_for(base_config):
                    new_configs = []
                    for value in opt.values:
                        for cfg in conditional_configs:
                            new_cfg = cfg.copy()
                            new_cfg[opt.name] = value
                            new_configs.append(new_cfg)
                    conditional_configs = new_configs
            
            configs.extend(conditional_configs)
        
        # Remove duplicates and invalid configs
        configs = self._deduplicate_and_validate(configs)
        
        return configs
    
    def _deduplicate_and_validate(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate configs and validate constraints."""
        seen = set()
        valid_configs = []
        
        for config in configs:
            # Validate: if speculative_model is None, remove speculative params
            if config.get("speculative_model") is None:
                config.pop("num_lookahead_slots", None)
                config.pop("ngram_prompt_lookup_max", None)
                config.pop("ngram_prompt_lookup_min", None)
            
            # Validate: if chunked prefill is False, remove chunk size
            if not config.get("enable_chunked_prefill", False):
                config.pop("max_chunked_prefill_len", None)
            
            # Create stable key for deduplication
            config_id = self.get_config_id(config)
            if config_id not in seen:
                seen.add(config_id)
                valid_configs.append(config)
        
        return valid_configs
    
    def get_config_id(self, config: Dict[str, Any]) -> str:
        """Generate stable ID for a config."""
        # Normalize: sort keys, convert to JSON
        return json.dumps(config, sort_keys=True, default=str)
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample a random valid configuration."""
        import random
        
        config = {}
        
        # Add non-conditional options
        for opt in self.all_options:
            if opt.conditional_on is None:
                config[opt.name] = random.choice(opt.values)
        
        # Add conditional options that are valid
        for opt in self.all_options:
            if opt.conditional_on is not None and opt.is_valid_for(config):
                # 50% chance to include this conditional option
                if random.random() < 0.5:
                    config[opt.name] = random.choice(opt.values)
        
        return config
    
    def estimate_size(self) -> int:
        """Estimate the total config space size."""
        return len(self.enumerate_configs())
    
    def print_summary(self):
        """Print a summary of the config space."""
        print("="*60)
        print("VLLM Configuration Space (Single GPU)")
        print("="*60)
        print(f"\nConfiguration Dimensions:")
        
        for opt in self.all_options:
            cond_str = ""
            if opt.conditional_on:
                cond_str = f" (if {opt.conditional_on})"
            print(f"  {opt.name}: {opt.values}{cond_str}")
        
        print(f"\nTotal Config Space Size: {self.estimate_size()}")
        print("="*60)


# Predefined config spaces for different scenarios
class ConfigSpacePresets:
    """Predefined configuration spaces for different use cases."""
    
    @staticmethod
    def minimal() -> VLLMConfigSpace:
        """Minimal space for quick testing."""
        space = VLLMConfigSpace()
        # Override with smaller options
        space.all_options = [
            ConfigOption("max_num_seqs", [128, 256, 512]),
            ConfigOption("max_num_batched_tokens", [2048, 4096]),
            ConfigOption("enable_chunked_prefill", [False, True]),
            # num_scheduler_steps removed - not in vLLM 0.16.0
            ConfigOption("speculative_model", [None, "ngram"]),
            ConfigOption("num_lookahead_slots", [2, 4],
                        conditional_on={"speculative_model": "ngram"}),
        ]
        return space
    
    @staticmethod
    def comprehensive() -> VLLMConfigSpace:
        """Full comprehensive space."""
        return VLLMConfigSpace()
    
    @staticmethod
    def scheduling_only() -> VLLMConfigSpace:
        """Focus on scheduling parameters only."""
        space = VLLMConfigSpace()
        space.all_options = (
            space.scheduling_options + 
            space.chunked_prefill_options +
            space.scheduler_options
        )
        return space
    
    @staticmethod
    def awq_quantized() -> VLLMConfigSpace:
        """
        Config space optimized for AWQ quantized models on limited VRAM.
        
        AWQ models use ~4-bit weights but still need careful memory management.
        Conservative batch sizes to avoid OOM on 14-16GB GPUs.
        """
        space = VLLMConfigSpace()
        space.scheduling_options = [
            ConfigOption("max_num_seqs", [64, 128, 256]),  # Reduced from 1024
            ConfigOption("max_num_batched_tokens", [1024, 2048, 4096]),  # Reduced from 8192
        ]
        space.chunked_prefill_options = [
            ConfigOption("enable_chunked_prefill", [False, True]),
            ConfigOption("max_chunked_prefill_len", [512, 1024],
                        conditional_on={"enable_chunked_prefill": True}),
        ]
        space.scheduler_options = [
            # num_scheduler_steps removed - not in vLLM 0.16.0
        ]
        space.quantization_options = [
            ConfigOption("quantization", ["awq"]),  # Fixed to AWQ
        ]
        space.speculative_options = [
            ConfigOption("speculative_model", [None, "ngram"]),
            ConfigOption("num_lookahead_slots", [1, 2],
                        conditional_on={"speculative_model": "ngram"}),
        ]
        space.all_options = (
            space.scheduling_options +
            space.chunked_prefill_options +
            space.scheduler_options +
            space.quantization_options +
            space.speculative_options
        )
        return space
    
    @staticmethod
    def memory_constrained() -> VLLMConfigSpace:
        """
        For systems with limited GPU memory (<16GB).
        Very conservative settings.
        """
        space = VLLMConfigSpace()
        space.scheduling_options = [
            ConfigOption("max_num_seqs", [64, 128]),
            ConfigOption("max_num_batched_tokens", [1024, 2048]),
        ]
        space.chunked_prefill_options = [
            ConfigOption("enable_chunked_prefill", [True]),  # Always on for memory
            ConfigOption("max_chunked_prefill_len", [512]),
        ]
        space.scheduler_options = [
            # num_scheduler_steps removed - not in vLLM 0.16.0
        ]
        space.quantization_options = [
            ConfigOption("quantization", [None, "fp8"]),
        ]
        space.speculative_options = [
            ConfigOption("speculative_model", [None, "ngram"]),  # Optional n-gram speculative
            ConfigOption("num_lookahead_slots", [1, 2],
                        conditional_on={"speculative_model": "ngram"}),
        ]
        space.all_options = (
            space.scheduling_options +
            space.chunked_prefill_options +
            space.scheduler_options +
            space.quantization_options +
            space.speculative_options
        )
        return space
