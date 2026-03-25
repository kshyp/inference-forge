# Quantization SME Implementation Summary

## Overview

Implemented a comprehensive **QuantizationSME** with hardware-aware recommendations for vLLM inference optimization. The system detects GPU capabilities at startup and uses multi-LLM consensus to recommend optimal quantization strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HARDWARE DETECTION                            │
│  forge/hardware/                                                 │
│  ├── database.py      - GPU specs database (23 GPUs)            │
│  └── detector.py      - Runtime hardware detection              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COORDINATOR AGENT                               │
│  - Detects platform with full hardware capabilities              │
│  - Passes hardware_capabilities to all SMEs via profiling_data   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 QUANTIZATION SME                                 │
│  forge/smes/quantization_sme.py                                  │
│  ├── Multi-LLM consensus for recommendations                     │
│  ├── Hardware-aware decision logic                               │
│  └── KV cache quantization support                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              HUGGINGFACE MODEL FINDER                            │
│  forge/smes/quantization/hf_model_finder.py                      │
│  - Searches HF for pre-quantized models                          │
│  - Prefers trusted authors (TheBloke, etc.)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Hardware Capability Detection (`forge/hardware/`)

**GPU Database** (`database.py`):
- 23 GPU models across architectures: Turing, Ampere, Ada, Hopper
- Specs include: VRAM, memory bandwidth, L2 cache, compute capability
- Native precision support per architecture
- Quantization recommendations per GPU

**Runtime Detection** (`detector.py`):
- Uses pynvml for accurate detection
- Falls back to nvidia-smi CLI if pynvml unavailable
- Returns `HardwareCapabilities` dataclass with full specs

**Key Hardware Properties:**
| Property | Description |
|----------|-------------|
| `gpu_architecture` | Turing, Ampere, Ada, Hopper |
| `compute_capability` | SM version (e.g., 9.0 for Hopper) |
| `native_precisions` | FP8, INT8, FP16, BF16 supported natively |
| `recommended_weight_quant` | Ordered list: ["fp8", "awq", "gptq"] |
| `kv_cache_quant_supported` | ["fp8"] for Hopper/Ada, [] otherwise |

### 2. QuantizationSME (`forge/smes/quantization_sme.py`)

**Registration:**
- Registers on NVIDIA CUDA and AMD ROCm platforms
- Stores hardware capabilities for analyze phase
- Requires: system metrics, vLLM logs, benchmark metrics
- Optional: NCU report, NSYS report

**Analysis Flow:**
1. Receive profiling data + hardware capabilities
2. Build structured prompt for LLM analysis
3. Call LLM pool for multi-model consensus
4. Search HuggingFace for pre-quantized models
5. Return recommendations with confidence scores

**Decision Logic:**
```
IF memory_bandwidth_bound AND supports_fp8:
    → FP8 weights + FP8 KV cache
ELIF kv_cache_pressure AND supports_fp8:
    → FP8 KV cache only (keep FP16 weights)
ELIF gpu_memory_high AND NOT supports_fp8:
    → AWQ/GPTQ 4-bit
ELIF already_quantized AND supports_fp8:
    → Recommend KV cache quantization
```

**Supported Quantization Methods:**
| Method | Hardware | Use Case |
|--------|----------|----------|
| FP8 | Hopper/Ada | Memory bandwidth bound |
| AWQ | All Tensor Core | Accuracy-sensitive |
| GPTQ | All Tensor Core | Speed-critical |
| INT8 | All Tensor Core | General purpose |
| KV Cache FP8 | Hopper/Ada | KV cache pressure |

### 3. HuggingFace Model Finder (`forge/smes/quantization/hf_model_finder.py`)

**Features:**
- Searches HF Hub for pre-quantized models
- Detects quantization method from tags and name
- Ranks models by downloads, likes, author trust
- Preferred authors: TheBloke (GPTQ/AWQ), neuralmagic

**Integration:**
- Called automatically by QuantizationSME
- If pre-quantized model found: `config_changes["model"] = model_id`
- If not found: `config_changes["model_source"] = "on_the_fly"`

**Usage:**
```python
model_id = await find_best_quantized_model(
    base_model="meta-llama/Llama-2-7b-hf",
    recommendation="awq",
    bits=4
)
# Returns: "TheBloke/Llama-2-7B-AWQ" or None
```

### 4. KV Cache Quantization

**vLLM Configuration:**
```bash
# FP8 KV cache (Hopper/Ada only)
--kv-cache-dtype fp8

# With calibration
--kv-cache-dtype fp8 --calculate-kv-scales
```

**Recommendations:**
- Triggered by `kv_cache_pressure` signal (>85% usage)
- Only recommended on hardware with FP8 support
- Independent of weight quantization
- 50% KV cache memory reduction

## Data Flow

```python
# 1. Coordinator detects hardware
platform_info = {
    "type": "nvidia_cuda",
    "gpu_model": "H100",
    "hardware_capabilities": {...}  # Full capabilities
}

# 2. QuantizationSME receives data
profiling_data = {
    "hardware_capabilities": {...},  # From platform_info
    "vllm_logs": {...},
    "system_metrics_report": {...},
    "ncu_report": {...},
}

# 3. SME generates recommendation
response = SMEResponse(
    findings={
        "primary_bottleneck": "memory_bandwidth_bound",
        "hardware_supports_fp8": True,
        "recommended_target": "both"
    },
    suggestions=[{
        "config_changes": {
            "quantization": "fp8",
            "kv_cache_dtype": "fp8",
            "model": "neuralmagic/Llama-2-7b-FP8"  # Pre-quantized
        },
        "expected_improvement": "-50% memory, +40% throughput"
    }]
)
```

## Testing

**Simple Test (no LLM calls):**
```bash
python test_quantization_simple.py
```
Tests: Database, hardware detection, registration, fallback logic

**Full Test (with LLM):**
```bash
python test_quantization_sme.py --mock-hardware H100 --scenario baseline
```

**Mock Hardware Options:**
- `H100` - Hopper, supports FP8
- `A100` - Ampere, no FP8
- `RTX4090` - Ada, supports FP8

## Files Created

```
forge/hardware/
├── __init__.py           - Module exports
├── database.py           - GPU specs database (23 GPUs)
└── detector.py           - Hardware detection

forge/smes/quantization/
├── __init__.py           - Module exports
└── hf_model_finder.py    - HF model discovery

forge/smes/quantization_sme.py  - Updated with full implementation
test_quantization_simple.py      - Simple tests (no LLM)
test_quantization_sme.py         - Full integration tests
QUANTIZATION_SME_DESIGN.md       - This document
```

## Next Steps

1. **Install HuggingFace Hub** (optional):
   ```bash
   pip install huggingface_hub
   ```

2. **Run integration test**:
   ```bash
   python test_quantization_sme.py --mock-hardware H100
   ```

3. **Test with actual hardware**:
   ```bash
   # Detects actual GPU
   python test_quantization_sme.py
   ```

4. **Integrate with profiler**:
   - Profiler captures NSYS + system metrics
   - Coordinator passes to QuantizationSME
   - SME recommends quantization strategy
   - BenchmarkAgent applies recommendations

## GPU Support Matrix

| GPU | Arch | FP8 | KV FP8 | Recommended |
|-----|------|-----|--------|-------------|
| H100 | Hopper | ✅ | ✅ | FP8 weights + KV cache |
| H200 | Hopper | ✅ | ✅ | FP8 weights + KV cache |
| RTX 4090 | Ada | ✅ | ✅ | FP8 weights + KV cache |
| L40S | Ada | ✅ | ✅ | FP8 weights + KV cache |
| A100 | Ampere | ❌ | ❌ | AWQ/GPTQ 4-bit |
| A10 | Ampere | ❌ | ❌ | AWQ/GPTQ 4-bit |
| RTX 3090 | Ampere | ❌ | ❌ | AWQ/GPTQ 4-bit |
| T4 | Turing | ❌ | ❌ | GPTQ/INT8 |
