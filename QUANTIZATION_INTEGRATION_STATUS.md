# Quantization SME Integration Status

## ✅ Integration Complete

The QuantizationSME is now fully integrated with the Inference Forge system. When you run `scripts/start_forge.py`, the quantization SME will:

1. **Register at startup** - Coordinator detects hardware and registers all SMEs including QuantizationSME
2. **Trigger on signals** - When profiling data indicates quantization opportunities
3. **Generate recommendations** - Using multi-LLM consensus with hardware awareness
4. **Queue experiments** - Adds quantized configurations to the experiment backlog
5. **Execute benchmarks** - BenchmarkAgent runs each experiment and tracks improvements

## 🔄 Complete Data Flow

```
scripts/start_forge.py
    ↓
InferenceEngine.run()
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ITERATION LOOP                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Phase 1: BENCHMARK                                          │
│   - Run vLLM with current config                            │
│   - Collect throughput, latency metrics                     │
│                                                             │
│ Phase 2: PROFILE                                            │
│   - Run vLLM with DEBUG logs                                │
│   - Collect NSYS trace                                      │
│   - Collect system metrics (NEW) ← Required by QuantizationSME│
│   - Analyze NSYS with LLM                                   │
│                                                             │
│ Phase 3: COORDINATE                                         │
│   ├─ Register SMEs (first iteration)                        │
│   │     └─ QuantizationSME.register()                       │
│   │            └─ Detects: FP8 support, VRAM, architecture  │
│   │                                                         │
│   └─ Analyze                                                │
│         ├─ Extract signals:                                 │
│         │    ├─ memory_bandwidth_bound (>80%)               │
│         │    ├─ kv_cache_pressure (>85%)                    │
│         │    ├─ gpu_memory_high                             │
│         │    └─ baseline_exploration (always)               │
│         │                                                   │
│         ├─ Consult SMEs                                     │
│         │    └─ QuantizationSME.analyze()                   │
│         │         ├─ Build prompt with:                     │
│         │         │    ├─ hardware_capabilities             │
│         │         │    ├─ system_metrics_report             │
│         │         │    ├─ vllm_logs                         │
│         │         │    ├─ nsys_report                       │
│         │         │    └─ benchmark_metrics                 │
│         │         │                                         │
│         │         ├─ Multi-LLM consensus                    │
│         │         │    ├─ Call all LLM sources              │
│         │         │    ├─ Compute consensus                 │
│         │         │    └─ Generate suggestions              │
│         │         │                                         │
│         │         └─ Search HF for quantized models         │
│         │              ├─ Try pre-quantized first           │
│         │              └─ Fall back to on-the-fly           │
│         │                                                   │
│         └─ Synthesize experiment plan                       │
│              └─ Rank experiments by consensus               │
│                                                             │
│ Phase 4: EXECUTE BACKLOG                                    │
│   ├─ Get next experiment from backlog                       │
│   ├─ Apply config changes                                   │
│   │    └─ quantization: fp8/awq/gptq                        │
│   │    └─ kv_cache_dtype: fp8 (if applicable)               │
│   │    └─ model: pre-quantized HF model (if found)          │
│   │                                                         │
│   ├─ Run benchmark with new config                          │
│   └─ Compare results vs baseline                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    ↓
Check convergence → Repeat or Exit
```

## 📊 What Triggers Quantization Recommendations

| Signal | Threshold | QuantizationSME Trigger |
|--------|-----------|------------------------|
| `baseline_exploration` | Always (first iteration) | ✅ Always runs |
| `memory_bandwidth_bound` | NCU memory BW > 80% | ✅ FP8 weights (if HW supports) |
| `kv_cache_pressure` | KV cache usage > 85% | ✅ FP8 KV cache (if HW supports) |
| `gpu_memory_high` | GPU memory > 80% | ✅ AWQ/GPTQ 4-bit |
| `low_throughput` | Throughput < 30 req/s | ✅ Explore quantization |

## 🎮 Hardware-Specific Recommendations

### Hopper (H100/H200)
- **FP8 weights** - Native hardware support
- **FP8 KV cache** - 50% memory reduction
- Expected: -50% memory bandwidth, +40% throughput

### Ada (RTX 4090/L40S)
- **FP8 weights** - Native hardware support  
- **FP8 KV cache** - 50% memory reduction
- Expected: -50% memory bandwidth, +30% throughput

### Ampere (A100/A10/RTX 3090)
- **AWQ/GPTQ 4-bit** - No FP8 support
- No KV cache quantization
- Expected: -50% model memory, +20% throughput

### Turing (T4/RTX 2080)
- **GPTQ/INT8** - Limited quantization support
- Expected: -30% model memory

## 🧪 Testing the Integration

### Quick Test (No LLM Calls)
```bash
python test_quantization_simple.py
```
Verifies: Hardware detection, registration, fallback logic

### Full Test (With Mock Hardware)
```bash
python test_quantization_sme.py --mock-hardware H100 --scenario baseline
```
Verifies: Full LLM consensus, HF model search, recommendations

### Live Test (Actual Hardware)
```bash
python scripts/start_forge.py \
    --model microsoft/Phi-3-mini-4k-instruct \
    --max-iterations 5
```

## 📁 Files Modified/Created

### New Files
```
forge/hardware/
├── __init__.py           # Module exports
├── database.py           # GPU specs database (23 GPUs)
└── detector.py           # Hardware detection at runtime

forge/smes/quantization/
├── __init__.py           # Module exports
└── hf_model_finder.py    # HuggingFace model discovery

test_quantization_simple.py    # Simple tests
test_quantization_sme.py       # Full integration tests
QUANTIZATION_INTEGRATION_STATUS.md  # This file
```

### Modified Files
```
forge/smes/quantization_sme.py      # Full implementation with consensus
forge/agents/coordinator/agent.py   # Hardware detection integration
forge/agents/profile/agent.py       # System metrics collection
forge/orchestrator.py               # Pass system metrics to coordinator
```

## ⚙️ Configuration Options

The QuantizationSME will automatically recommend based on hardware, but you can observe its behavior in the logs:

```
[QuantizationSME] Calling LLM pool (sources: 5)...
[QuantizationSME] Got 5 call results
   ✓ anthropic/claude-sonnet-4-20250514
   ✓ anthropic/claude-haiku-4-5-20251001
   ...
[QuantizationSME] Computing consensus...
[QuantizationSME] Consensus: 2 suggestions
   → {'quantization': 'fp8', 'kv_cache_dtype': 'fp8'} (confidence: 0.85)
   → {'quantization': 'awq'} (confidence: 0.75)
[QuantizationSME] Searching HF for fp8 model...
   ✓ Found: neuralmagic/Phi-3-mini-FP8
```

## 🎯 Expected Behavior

1. **First Iteration**: Baseline + QuantizationSME triggered by `baseline_exploration`
2. **Subsequent Iterations**: If signals detected (memory pressure, KV cache pressure), quantization recommendations added
3. **Backlog Execution**: Each quantization config is tested in sequence
4. **Convergence**: System stops when best quantization strategy is found

## 📝 Notes

- **HuggingFace Hub**: Optional dependency. Install with `pip install huggingface_hub` to enable pre-quantized model discovery
- **FP8 KV Cache**: Only on Hopper/Ada GPUs (H100, H200, RTX 4090, L40S)
- **On-the-fly Quantization**: vLLM supports FP8, AWQ, GPTQ quantization flags
- **Model Changes**: If a pre-quantized model is found, the experiment will change the `model` parameter

## ✅ Verification Checklist

- [x] Hardware detection at startup
- [x] QuantizationSME registration
- [x] Signal extraction (memory_bandwidth_bound, kv_cache_pressure)
- [x] Multi-LLM consensus for recommendations
- [x] System metrics collection in profiler
- [x] Hardware capabilities passed to all SMEs
- [x] Experiment backlog integration
- [x] HuggingFace model discovery
- [x] KV cache quantization support
