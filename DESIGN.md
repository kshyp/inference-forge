# Inference Forge
## VLLM Optimization Swarm - Design Document

**Version:** 1.4  
**Last Updated:** 2026-03-16  
**Status:** All Phases Complete - End-to-End Workflow Verified ✅

---

## 1. Overview

Inference Forge is a multi-agent swarm for autonomously optimizing vLLM inference configurations. The system runs benchmarks, profiles performance, analyzes bottlenecks via specialized expert agents (SMEs), and iteratively improves configurations.

### Key Design Principles

1. **Simplicity First** - No hot-reload, no complex discovery. Filesystem + restart.
2. **Agent Specialization** - Each agent has one clear responsibility
3. **SME Plugin Architecture** - Experts are Python code, loaded at startup
4. **Stateful & Recoverable** - SQLite checkpointing for crash recovery
5. **Observable** - Health endpoints for monitoring
6. **Deterministic** - Reproducible LLM outputs for consistent recommendations

---

## 2. Architecture

### 2.1 Agent Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE FORGE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│   │  AGENT 1    │───►│  AGENT 2    │───│    AGENT 3          │ │
│   │  Benchmark  │    │  Profile    │    │  Coordinator        │ │
│   │  Runner     │    │  Executor   │    │  (SME Consultation) │ │
│   │             │    │             │    │                     │ │
│   │ I/O Agent   │    │ I/O Agent   │    │ BRAIN Agent         │ │
│   │ • No LLM    │    │ • No LLM    │    │ • Uses LLM for:     │ │
│   │ • Pure exec │    │ • Pure exec │    │   - Triage          │ │
│   │             │    │             │    │   - SME selection   │ │
│   └──────▲──────┘    └─────────────┘    │   - Synthesis       │ │
│          │                               │   - Ranking         │ │
│          │                               └──────────┬──────────┘ │
│          │                                          │            │
│          │    ┌─────────────────────────────────────┘            │
│          │    │ (Loop: Experiment Plan → Agent 1)                │
│          │    │                                                   │
│          │    │  ┌─────────────────────────────────────────────┐ │
│          │    └──┤  SME COUNCIL (Consulted by Coordinator)     │ │
│          │       │                                              │ │
│          │       │  • quantization (Python class)               │ │
│          │       │  • scheduling (multi-LLM consensus)          │ │
│          │       │  • speculative_decoding                      │ │
│          │       │  • memory_kv_cache                           │ │
│          │       │  • [Future: model_parallelism, etc.]         │ │
│          │       │                                              │ │
│          │       │  Auto-discovered from forge/smes/            │ │
│          │       └──────────────────────────────────────────────┘ │
│          │                                                      │
│          └──────────────────────────────────────────────────────┘
│                                                                  │
│   STATE: SQLite (data/state.db)                                  │
│   • Checkpoints for crash recovery                               │
│   • Experiment history                                           │
│   • Event log                                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Responsibilities

| Agent | Type | Status | Responsibility | vLLM Logging | LLM Usage | Checkpoint Strategy |
|-------|------|--------|----------------|--------------|-----------|---------------------|
| **Agent 1: Benchmark Runner** | I/O | ✅ Complete | Sweep to find saturation point (clean, low overhead) | INFO level | None | GuideLLM sweep progress |
| **Agent 2: Profile Executor** | I/O | ✅ Complete | Steady-state run AT saturation with DEBUG logging + profilers | DEBUG level | None | Profilers completed, vLLM state |
| **Agent 3: Coordinator** | Brain | ✅ Complete | Platform discovery, SME registration, triage, synthesis, ranking | N/A | Heavy | SME responses, final plan |

**Key Separation:**
- Agent 1 finds saturation with **INFO** logging (minimal overhead on benchmark)
- Agent 2 profiles at saturation with **DEBUG** logging (detailed data for SMEs)
- Agent 2 can attach NCU/NSys profilers without skewing Agent 1's benchmark results

**See:** `forge/agents/profile/design.md` for detailed Profiler Agent architecture

### 2.3 Data Flow

```
ITERATION CYCLE:

1. Agent 1 (Benchmark)
   Input:  vLLM command + config flags
   Action: Run vLLM with INFO logging (low overhead) → GuideLLM sweep → Find saturation point
   Output: SaturationPoint (RPS where latency spikes) + minimal metrics
   → Stop vLLM → Handoff to Agent 2

2. Agent 2 (Profile)
   Input:  SaturationPoint + config flags + required_data_types (from SME Registry)
   Action: 
     a. Start NEW vLLM with DEBUG logging (higher overhead, detailed logs)
     b. Run steady-state benchmark AT saturation rate
     c. Attach profilers (NCU, NSys) during run
     d. Collect debug logs + profiler reports
     e. Stop vLLM
   Output: ProfilingReports (paths to reports: vllm_logs_debug, ncu_report, nsys_report)
   → Handoff to Agent 3

3. Agent 3 (Coordinator)
   Input:  ProfilingReports + SaturationPoint
   Action: 
     a. Extract signals (memory pressure, GPU util, etc.) from DEBUG logs
     b. Select relevant SMEs from registry (based on triggers)
     c. Consult SMEs in parallel (call SME.analyze() for each)
     d. Synthesize opinions → Rank experiments
   Output: ExperimentPlan (ranked list of config changes)
   → Handoff to Agent 1 (loop)

KEY ARCHITECTURAL PRINCIPLE:
- Agent 1 (Benchmark): Clean, low-overhead sweep to find saturation
- Agent 2 (Profile): High-overhead profiling at saturation point with DEBUG logging
- Separation ensures benchmark results aren't skewed by profiling overhead

STOP CONDITIONS:
- No improvement after N iterations
- Config converges (same flags suggested)
- Max iterations reached
- User interrupt
```

---

## 3. SME (Subject Matter Expert) System

### 3.1 SME Architecture

SMEs are **Python classes** (code, not just data) that:
1. **Register** with platform-specific data requirements
2. **Analyze** profiling data and suggest experiments

```
forge/smes/
├── base.py                    # BaseSME abstract class
├── quantization_sme.py        # Quantization expert
├── scheduling_sme.py          # Scheduling expert with multi-LLM consensus
├── speculative_sme.py         # Speculative decoding expert
├── model_parallelism_sme.py   # TP/PP/PD disaggregation expert
├── cpu_optimizations_sme.py   # CPU-only expert
└── __init__.py                # Exports all SME classes
```

### 3.2 BaseSME Interface

```python
class BaseSME(ABC):
    """Base class for Subject Matter Experts."""
    
    @abstractmethod
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        """
        Register with Coordinator for this platform.
        Returns RegistrationInfo if applicable, None to skip.
        """
        pass
    
    @abstractmethod
    def analyze(self, profiling_data: Dict[str, Any],
                benchmark_metrics: Dict[str, Any]) -> SMEResponse:
        """Analyze data and return findings + experiment suggestions."""
        pass
```

### 3.3 Key Dataclasses

```python
@dataclass
class DataRequirement:
    """Data required from the profiler."""
    data_type: str       # "ncu_report", "vllm_logs", "benchmark_metrics"
    required: bool       # True = must have, False = optional
    extractors: List[str]  # Specific metrics to extract

@dataclass
class RegistrationInfo:
    """Returned when SME registers successfully."""
    sme_id: str
    triggers: List[str]                   # Signals this SME handles
    data_requirements: List[DataRequirement]  # What data it needs

@dataclass
class ExperimentSuggestion:
    """A suggested configuration experiment."""
    config_changes: Dict[str, Any]
    expected_improvement: str
    confidence: float
    rationale: str

@dataclass
class SMEResponse:
    """Return value from SME.analyze()."""
    findings: Dict[str, Any]
    suggestions: List[ExperimentSuggestion]
    confidence: float
```

### 3.4 SME AI Brain (Multi-LLM Consensus)

Each SME has an **AI Brain** powered by multi-LLM consensus:

```
┌─────────────────────────────────────────────────────────────┐
│  SME Brain (e.g., SchedulingSME)                             │
│                                                              │
│  1. Prepare context (structured prompt with ALL metrics)     │
│                                                              │
│  2. Parallel invocation with deterministic mode:             │
│     ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐     │
│     │  Claude     │ │   GPT-4     │ │  Moonshot       │     │
│     │  temp=0     │ │   temp=0    │ │  temp=0         │     │
│     │  seed=42    │ │   seed=42   │ │  seed=42        │     │
│     └──────┬──────┘ └──────┬──────┘ └────────┬────────┘     │
│            └───────────────┼─────────────────┘               │
│                            ▼                                 │
│  3. ConsensusEngine.compute()                                │
│     - Parse JSON from each response                          │
│     - Cluster similar suggestions (threshold=0.8)            │
│     - Score by: confidence × agreement × quality             │
│                                                              │
│  4. Return SMEResponse with consensus suggestions            │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- **Robustness**: Single model failure doesn't break the system
- **Consensus as signal**: Agreement = higher confidence suggestion
- **Explainability**: Each model provides reasoning; divergence is visible
- **Extensibility**: Add new providers by setting env var
- **Determinism**: `temperature=0.0` ensures reproducible results

**Configuration:**
```yaml
# Auto-discovery from environment variables
intelligence:
  enabled_providers:
    - anthropic     # ANTHROPIC_API_KEY
    - openai        # OPENAI_API_KEY
    - moonshot      # MOONSHOT_API_KEY
    - gemini        # GEMINI_API_KEY
    - vllm_local    # VLLM_LOCAL_URL
```

**LLM Client Architecture:**
```
forge/llm/
├── base.py           # LLMClient base class + provider implementations
├── pool.py           # IntelligencePool for parallel calls
└── consensus.py      # ConsensusEngine for aggregating responses
```

### 3.5 Tiered Metrics Architecture

SMEs use a **Tier 1 + Tier 2** metrics system:

**Tier 1 (Base Metrics - Always Collected):**
- `system_metrics_report` - nvidia-smi GPU metrics (no root needed)
- `vmstat_report` - CPU, memory, IO stats
- `mpstat_report` - Per-CPU utilization
- `vllm_logs` - vLLM DEBUG logs

**Tier 2 (Deep Profiling - Optional, Requires Root):**
- `ncu_report` - NVIDIA Compute Profiler (compute/memory bandwidth)
- `nsys_report` - NVIDIA Systems Profiler (timeline, kernel gaps)

**Benefits:**
- Tier 1 provides foundation for all analysis (no permissions needed)
- Tier 2 adds detail when available (requires root/CUDA toolkit)
- SMEs gracefully handle missing Tier 2 data

### 3.6 Platform-Aware Registration

Each SME's `register()` method decides:
1. **Whether to register** at all (e.g., skip if single GPU for ModelParallelismSME)
2. **What data to request** based on platform (different profilers for CUDA vs ROCm)

**Example: Platform-conditional Registration**

```python
class QuantizationSME(BaseSME):
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        platform_type = platform_info.get("type", "")
        
        if platform_type == "nvidia_cuda":
            return RegistrationInfo(
                sme_id="quantization",
                triggers=["gpu_memory_high", "memory_bandwidth_bound"],
                data_requirements=[
                    DataRequirement("ncu_report", required=True, 
                                  extractors=["memory_bandwidth_utilization", ...]),
                    DataRequirement("vllm_logs", required=True, 
                                  extractors=["current_quantization", "model_dtype"]),
                    DataRequirement("nsys_report", required=False, ...),
                ]
            )
        
        elif platform_type == "amd_rocm":
            return RegistrationInfo(
                sme_id="quantization",
                triggers=["gpu_memory_high", "memory_bandwidth_bound"],
                data_requirements=[
                    DataRequirement("rocprof_report", required=True, ...),
                    DataRequirement("vllm_logs", required=True, ...),
                ]
            )
        
        # CPU or other platforms - not applicable
        return None
```

**Example: GPU-count Gate**

```python
class ModelParallelismSME(BaseSME):
    def register(self, platform_info: Dict[str, Any]) -> Optional[RegistrationInfo]:
        if platform_info.get("gpu_count", 1) <= 1:
            print(f"[ModelParallelismSME] Skipping: only 1 GPU. Need 2+ for TP/PP.")
            return None  # Skip registration
        
        return RegistrationInfo(
            sme_id="model_parallelism",
            triggers=["multi_gpu", "high_inter_gpu_traffic"],
            data_requirements=[...]
        )
```

### 3.7 SME Registry

The `SMERegistry` coordinates SME registration and data collection:

```python
class SMERegistry:
    def discover_sme_classes(self) -> List[Type[BaseSME]]:
        """Auto-discover all SME classes in forge.smes package."""
        
    def register_all(self, platform_info: Dict[str, Any]) -> Dict[str, RegistrationInfo]:
        """Register all applicable SMEs for this platform."""
        
    def get_profiler_requirements(self) -> Dict[str, Dict[str, Any]]:
        """
        Get consolidated requirements for Agent 2 (Profiler).
        Returns: {
            "ncu_report": {"required": True, "extractors": [...]},
            "vllm_logs": {"required": True, "extractors": [...]},
            ...
        }
        """
        
    def consult(self, signals: List[str], profiling_data: Dict[str, Any],
                benchmark_metrics: Dict[str, Any]) -> List[SMEResponse]:
        """Consult SMEs that match the given signals."""
```

### 3.8 SME Registration Flow

```
1. Coordinator detects platform → platform_info
2. Registry discovers SME classes (auto-scan forge.smes)
3. For each SME class:
   a. Instantiate SME
   b. Call SME.register(platform_info)
   c. If returns RegistrationInfo → registered
   d. If returns None → skipped (not applicable)
4. Registry aggregates data requirements from all registered SMEs
5. Pass consolidated requirements to Agent 2 (Profiler)
```

### 3.9 Coordinator → SME Consultation

```python
# Coordinator extracts signals from profiling data
signals = ["gpu_memory_high", "low_gpu_utilization", "high_ttft"]

# Registry finds SMEs that can address these signals
matched_smes = registry.consult(signals, profiling_data, benchmark_metrics)

# Each SME returns findings and experiment suggestions
for response in matched_smes:
    for suggestion in response.suggestions:
        # Synthesize and rank suggestions
        experiments_to_try.append(suggestion)
```

### 3.10 Implemented SMEs

| SME | File | Platforms | Required Data | Triggers | Responsibility |
|-----|------|-----------|---------------|----------|----------------|
| **quantization** | `quantization_sme.py` | nvidia_cuda, amd_rocm | ncu_report/rocprof_report, vllm_logs | memory pressure, bandwidth bound | Recommend AWQ/GPTQ/FP8 |
| **scheduling** | `scheduling_sme.py` | * (all) | vllm_logs, benchmark_metrics, system_metrics | low GPU util, queue buildup, high TTFT | Batch size, chunked prefill |
| **model_parallelism** | `model_parallelism_sme.py` | nvidia_cuda, amd_rocm | ncu_report, nsys_report, vllm_logs | multi_gpu, high_inter_gpu_traffic | TP/PP/PD disaggregation |
| **speculative** | `speculative_sme.py` | nvidia_cuda, amd_rocm | vllm_logs, benchmark_metrics, ncu_report | slow decode, low acceptance rate | Draft model tuning |
| **cpu_optimizations** | `cpu_optimizations_sme.py` | cpu | cpu_profile_report, system_metrics, vllm_logs | cpu_low_util, numa_remote_access | Threading, NUMA, int8 |

**Future SMEs** (post-MVP):
- attention_optimization (FlashInfer, custom kernels)
- power_efficiency (GPU power capping, thermal management)

---

## 4. State Management & Checkpointing

### 4.1 SQLite Schema

```sql
-- Agent checkpoints for crash recovery
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    checkpoint_data TEXT NOT NULL,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, task_id)
);

-- Experiment history
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    iteration INTEGER NOT NULL,
    parent_experiment_id TEXT,
    config_flags TEXT NOT NULL,        -- JSON
    benchmark_results TEXT,            -- JSON
    profiling_reports TEXT,            -- JSON list of paths
    expert_opinions TEXT,              -- JSON
    final_recommendations TEXT,        -- JSON
    status TEXT,                       -- running, completed, failed, converged
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Event log (for debugging/replay)
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    agent_id TEXT,
    task_id TEXT,
    payload TEXT,                      -- JSON
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2 Checkpoint Strategy by Agent

**Agent 1 (Benchmark):**
```python
# During GuideLLM sweep
await checkpoint({
    "sweep_progress": {
        "completed_rps_levels": [10, 20, 50],
        "current_rps": 100,
        "partial_metrics": {...}
    },
    "vllm_server_pid": 12345
})
```

**Agent 2 (Profile):**
```python
# After each profiler
await checkpoint({
    "completed_profilers": ["nsys", "ncu"],
    "pending_profilers": ["pytorch_profiler"],
    "report_paths": {
        "nsys": "/data/.../profile.nsys-rep",
        "ncu": "/data/.../profile.ncu-rep"
    }
})
```

**Agent 3 (Coordinator):**
```python
# After each SME consultation
await checkpoint({
    "signals_extracted": ["memory_pressure", "low_util"],
    "smes_consulted": ["quantization", "scheduling"],
    "sme_responses": {...},           # Raw LLM outputs
    "synthesis_in_progress": True,
    "ranked_experiments": [...]       # Final output
})
```

### 4.3 Recovery Flow

```
Agent Restarts → Load checkpoint for current task (if any)
              → Resume from last saved state
              → If task completed in previous run, mark done
              → If corrupted, mark failed, alert operator
```

---

## 5. Health Monitoring

### 5.1 Health Endpoint

Each agent exposes a FastAPI endpoint:

```python
@app.get("/health")
async def health():
    return {
        "agent_id": agent.agent_id,
        "state": agent.state.value,  # idle, running, error
        "current_task": {
            "id": agent.current_task.id if agent.current_task else None,
            "type": agent.current_task.type if agent.current_task else None,
            "progress": agent.current_task.progress if agent.current_task else 0,
        },
        "uptime_seconds": time.time() - agent.start_time,
        "last_checkpoint": agent.last_checkpoint_time
    }
```

### 5.2 Simple Monitor (External)

```bash
# scripts/health_check.sh
for agent in benchmark profile coordinator; do
    response=$(curl -s http://localhost:$PORT/health || echo "DOWN")
    if [ "$response" = "DOWN" ]; then
        echo "$agent is DOWN, restarting..."
        systemctl restart inference-forge-$agent
    fi
done
```

---

## 6. Testing Strategy

### 6.1 Component Tests

```bash
# Test SME system
python test_sme_registry.py

# Test profiler agent
python test_profiler_agent_integration.py

# Test full agent workflow (mock mode)
python test_full_workflow.py
```

### 6.2 End-to-End Test

```bash
# Full convergence test with real vLLM
python test_convergence_loop.py

# Expected output:
# - Iteration 1: Baseline config, find saturation, get SME recommendations
# - Iteration 2: Apply recommendations, measure improvement
# - Comparison: Show actual vs expected improvement
```

### 6.3 Saturation Detection

The benchmark agent uses this logic to find saturation:

```python
# Sweep increasing rates until latency spikes
for rate in [10, 25, 50, 100, 150, 200, 300]:
    metrics = run_benchmark(rate)
    
    # Saturation = latency > 500ms OR throughput < input_rate * 0.75
    if metrics["latency_p99"] > 500 or metrics["throughput"] < rate * 0.75:
        print(f"Saturation detected at {rate} RPS")
        saturation_rate = rate
        break
```

---

## 7. Configuration

### 7.1 config.yaml

```yaml
# Inference Forge Configuration

# Model to optimize
model:
  name: "gpt2"  # or your model path
  max_model_len: 512
  dtype: "float16"

# Benchmark settings
benchmark:
  sweep_rates: [10, 25, 50, 100, 150, 200, 300]
  saturation_latency_threshold_ms: 500
  max_requests_per_benchmark: 300

# Profiling settings
profiling:
  duration_seconds: 60
  collect_base_metrics: true  # Tier 1: always collected
  collect_nsys: false         # Tier 2: requires root
  collect_ncu: false          # Tier 2: requires root + CUDA toolkit

# SME settings
smes:
  enabled:
    - quantization
    - scheduling
    - speculative
    - model_parallelism
    - cpu_optimizations
  
  # Multi-LLM consensus settings
  consensus:
    temperature: 0.0        # Deterministic outputs
    similarity_threshold: 0.8
    min_agreement_ratio: 0.5

# LLM Providers (auto-discovered from environment)
# Set these environment variables:
# - ANTHROPIC_API_KEY
# - OPENAI_API_KEY
# - MOONSHOT_API_KEY
# - GEMINI_API_KEY
# - VLLM_LOCAL_URL
```

---

## 8. Project Structure

```
inference-forge/
├── forge/                      # Main Python package
│   ├── core/                   # State, events, checkpointing
│   │   ├── state.py            # SQLite persistence
│   │   ├── events.py           # Event dataclasses
│   │   ├── checkpoint.py       # Checkpoint mixin
│   │   └── health.py           # Health server
│   ├── agents/                 # Agent implementations
│   │   ├── base.py             # BaseAgent abstract class
│   │   ├── benchmark/          # Agent 1: Benchmark Runner
│   │   │   └── agent.py
│   │   ├── profile/            # Agent 2: Profile Executor
│   │   │   ├── agent.py
│   │   │   ├── orchestrator.py
│   │   │   └── profilers/
│   │   └── coordinator/        # Agent 3: Coordinator
│   │       ├── agent.py
│   │       └── synthesis.py
│   ├── smes/                   # SME implementations
│   │   ├── base.py
│   │   ├── scheduling_sme.py   # Multi-LLM consensus
│   │   ├── quantization_sme.py
│   │   ├── speculative_sme.py
│   │   ├── model_parallelism_sme.py
│   │   └── cpu_optimizations_sme.py
│   ├── llm/                    # Multi-LLM consensus
│   │   ├── base.py             # LLM clients
│   │   ├── pool.py             # IntelligencePool
│   │   └── consensus.py        # ConsensusEngine
│   └── sme_registry.py         # SME auto-discovery
├── scripts/                    # Entry points
│   ├── start_forge.py
│   └── autotuner/              # vLLM/GuideLLM scripts
├── smes/                       # SME definitions (legacy)
├── data/                       # Runtime data (gitignored)
├── config.yaml                 # Configuration
├── pyproject.toml              # Package config
├── DESIGN.md                   # This file
└── PROGRESS.md                 # Implementation tracker
```

---

## 9. Known Limitations

1. **Small models don't show benefits**: gpt2 (124M params) is too small for chunked_prefill to help
   - Solution: Test with larger models (7B+) for meaningful results

2. **Uniform workloads**: Current wikitext dataset has uniform prompt lengths
   - Solution: Use heterogeneous workloads (short + long prompts) to test chunked_prefill

3. **No real NCU profiler**: Requires CUDA toolkit + root permissions
   - Solution: MockNCUProfiler provides synthetic data for testing

4. **Single-node only**: No distributed/multi-node support yet
   - Solution: ModelParallelismSME prepared for multi-GPU on single node

---

## 10. Next Steps

1. **Test with larger models**: Run convergence test with Llama-7B or similar
2. **Heterogeneous workloads**: Add mixed prompt length datasets
3. **Real NCU integration**: Implement actual NCU profiler when hardware available
4. **Health monitoring**: Add external health check scripts
5. **Web UI**: Optional dashboard for viewing experiments
