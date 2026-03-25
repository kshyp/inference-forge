"""GPU specifications database for hardware capability detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum


class GPUArchitecture(Enum):
    """NVIDIA GPU architectures."""
    UNKNOWN = "unknown"
    TURING = "turing"      # SM 7.5, RTX 20xx, T4
    AMPERE = "ampere"      # SM 8.0/8.6, A100, RTX 30xx
    ADA = "ada"            # SM 8.9, RTX 4090, L40S
    HOPPER = "hopper"      # SM 9.0, H100, H200
    BLACKWELL = "blackwell"  # SM 10.0+, B100, B200 (future)


@dataclass
class GPUSpecs:
    """Specifications for a GPU model."""
    name: str
    architecture: GPUArchitecture
    compute_capability: tuple  # (major, minor)
    tensor_core_version: int
    
    # Memory specs
    vram_gb: float
    memory_bandwidth_gbps: float
    l2_cache_mb: float
    
    # Precision support (native hardware acceleration)
    native_precisions: Set[str]  # {"fp16", "bf16", "fp8", "int8", "int4", "tf32"}
    
    # Quantization recommendations (ordered by preference)
    recommended_weight_quant: List[str]  # ["fp8", "awq", "gptq", "int8"]
    kv_cache_quant_supported: List[str]  # ["fp8"] or []
    
    # Peak compute (for reference)
    fp16_tflops: float
    fp8_tflops: Optional[float] = None
    int8_tops: Optional[float] = None


# Comprehensive GPU specifications database
# Data sources: NVIDIA whitepapers, product briefs, and microbenchmark studies
GPU_SPECS_DATABASE: Dict[str, GPUSpecs] = {
    # Hopper GPUs (SM 9.0) - Full FP8 support
    "H100": GPUSpecs(
        name="H100",
        architecture=GPUArchitecture.HOPPER,
        compute_capability=(9, 0),
        tensor_core_version=4,
        vram_gb=80.0,
        memory_bandwidth_gbps=3350,  # HBM3
        l2_cache_mb=50.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=1979,  # With sparsity
        fp8_tflops=3958,
        int8_tops=3958,
    ),
    "H100 NVL": GPUSpecs(
        name="H100 NVL",
        architecture=GPUArchitecture.HOPPER,
        compute_capability=(9, 0),
        tensor_core_version=4,
        vram_gb=94.0,
        memory_bandwidth_gbps=3900,
        l2_cache_mb=50.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=1979,
        fp8_tflops=3958,
        int8_tops=3958,
    ),
    "H200": GPUSpecs(
        name="H200",
        architecture=GPUArchitecture.HOPPER,
        compute_capability=(9, 0),
        tensor_core_version=4,
        vram_gb=141.0,
        memory_bandwidth_gbps=4900,  # HBM3e
        l2_cache_mb=50.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=1979,
        fp8_tflops=3958,
        int8_tops=3958,
    ),
    "H800": GPUSpecs(
        name="H800",
        architecture=GPUArchitecture.HOPPER,
        compute_capability=(9, 0),
        tensor_core_version=4,
        vram_gb=80.0,
        memory_bandwidth_gbps=3350,
        l2_cache_mb=50.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=1979,
        fp8_tflops=3958,
        int8_tops=3958,
    ),
    
    # Ada Lovelace GPUs (SM 8.9) - FP8 support
    "RTX 4090": GPUSpecs(
        name="RTX 4090",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=24.0,
        memory_bandwidth_gbps=1008,
        l2_cache_mb=72.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=330,  # With sparsity
        fp8_tflops=660,
        int8_tops=660,
    ),
    "RTX 4080": GPUSpecs(
        name="RTX 4080",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=16.0,
        memory_bandwidth_gbps=717,
        l2_cache_mb=64.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=240,
        fp8_tflops=480,
        int8_tops=480,
    ),
    "RTX 4080 SUPER": GPUSpecs(
        name="RTX 4080 SUPER",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=16.0,
        memory_bandwidth_gbps=736,
        l2_cache_mb=64.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=260,
        fp8_tflops=520,
        int8_tops=520,
    ),
    "L40S": GPUSpecs(
        name="L40S",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=48.0,
        memory_bandwidth_gbps=864,
        l2_cache_mb=48.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=366,
        fp8_tflops=733,
        int8_tops=733,
    ),
    "L40": GPUSpecs(
        name="L40",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=48.0,
        memory_bandwidth_gbps=864,
        l2_cache_mb=48.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=181,
        fp8_tflops=362,
        int8_tops=362,
    ),
    "L4": GPUSpecs(
        name="L4",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=24.0,
        memory_bandwidth_gbps=300,
        l2_cache_mb=24.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=60,
        fp8_tflops=120,
        int8_tops=120,
    ),
    "RTX 6000 Ada": GPUSpecs(
        name="RTX 6000 Ada",
        architecture=GPUArchitecture.ADA,
        compute_capability=(8, 9),
        tensor_core_version=4,
        vram_gb=48.0,
        memory_bandwidth_gbps=960,
        l2_cache_mb=48.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "fp8", "int8"},
        recommended_weight_quant=["fp8", "awq", "gptq", "int8"],
        kv_cache_quant_supported=["fp8"],
        fp16_tflops=364,
        fp8_tflops=728,
        int8_tops=728,
    ),
    
    # Ampere GPUs (SM 8.0/8.6) - No FP8, but has BF16 and INT8
    "A100": GPUSpecs(
        name="A100",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 0),
        tensor_core_version=3,
        vram_gb=80.0,
        memory_bandwidth_gbps=2039,  # HBM2e
        l2_cache_mb=40.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],  # No FP8
        kv_cache_quant_supported=[],  # No native FP8 support
        fp16_tflops=624,  # With sparsity
        fp8_tflops=None,
        int8_tops=1248,
    ),
    "A100 40GB": GPUSpecs(
        name="A100 40GB",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 0),
        tensor_core_version=3,
        vram_gb=40.0,
        memory_bandwidth_gbps=1555,
        l2_cache_mb=40.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=624,
        fp8_tflops=None,
        int8_tops=1248,
    ),
    "A10": GPUSpecs(
        name="A10",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=24.0,
        memory_bandwidth_gbps=600,
        l2_cache_mb=24.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=125,
        fp8_tflops=None,
        int8_tops=250,
    ),
    "A10G": GPUSpecs(
        name="A10G",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=24.0,
        memory_bandwidth_gbps=600,
        l2_cache_mb=24.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=125,
        fp8_tflops=None,
        int8_tops=250,
    ),
    "A30": GPUSpecs(
        name="A30",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 0),
        tensor_core_version=3,
        vram_gb=24.0,
        memory_bandwidth_gbps=933,
        l2_cache_mb=24.0,
        native_precisions={"fp64", "fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=165,
        fp8_tflops=None,
        int8_tops=330,
    ),
    "RTX 3090": GPUSpecs(
        name="RTX 3090",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=24.0,
        memory_bandwidth_gbps=936,
        l2_cache_mb=6.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=142,
        fp8_tflops=None,
        int8_tops=284,
    ),
    "RTX 3090 Ti": GPUSpecs(
        name="RTX 3090 Ti",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=24.0,
        memory_bandwidth_gbps=1008,
        l2_cache_mb=6.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=160,
        fp8_tflops=None,
        int8_tops=320,
    ),
    "RTX 3080": GPUSpecs(
        name="RTX 3080",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=10.0,
        memory_bandwidth_gbps=760,
        l2_cache_mb=5.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=119,
        fp8_tflops=None,
        int8_tops=238,
    ),
    "RTX 3080 Ti": GPUSpecs(
        name="RTX 3080 Ti",
        architecture=GPUArchitecture.AMPERE,
        compute_capability=(8, 6),
        tensor_core_version=3,
        vram_gb=12.0,
        memory_bandwidth_gbps=912,
        l2_cache_mb=6.0,
        native_precisions={"fp32", "tf32", "fp16", "bf16", "int8"},
        recommended_weight_quant=["awq", "gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=136,
        fp8_tflops=None,
        int8_tops=272,
    ),
    
    # Turing GPUs (SM 7.5) - Limited support
    "T4": GPUSpecs(
        name="T4",
        architecture=GPUArchitecture.TURING,
        compute_capability=(7, 5),
        tensor_core_version=2,
        vram_gb=16.0,
        memory_bandwidth_gbps=320,
        l2_cache_mb=4.0,
        native_precisions={"fp32", "fp16", "int8", "int4"},  # No BF16
        recommended_weight_quant=["gptq", "int8"],  # AWQ less optimal
        kv_cache_quant_supported=[],
        fp16_tflops=65,
        fp8_tflops=None,
        int8_tops=130,
    ),
    "RTX 2080 Ti": GPUSpecs(
        name="RTX 2080 Ti",
        architecture=GPUArchitecture.TURING,
        compute_capability=(7, 5),
        tensor_core_version=2,
        vram_gb=11.0,
        memory_bandwidth_gbps=616,
        l2_cache_mb=5.5,
        native_precisions={"fp32", "fp16", "int8", "int4"},
        recommended_weight_quant=["gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=57,
        fp8_tflops=None,
        int8_tops=114,
    ),
    "RTX 2060": GPUSpecs(
        name="RTX 2060",
        architecture=GPUArchitecture.TURING,
        compute_capability=(7, 5),
        tensor_core_version=2,
        vram_gb=6.0,
        memory_bandwidth_gbps=336,
        l2_cache_mb=3.0,
        native_precisions={"fp32", "fp16", "int8", "int4"},
        recommended_weight_quant=["gptq", "int8"],
        kv_cache_quant_supported=[],
        fp16_tflops=26,
        fp8_tflops=None,
        int8_tops=52,
    ),
}


def get_gpu_specs(gpu_model: str) -> Optional[GPUSpecs]:
    """Get specifications for a GPU model.
    
    Args:
        gpu_model: GPU model name (e.g., "A100", "RTX 4090")
        
    Returns:
        GPUSpecs if found, None otherwise
    """
    # Try exact match first
    if gpu_model in GPU_SPECS_DATABASE:
        return GPU_SPECS_DATABASE[gpu_model]
    
    # Try case-insensitive match
    for key, specs in GPU_SPECS_DATABASE.items():
        if key.upper() == gpu_model.upper():
            return specs
    
    # Try substring match (for partial names like "A100-SXM4-80GB" -> "A100")
    for key, specs in GPU_SPECS_DATABASE.items():
        if key.upper() in gpu_model.upper():
            return specs
    
    return None


def get_compute_capability(arch: GPUArchitecture) -> tuple:
    """Get default compute capability for an architecture."""
    mapping = {
        GPUArchitecture.TURING: (7, 5),
        GPUArchitecture.AMPERE: (8, 0),
        GPUArchitecture.ADA: (8, 9),
        GPUArchitecture.HOPPER: (9, 0),
        GPUArchitecture.BLACKWELL: (10, 0),
    }
    return mapping.get(arch, (0, 0))


def supports_fp8(arch: GPUArchitecture) -> bool:
    """Check if architecture supports native FP8."""
    return arch in (GPUArchitecture.ADA, GPUArchitecture.HOPPER, GPUArchitecture.BLACKWELL)


def supports_bf16(arch: GPUArchitecture) -> bool:
    """Check if architecture supports BF16."""
    return arch in (
        GPUArchitecture.AMPERE, 
        GPUArchitecture.ADA, 
        GPUArchitecture.HOPPER,
        GPUArchitecture.BLACKWELL
    )
