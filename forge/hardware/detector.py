"""Hardware capability detection using pynvml and system queries."""

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from .database import (
    GPU_SPECS_DATABASE, 
    GPUSpecs, 
    GPUArchitecture,
    get_gpu_specs,
    supports_fp8,
    supports_bf16,
)

logger = logging.getLogger(__name__)


@dataclass
class HardwareCapabilities:
    """Complete hardware capabilities for a detected system."""
    
    # GPU identification
    gpu_model: str
    gpu_count: int
    gpu_architecture: GPUArchitecture
    compute_capability: tuple  # (major, minor)
    
    # Tensor cores
    has_tensor_cores: bool
    tensor_core_version: int
    
    # Memory
    vram_gb: float
    vram_gb_per_gpu: float
    memory_bandwidth_gbps: float
    l2_cache_mb: float
    
    # Precision support (native hardware)
    native_precisions: Set[str] = field(default_factory=set)
    
    # Quantization recommendations (ordered by preference)
    recommended_weight_quant: List[str] = field(default_factory=list)
    kv_cache_quant_supported: List[str] = field(default_factory=list)
    
    # Compute specs
    fp16_tflops: Optional[float] = None
    fp8_tflops: Optional[float] = None
    int8_tops: Optional[float] = None
    
    # System info
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "gpu_architecture": self.gpu_architecture.value,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "has_tensor_cores": self.has_tensor_cores,
            "tensor_core_version": self.tensor_core_version,
            "vram_gb": self.vram_gb,
            "vram_gb_per_gpu": self.vram_gb_per_gpu,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "l2_cache_mb": self.l2_cache_mb,
            "native_precisions": sorted(self.native_precisions),
            "recommended_weight_quant": self.recommended_weight_quant,
            "kv_cache_quant_supported": self.kv_cache_quant_supported,
            "fp16_tflops": self.fp16_tflops,
            "fp8_tflops": self.fp8_tflops,
            "int8_tops": self.int8_tops,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "supports_fp8": "fp8" in self.native_precisions,
            "supports_bf16": "bf16" in self.native_precisions,
        }
    
    def supports_quantization(self, method: str) -> bool:
        """Check if hardware supports a specific quantization method."""
        method = method.lower()
        
        if method == "fp8":
            return "fp8" in self.native_precisions
        elif method in ("awq", "gptq", "int8", "int4"):
            # These work on all Tensor Core GPUs
            return self.has_tensor_cores
        elif method == "smoothquant":
            return "int8" in self.native_precisions
        
        return False
    
    def get_optimal_weight_quant(self) -> Optional[str]:
        """Get the optimal weight quantization method for this hardware."""
        if self.recommended_weight_quant:
            return self.recommended_weight_quant[0]
        return None
    
    def can_quantize_kv_cache(self) -> bool:
        """Check if KV cache quantization is supported."""
        return len(self.kv_cache_quant_supported) > 0


class HardwareDetector:
    """Detect hardware capabilities at runtime."""
    
    def __init__(self):
        self._pynvml_available = self._check_pynvml()
        
    def _check_pynvml(self) -> bool:
        """Check if pynvml is available."""
        try:
            import pynvml
            return True
        except ImportError:
            logger.warning("pynvml not available, using fallback detection")
            return False
    
    def detect(self) -> HardwareCapabilities:
        """Detect hardware capabilities.
        
        Returns:
            HardwareCapabilities for the detected system
        """
        if self._pynvml_available:
            return self._detect_with_pynvml()
        else:
            return self._detect_fallback()
    
    def _detect_with_pynvml(self) -> HardwareCapabilities:
        """Detect using pynvml (NVIDIA Management Library)."""
        import pynvml
        
        pynvml.nvmlInit()
        
        try:
            gpu_count = pynvml.nvmlDeviceGetCount()
            
            if gpu_count == 0:
                logger.warning("No GPUs detected")
                return self._create_cpu_capabilities()
            
            # Get info from first GPU (assuming homogeneous)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_name = name.decode() if isinstance(name, bytes) else name
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_bytes = mem_info.total
            
            # Driver version
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver_version = driver.decode() if isinstance(driver, bytes) else driver
            
            # Try to get CUDA version from torch
            cuda_version = self._get_cuda_version()
            
            # Get compute capability
            major, minor = self._get_compute_capability(handle)
            compute_capability = (major, minor)
            
            # Determine architecture from compute capability
            arch = self._arch_from_compute_capability(major, minor)
            
            # Look up specs from database
            specs = get_gpu_specs(gpu_name)
            
            if specs:
                # Use database specs but override with detected values
                return HardwareCapabilities(
                    gpu_model=specs.name,
                    gpu_count=gpu_count,
                    gpu_architecture=specs.architecture,
                    compute_capability=specs.compute_capability,
                    has_tensor_cores=specs.tensor_core_version >= 1,
                    tensor_core_version=specs.tensor_core_version,
                    vram_gb=(vram_bytes * gpu_count) / (1024**3),
                    vram_gb_per_gpu=vram_bytes / (1024**3),
                    memory_bandwidth_gbps=specs.memory_bandwidth_gbps,
                    l2_cache_mb=specs.l2_cache_mb,
                    native_precisions=specs.native_precisions,
                    recommended_weight_quant=specs.recommended_weight_quant,
                    kv_cache_quant_supported=specs.kv_cache_quant_supported,
                    fp16_tflops=specs.fp16_tflops,
                    fp8_tflops=specs.fp8_tflops,
                    int8_tops=specs.int8_tops,
                    cuda_version=cuda_version,
                    driver_version=driver_version,
                )
            else:
                # GPU not in database, infer from compute capability
                return self._infer_capabilities_from_cc(
                    gpu_name, gpu_count, vram_bytes, compute_capability,
                    cuda_version, driver_version
                )
                
        finally:
            pynvml.nvmlShutdown()
    
    def _get_compute_capability(self, handle) -> tuple:
        """Get compute capability for a GPU handle."""
        try:
            import pynvml
            # Get compute capability from PCI info or major/minor version
            # pynvml doesn't directly expose compute capability, so we infer from name
            # or use nvidia-smi --query-gpu=compute_cap
            
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                cc_str = result.stdout.strip().split('\n')[0].strip()
                major, minor = cc_str.split('.')
                return int(major), int(minor)
        except Exception as e:
            logger.debug(f"Could not get compute capability: {e}")
        
        return (0, 0)
    
    def _arch_from_compute_capability(self, major: int, minor: int) -> GPUArchitecture:
        """Determine architecture from compute capability."""
        cc = (major, minor)
        
        if cc >= (10, 0):
            return GPUArchitecture.BLACKWELL
        elif cc >= (9, 0):
            return GPUArchitecture.HOPPER
        elif cc >= (8, 9):
            return GPUArchitecture.ADA
        elif cc >= (8, 0):
            return GPUArchitecture.AMPERE
        elif cc >= (7, 5):
            return GPUArchitecture.TURING
        else:
            return GPUArchitecture.UNKNOWN
    
    def _infer_capabilities_from_cc(
        self, 
        gpu_name: str, 
        gpu_count: int, 
        vram_bytes: int,
        compute_capability: tuple,
        cuda_version: Optional[str],
        driver_version: str
    ) -> HardwareCapabilities:
        """Infer capabilities from compute capability when GPU not in database."""
        arch = self._arch_from_compute_capability(*compute_capability)
        
        # Infer precision support from architecture
        native_precisions = {"fp32", "fp16", "int8"}
        
        if supports_bf16(arch):
            native_precisions.add("bf16")
        if supports_fp8(arch):
            native_precisions.add("fp8")
        
        # Tensor core version from architecture
        tc_version = {
            GPUArchitecture.TURING: 2,
            GPUArchitecture.AMPERE: 3,
            GPUArchitecture.ADA: 4,
            GPUArchitecture.HOPPER: 4,
            GPUArchitecture.BLACKWELL: 5,
        }.get(arch, 0)
        
        # Recommendations based on architecture
        if supports_fp8(arch):
            recommended = ["fp8", "awq", "gptq", "int8"]
            kv_supported = ["fp8"]
        elif arch == GPUArchitecture.AMPERE:
            recommended = ["awq", "gptq", "int8"]
            kv_supported = []
        else:
            recommended = ["gptq", "int8"]
            kv_supported = []
        
        return HardwareCapabilities(
            gpu_model=gpu_name,
            gpu_count=gpu_count,
            gpu_architecture=arch,
            compute_capability=compute_capability,
            has_tensor_cores=tc_version >= 1,
            tensor_core_version=tc_version,
            vram_gb=(vram_bytes * gpu_count) / (1024**3),
            vram_gb_per_gpu=vram_bytes / (1024**3),
            memory_bandwidth_gbps=0.0,  # Unknown
            l2_cache_mb=0.0,  # Unknown
            native_precisions=native_precisions,
            recommended_weight_quant=recommended,
            kv_cache_quant_supported=kv_supported,
            cuda_version=cuda_version,
            driver_version=driver_version,
        )
    
    def _detect_fallback(self) -> HardwareCapabilities:
        """Fallback detection using nvidia-smi CLI."""
        try:
            # Get GPU name
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return self._create_cpu_capabilities()
            
            gpu_names = result.stdout.strip().split('\n')
            gpu_name = gpu_names[0].strip()
            gpu_count = len(gpu_names)
            
            # Get memory
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                vram_mb = float(result.stdout.strip().split('\n')[0])
                vram_bytes = int(vram_mb * 1024 * 1024)
            else:
                vram_bytes = 0
            
            # Get compute capability
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                cc_str = result.stdout.strip().split('\n')[0].strip()
                major, minor = cc_str.split('.')
                compute_capability = (int(major), int(minor))
            else:
                compute_capability = (0, 0)
            
            # Get driver version
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            driver_version = result.stdout.strip().split('\n')[0].strip() if result.returncode == 0 else "unknown"
            
            cuda_version = self._get_cuda_version()
            
            # Check database
            specs = get_gpu_specs(gpu_name)
            
            if specs:
                return HardwareCapabilities(
                    gpu_model=specs.name,
                    gpu_count=gpu_count,
                    gpu_architecture=specs.architecture,
                    compute_capability=specs.compute_capability,
                    has_tensor_cores=specs.tensor_core_version >= 1,
                    tensor_core_version=specs.tensor_core_version,
                    vram_gb=(vram_bytes * gpu_count) / (1024**3),
                    vram_gb_per_gpu=vram_bytes / (1024**3),
                    memory_bandwidth_gbps=specs.memory_bandwidth_gbps,
                    l2_cache_mb=specs.l2_cache_mb,
                    native_precisions=specs.native_precisions,
                    recommended_weight_quant=specs.recommended_weight_quant,
                    kv_cache_quant_supported=specs.kv_cache_quant_supported,
                    fp16_tflops=specs.fp16_tflops,
                    fp8_tflops=specs.fp8_tflops,
                    int8_tops=specs.int8_tops,
                    cuda_version=cuda_version,
                    driver_version=driver_version,
                )
            else:
                return self._infer_capabilities_from_cc(
                    gpu_name, gpu_count, vram_bytes, compute_capability,
                    cuda_version, driver_version
                )
                
        except Exception as e:
            logger.warning(f"Fallback detection failed: {e}")
            return self._create_cpu_capabilities()
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version from torch or nvcc."""
        try:
            import torch
            return torch.version.cuda
        except ImportError:
            pass
        
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "release" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "release":
                                return parts[i + 1].rstrip(',')
        except Exception:
            pass
        
        return None
    
    def _create_cpu_capabilities(self) -> HardwareCapabilities:
        """Create capabilities for CPU-only system."""
        return HardwareCapabilities(
            gpu_model="CPU",
            gpu_count=0,
            gpu_architecture=GPUArchitecture.UNKNOWN,
            compute_capability=(0, 0),
            has_tensor_cores=False,
            tensor_core_version=0,
            vram_gb=0.0,
            vram_gb_per_gpu=0.0,
            memory_bandwidth_gbps=0.0,
            l2_cache_mb=0.0,
            native_precisions=set(),
            recommended_weight_quant=[],
            kv_cache_quant_supported=[],
            cuda_version=None,
            driver_version=None,
        )


def detect_hardware() -> HardwareCapabilities:
    """Convenience function to detect hardware capabilities."""
    detector = HardwareDetector()
    return detector.detect()


def get_platform_info_with_hardware() -> Dict[str, Any]:
    """Get platform info dict with hardware capabilities included.
    
    Returns:
        Dict suitable for passing to SME.register() and analyze()
    """
    caps = detect_hardware()
    
    return {
        "type": "nvidia_cuda" if caps.gpu_count > 0 else "cpu",
        "gpu_model": caps.gpu_model,
        "gpu_count": caps.gpu_count,
        "cuda_version": caps.cuda_version,
        "driver_version": caps.driver_version,
        "hardware_capabilities": caps.to_dict(),
    }
