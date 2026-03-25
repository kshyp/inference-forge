#!/usr/bin/env python3
"""Test script for QuantizationSME with hardware-aware recommendations.

This script tests:
1. Hardware capability detection
2. QuantizationSME registration
3. Quantization recommendations with different hardware scenarios
4. HuggingFace model discovery (optional)

Usage:
    python test_quantization_sme.py [--mock-hardware H100|A100|RTX4090]

Example:
    python test_quantization_sme.py --mock-hardware H100
"""

import asyncio
import argparse
import json
from pathlib import Path

# Add forge to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from forge.hardware import HardwareCapabilities, GPUArchitecture
from forge.smes.quantization_sme import QuantizationSME
from forge.sme_registry import SMERegistry
from forge.core.events import Task


async def test_hardware_detection():
    """Test hardware capability detection."""
    print("\n" + "="*70)
    print("TEST 1: Hardware Capability Detection")
    print("="*70)
    
    from forge.hardware import detect_hardware
    
    try:
        caps = detect_hardware()
        print(f"\n✓ Hardware detection successful")
        print(f"  GPU Model: {caps.gpu_model}")
        print(f"  GPU Count: {caps.gpu_count}")
        print(f"  Architecture: {caps.gpu_architecture.value}")
        print(f"  Compute Capability: {caps.compute_capability}")
        print(f"  VRAM: {caps.vram_gb:.1f} GB")
        print(f"  Tensor Cores: v{caps.tensor_core_version}")
        print(f"  Native Precisions: {', '.join(sorted(caps.native_precisions))}")
        print(f"  Recommended Quantization: {', '.join(caps.recommended_weight_quant[:2])}")
        print(f"  KV Cache Quantization: {', '.join(caps.kv_cache_quant_supported) or 'Not supported'}")
        return caps
    except Exception as e:
        print(f"\n✗ Hardware detection failed: {e}")
        return None


def create_mock_hardware(gpu_model: str) -> dict:
    """Create mock hardware capabilities for testing."""
    
    mock_specs = {
        "H100": {
            "gpu_model": "H100",
            "gpu_architecture": "hopper",
            "compute_capability": "9.0",
            "vram_gb": 80.0,
            "native_precisions": ["fp16", "bf16", "fp8", "int8"],
            "recommended_weight_quant": ["fp8", "awq", "gptq"],
            "kv_cache_quant_supported": ["fp8"],
            "supports_fp8": True,
            "supports_bf16": True,
        },
        "A100": {
            "gpu_model": "A100",
            "gpu_architecture": "ampere",
            "compute_capability": "8.0",
            "vram_gb": 80.0,
            "native_precisions": ["fp16", "bf16", "int8"],
            "recommended_weight_quant": ["awq", "gptq", "int8"],
            "kv_cache_quant_supported": [],
            "supports_fp8": False,
            "supports_bf16": True,
        },
        "RTX4090": {
            "gpu_model": "RTX 4090",
            "gpu_architecture": "ada",
            "compute_capability": "8.9",
            "vram_gb": 24.0,
            "native_precisions": ["fp16", "bf16", "fp8", "int8"],
            "recommended_weight_quant": ["fp8", "awq", "gptq"],
            "kv_cache_quant_supported": ["fp8"],
            "supports_fp8": True,
            "supports_bf16": True,
        },
    }
    
    return mock_specs.get(gpu_model, mock_specs["A100"])


async def test_sme_registration(hardware_caps: dict):
    """Test QuantizationSME registration with hardware capabilities."""
    print("\n" + "="*70)
    print("TEST 2: QuantizationSME Registration")
    print("="*70)
    
    platform_info = {
        "type": "nvidia_cuda",
        "gpu_model": hardware_caps["gpu_model"],
        "gpu_count": 1,
        "cuda_version": "12.2",
        "driver_version": "535.104",
        "hardware_capabilities": hardware_caps,
    }
    
    sme = QuantizationSME()
    reg_info = sme.register(platform_info)
    
    if reg_info:
        print(f"\n✓ SME registered successfully")
        print(f"  SME ID: {reg_info.sme_id}")
        print(f"  Description: {reg_info.description}")
        print(f"  Data Requirements: {len(reg_info.data_requirements)} types")
        
        # Show data requirements
        print(f"\n  Data Requirements:")
        for req in reg_info.data_requirements[:5]:
            print(f"    - {req.data_type} (required: {req.required})")
        if len(reg_info.data_requirements) > 5:
            print(f"    ... and {len(reg_info.data_requirements) - 5} more")
        
        return sme
    else:
        print(f"\n✗ SME registration failed")
        return None


async def test_analyze_scenario(sme: QuantizationSME, scenario: str, hardware_caps: dict):
    """Test analyze with a specific scenario."""
    print(f"\n--- Scenario: {scenario} ---")
    
    # Create mock profiling data based on scenario
    if scenario == "memory_bandwidth_bound":
        profiling_data = {
            "hardware_capabilities": hardware_caps,
            "system_metrics_report": {
                "gpu_utilization_percent": 85.0,
                "memory_used_mb": 60000,
                "memory_total_mb": 80000,
            },
            "vllm_logs": {
                "config": {
                    "model_name": "meta-llama/Llama-2-7b-hf",
                    "model_dtype": "fp16",
                    "quantization": "none",
                },
                "metrics": {
                    "kv_cache_usage_percent": 60.0,
                    "model_weights_memory_mb": 14000,
                }
            },
            "ncu_report": {
                "memory_bandwidth_utilization_percent": 85.0,
                "compute_utilization_percent": 60.0,
            },
        }
        benchmark_metrics = {
            "throughput_rps": 45.0,
            "output_tokens_per_sec": 1200.0,
        }
        
    elif scenario == "kv_cache_pressure":
        profiling_data = {
            "hardware_capabilities": hardware_caps,
            "system_metrics_report": {
                "gpu_utilization_percent": 70.0,
                "memory_used_mb": 75000,
                "memory_total_mb": 80000,
            },
            "vllm_logs": {
                "config": {
                    "model_name": "meta-llama/Llama-2-7b-hf",
                    "model_dtype": "fp16",
                    "quantization": "none",
                },
                "metrics": {
                    "kv_cache_usage_percent": 92.0,
                    "model_weights_memory_mb": 14000,
                }
            },
            "ncu_report": {
                "memory_bandwidth_utilization_percent": 65.0,
                "compute_utilization_percent": 70.0,
            },
        }
        benchmark_metrics = {
            "throughput_rps": 30.0,
            "output_tokens_per_sec": 800.0,
        }
        
    elif scenario == "baseline":
        profiling_data = {
            "hardware_capabilities": hardware_caps,
            "system_metrics_report": {
                "gpu_utilization_percent": 50.0,
                "memory_used_mb": 40000,
                "memory_total_mb": 80000,
            },
            "vllm_logs": {
                "config": {
                    "model_name": "meta-llama/Llama-2-7b-hf",
                    "model_dtype": "fp16",
                    "quantization": "none",
                },
                "metrics": {
                    "kv_cache_usage_percent": 40.0,
                    "model_weights_memory_mb": 14000,
                }
            },
        }
        benchmark_metrics = {
            "throughput_rps": 20.0,
            "output_tokens_per_sec": 500.0,
        }
    else:
        return
    
    try:
        from pathlib import Path
        profile_dir = Path("./test_data/profile")
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        response = await sme.analyze(profile_dir, profiling_data, benchmark_metrics)
        
        print(f"  Is Relevant: {response.is_relevant}")
        print(f"  Relevance Score: {response.relevance_score:.2f}")
        print(f"  Relevance Reason: {response.relevance_reason}")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Findings:")
        for key, value in response.findings.items():
            print(f"    - {key}: {value}")
        
        print(f"  Suggestions:")
        for i, suggestion in enumerate(response.suggestions, 1):
            print(f"    {i}. {suggestion.config_changes}")
            print(f"       Expected: {suggestion.expected_improvement}")
            print(f"       Confidence: {suggestion.confidence:.2f}")
            
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


async def test_hf_model_finder():
    """Test HuggingFace model discovery."""
    print("\n" + "="*70)
    print("TEST 4: HuggingFace Quantized Model Discovery")
    print("="*70)
    
    try:
        from forge.smes.quantization.hf_model_finder import HFQuantizedModelFinder
        
        finder = HFQuantizedModelFinder()
        
        if not finder.is_available():
            print("\nℹ️ HuggingFace Hub not available (install with: pip install huggingface_hub)")
            return
        
        print("\n✓ HuggingFace Hub available")
        
        # Test search for AWQ models
        print("\n  Searching for Llama-2-7b AWQ models...")
        model_id = await finder.find_model(
            base_model="meta-llama/Llama-2-7b-hf",
            quantization="awq",
            bits=4,
            prefer_author="TheBloke"
        )
        
        if model_id:
            print(f"  ✓ Found: {model_id}")
        else:
            print(f"  ℹ️ No models found (may need HF token for gated models)")
            
    except Exception as e:
        print(f"\n✗ HF model finder test failed: {e}")


async def test_sme_registry_integration(hardware_caps: dict):
    """Test SME registry integration with quantization SME."""
    print("\n" + "="*70)
    print("TEST 5: SME Registry Integration")
    print("="*70)
    
    platform_info = {
        "type": "nvidia_cuda",
        "gpu_model": hardware_caps["gpu_model"],
        "gpu_count": 1,
        "hardware_capabilities": hardware_caps,
    }
    
    registry = SMERegistry()
    registry.discover_sme_classes()
    registered = registry.register_all(platform_info)
    
    print(f"\n✓ Registered {len(registered)} SMEs")
    for sme_id in registered:
        print(f"  - {sme_id}")
    
    # Check if quantization SME is registered
    if "quantization" in registered:
        print("\n✓ QuantizationSME successfully registered")
        
        # Get data requirements
        requirements = registry.get_profiler_requirements()
        print(f"\n  Data requirements:")
        for data_type, info in requirements.items():
            if "quantization" in info.get("requested_by", []):
                print(f"    - {data_type} (required: {info['required']})")
    else:
        print("\n✗ QuantizationSME not registered")


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test QuantizationSME")
    parser.add_argument(
        "--mock-hardware",
        choices=["H100", "A100", "RTX4090"],
        help="Mock hardware for testing (default: detect actual hardware)"
    )
    parser.add_argument(
        "--scenario",
        choices=["memory_bandwidth", "kv_cache", "baseline", "all"],
        default="all",
        help="Test scenario"
    )
    args = parser.parse_args()
    
    print("="*70)
    print("QUANTIZATION SME TEST SUITE")
    print("="*70)
    
    # Test 1: Hardware Detection
    if args.mock_hardware:
        hardware_caps = create_mock_hardware(args.mock_hardware)
        print(f"\nℹ️ Using mock hardware: {args.mock_hardware}")
        print(f"   Architecture: {hardware_caps['gpu_architecture']}")
        print(f"   Supports FP8: {hardware_caps['supports_fp8']}")
    else:
        actual_caps = await test_hardware_detection()
        if actual_caps:
            hardware_caps = actual_caps.to_dict()
        else:
            print("\n⚠️ Using mock H100 hardware")
            hardware_caps = create_mock_hardware("H100")
    
    # Test 2: SME Registration
    sme = await test_sme_registration(hardware_caps)
    if not sme:
        print("\n✗ Cannot continue without SME registration")
        return
    
    # Test 3: Analyze Scenarios
    print("\n" + "="*70)
    print("TEST 3: Quantization Recommendations by Scenario")
    print("="*70)
    
    scenarios = []
    if args.scenario == "all":
        scenarios = ["memory_bandwidth_bound", "kv_cache_pressure", "baseline"]
    elif args.scenario == "memory_bandwidth":
        scenarios = ["memory_bandwidth_bound"]
    elif args.scenario == "kv_cache":
        scenarios = ["kv_cache_pressure"]
    elif args.scenario == "baseline":
        scenarios = ["baseline"]
    
    for scenario in scenarios:
        await test_analyze_scenario(sme, scenario, hardware_caps)
    
    # Test 4: HF Model Finder (optional)
    await test_hf_model_finder()
    
    # Test 5: Registry Integration
    await test_sme_registry_integration(hardware_caps)
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
