#!/usr/bin/env python3
"""Simple test for quantization SME components without LLM calls."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_hardware_database():
    """Test GPU specs database."""
    print("\n" + "="*60)
    print("TEST 1: GPU Specs Database")
    print("="*60)
    
    from forge.hardware.database import GPU_SPECS_DATABASE, get_gpu_specs, GPUArchitecture
    
    print(f"\n✓ Database has {len(GPU_SPECS_DATABASE)} GPUs")
    
    # Test key GPUs
    test_gpus = ["H100", "A100", "RTX 4090", "L40S", "A10", "T4"]
    for gpu in test_gpus:
        specs = get_gpu_specs(gpu)
        if specs:
            fp8_support = "fp8" in specs.native_precisions
            kv_support = specs.kv_cache_quant_supported
            print(f"  {gpu:15} | Arch: {specs.architecture.value:10} | "
                  f"FP8: {fp8_support!s:5} | KV: {kv_support}")
        else:
            print(f"  {gpu:15} | NOT FOUND")
    
    return True


def test_hardware_detection():
    """Test hardware detection."""
    print("\n" + "="*60)
    print("TEST 2: Hardware Detection")
    print("="*60)
    
    from forge.hardware.detector import HardwareDetector
    
    detector = HardwareDetector()
    print(f"\n✓ HardwareDetector created")
    print(f"  pynvml available: {detector._pynvml_available}")
    
    # Try detection
    try:
        caps = detector.detect()
        print(f"\n✓ Hardware detection successful")
        print(f"  GPU: {caps.gpu_model}")
        print(f"  Count: {caps.gpu_count}")
        print(f"  Arch: {caps.gpu_architecture.value}")
        print(f"  Compute Capability: {caps.compute_capability}")
        print(f"  VRAM: {caps.vram_gb:.1f} GB")
        print(f"  Precisions: {', '.join(sorted(caps.native_precisions))}")
        print(f"  Weight Quant: {caps.recommended_weight_quant}")
        print(f"  KV Cache: {caps.kv_cache_quant_supported}")
        return caps.to_dict()
    except Exception as e:
        print(f"\n✗ Hardware detection failed: {e}")
        return None


def test_sme_registration():
    """Test SME registration."""
    print("\n" + "="*60)
    print("TEST 3: QuantizationSME Registration")
    print("="*60)
    
    from forge.smes.quantization_sme import QuantizationSME
    
    # Create mock hardware capabilities
    hw_caps = {
        "gpu_model": "H100",
        "gpu_architecture": "hopper",
        "compute_capability": "9.0",
        "vram_gb": 80.0,
        "native_precisions": ["fp16", "bf16", "fp8", "int8"],
        "recommended_weight_quant": ["fp8", "awq", "gptq"],
        "kv_cache_quant_supported": ["fp8"],
        "supports_fp8": True,
    }
    
    platform_info = {
        "type": "nvidia_cuda",
        "gpu_model": "H100",
        "gpu_count": 1,
        "hardware_capabilities": hw_caps,
    }
    
    sme = QuantizationSME()
    reg_info = sme.register(platform_info)
    
    if reg_info:
        print(f"\n✓ SME registered successfully")
        print(f"  SME ID: {reg_info.sme_id}")
        print(f"  Triggers: {len(reg_info.triggers)}")
        print(f"  Data Requirements: {len(reg_info.data_requirements)}")
        
        # Show first few triggers
        print(f"\n  Sample triggers:")
        for trigger in reg_info.triggers[:5]:
            print(f"    - {trigger}")
        
        return sme
    else:
        print(f"\n✗ SME registration failed")
        return None


def test_recommendation_logic():
    """Test recommendation logic without LLM."""
    print("\n" + "="*60)
    print("TEST 4: Recommendation Logic (Fallback)")
    print("="*60)
    
    from forge.smes.quantization_sme import QuantizationSME
    
    sme = QuantizationSME()
    
    # Test with different hardware
    scenarios = [
        ("H100 (Hopper)", {
            "supports_fp8": True,
            "recommended_weight_quant": ["fp8", "awq", "gptq"],
        }),
        ("A100 (Ampere)", {
            "supports_fp8": False,
            "recommended_weight_quant": ["awq", "gptq", "int8"],
        }),
        ("RTX 4090 (Ada)", {
            "supports_fp8": True,
            "recommended_weight_quant": ["fp8", "awq", "gptq"],
        }),
    ]
    
    for name, hw_caps in scenarios:
        response = sme._create_fallback_response(hw_caps)
        
        print(f"\n  {name}:")
        print(f"    Findings: {response.findings}")
        if response.suggestions:
            for s in response.suggestions:
                print(f"    Suggestion: {s.config_changes}")
                print(f"      Expected: {s.expected_improvement}")
                print(f"      Confidence: {s.confidence}")


def test_hf_finder():
    """Test HF model finder initialization."""
    print("\n" + "="*60)
    print("TEST 5: HF Model Finder")
    print("="*60)
    
    try:
        from forge.smes.quantization.hf_model_finder import HFQuantizedModelFinder
        
        finder = HFQuantizedModelFinder()
        print(f"\n✓ HF Model Finder created")
        print(f"  Available: {finder.is_available()}")
        
        if finder.is_available():
            print(f"  API initialized: {finder.api is not None}")
        else:
            print(f"  ℹ️ HuggingFace Hub not installed (pip install huggingface_hub)")
        
        return True
    except Exception as e:
        print(f"\n✗ HF Model Finder failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("QUANTIZATION SME SIMPLE TESTS")
    print("="*60)
    
    results = []
    
    # Test 1
    results.append(("Database", test_hardware_database()))
    
    # Test 2
    hw_caps = test_hardware_detection()
    results.append(("Detection", hw_caps is not None))
    
    # Test 3
    sme = test_sme_registration()
    results.append(("Registration", sme is not None))
    
    # Test 4
    test_recommendation_logic()
    results.append(("Logic", True))
    
    # Test 5
    results.append(("HF Finder", test_hf_finder()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(p for _, p in results)
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
