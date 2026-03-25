#!/usr/bin/env python3
"""Quick test for placeholder cleaning and model discovery logic."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_placeholder_cleaning():
    """Test the placeholder cleaning logic."""
    print("="*60)
    print("TEST 1: Placeholder Value Cleaning")
    print("="*60)
    
    from forge.smes.quantization_sme import QuantizationSME
    
    # Create SME instance (without LLM pool)
    sme = QuantizationSME.__new__(QuantizationSME)
    
    test_cases = [
        # (input, expected_to_be_cleaned)
        ({'quantization': 'gptq', 'model_id': 'Use GPTQ-quantized variant of current model'}, True),
        ({'quantization': 'awq', 'model': 'HF model ID if pre-quantized'}, True),
        ({'quantization': 'fp8', 'model_id': 'Use FP8-quantized variant'}, True),
        ({'quantization': 'gptq', 'model_id': 'TheBloke/Llama-2-7B-GPTQ'}, False),
        ({'quantization': 'awq', 'model': 'TheBloke/Llama-2-7B-AWQ'}, False),
        ({'quantization': 'fp8', 'kv_cache_dtype': 'fp8'}, False),
    ]
    
    print()
    all_passed = True
    for config, should_clean in test_cases:
        original = dict(config)
        sme._clean_placeholder_values(config)
        
        was_cleaned = len(config) < len(original)
        
        if was_cleaned == should_clean:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        print(f"  {status}: {original}")
        if was_cleaned:
            print(f"         → Cleaned to: {config}")
    
    return all_passed


async def test_hf_search():
    """Test HF model search (if available)."""
    print("\n" + "="*60)
    print("TEST 2: HuggingFace Model Search")
    print("="*60)
    
    try:
        from forge.smes.quantization.hf_model_finder import HFQuantizedModelFinder
        
        finder = HFQuantizedModelFinder()
        
        if not finder.is_available():
            print("\n  ℹ️ HuggingFace Hub not available")
            print("     Install with: pip install huggingface_hub")
            return True
        
        print("\n  ✓ HuggingFace Hub available")
        
        # Test search for a well-known quantized model
        print("\n  Searching for Llama-2-7b GPTQ models...")
        model_id = await finder.find_model(
            base_model="meta-llama/Llama-2-7b-hf",
            quantization="gptq",
            bits=4,
            prefer_author="TheBloke"
        )
        
        if model_id:
            print(f"  ✓ Found: {model_id}")
        else:
            print(f"  ℹ️ No model found (may need HF token for gated models)")
        
        # Test AWQ search
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
            print(f"  ℹ️ No model found")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ HF search test failed: {e}")
        return False


def test_end_to_end_flow():
    """Test the full flow with mock data."""
    print("\n" + "="*60)
    print("TEST 3: End-to-End Flow (Mock)")
    print("="*60)
    
    from forge.smes.quantization_sme import QuantizationSME
    
    # Simulate what happens after LLM consensus
    sme = QuantizationSME.__new__(QuantizationSME)
    
    # Mock config_changes from LLM (with placeholder)
    config_changes = {
        'quantization': 'gptq',
        'model_id': 'Use GPTQ-quantized variant of current model'
    }
    
    print(f"\n  Input from LLM: {config_changes}")
    
    # Clean placeholders
    sme._clean_placeholder_values(config_changes)
    print(f"  After cleaning: {config_changes}")
    
    # Verify placeholder was removed
    if 'model_id' in config_changes:
        print("  ✗ FAIL: placeholder still present")
        return False
    
    if config_changes.get('quantization') != 'gptq':
        print("  ✗ FAIL: quantization field lost")
        return False
    
    print("  ✓ PASS: placeholder cleaned, quantization preserved")
    
    # What happens next (without actual HF search)
    print("\n  Next step: HF search would run for base model")
    print("  If found: config_changes['model'] = 'TheBloke/...'")
    print("  If not:  config_changes['model_source'] = 'on_the_fly'")
    
    return True


async def main():
    """Run all tests."""
    print("="*60)
    print("PLACEHOLDER CLEANING & MODEL DISCOVERY TEST")
    print("="*60)
    
    results = []
    
    # Test 1: Placeholder cleaning
    results.append(("Placeholder Cleaning", test_placeholder_cleaning()))
    
    # Test 2: HF search
    results.append(("HF Model Search", await test_hf_search()))
    
    # Test 3: End-to-end flow
    results.append(("End-to-End Flow", test_end_to_end_flow()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print(f"\n{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
