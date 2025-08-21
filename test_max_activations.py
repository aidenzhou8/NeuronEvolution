#!/usr/bin/env python
"""
Test script to verify that max activations are being loaded correctly
"""

from global_analysis import load_max_activations

def test_max_activations():
    """Test loading and displaying max activations for a few neurons"""
    
    print("🧪 Testing max activations loading...")
    
    # Load max activations
    max_activations = load_max_activations()
    
    if max_activations is None:
        print("❌ Failed to load max activations")
        return False
    
    # Test a few specific neurons
    test_neurons = [
        "L0N0", "L0N1", "L1N0", "L2N0", "L3N0", "L4N0", "L5N0"
    ]
    
    print(f"\n📊 Testing {len(test_neurons)} neurons:")
    for neuron in test_neurons:
        if neuron in max_activations:
            print(f"   {neuron}: {max_activations[neuron]}")
        else:
            print(f"   {neuron}: ❌ Not found")
    
    # Test some statistics
    all_activations = list(max_activations.values())
    print(f"\n📈 Statistics:")
    print(f"   Total neurons: {len(max_activations)}")
    print(f"   Average max activation: {sum(all_activations) / len(all_activations):.4f}")
    print(f"   Min max activation: {min(all_activations):.4f}")
    print(f"   Max max activation: {max(all_activations):.4f}")
    
    # Test layer-specific statistics
    print(f"\n🏗️  Layer statistics:")
    for layer in range(6):
        layer_activations = []
        for neuron_key, activation in max_activations.items():
            if neuron_key.startswith(f"L{layer}N"):
                layer_activations.append(activation)
        
        if layer_activations:
            avg = sum(layer_activations) / len(layer_activations)
            print(f"   Layer {layer}: {len(layer_activations)} neurons, avg: {avg:.4f}")
    
    return True

if __name__ == "__main__":
    success = test_max_activations()
    if success:
        print(f"\n✅ Max activations test passed!")
    else:
        print(f"\n❌ Max activations test failed!")
