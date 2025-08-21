#!/usr/bin/env python
"""
Demonstration script showing how individual peak activations work
"""

from global_analysis import load_max_activations, run_checkpoint_analysis

def demo_single_neuron():
    """Demonstrate using individual peak activation for a single neuron"""
    
    print("🚀 Demo: Using individual peak activation for L0N0")
    
    # Load max activations
    max_activations = load_max_activations()
    if max_activations is None:
        print("❌ Cannot proceed without max activations data")
        return
    
    # Test with L0N0
    layer = 0
    neuron = 0
    
    print(f"\n📊 Neuron L{layer}N{neuron}:")
    neuron_key = f"L{layer}N{neuron}"
    if neuron_key in max_activations:
        peak_activation = max_activations[neuron_key]
        print(f"   Max activation from Neuroscope: {peak_activation}")
        print(f"   This will be used as --peak_activation parameter")
        
        # Show what the command would look like
        print(f"\n🔧 Command that would be run:")
        print(f"   python checkpoints_demo.py --series --model pythia-1.4b --layer blocks.{layer}.mlp --neuron {neuron} --peak_activation {peak_activation} ...")
        
        # Ask if user wants to actually run it
        response = input(f"\n❓ Do you want to run the analysis for L{layer}N{neuron}? (y/n): ")
        if response.lower() == 'y':
            print(f"\n🏃 Running analysis...")
            success = run_checkpoint_analysis(layer, neuron, max_activations)
            if success:
                print(f"✅ Analysis completed successfully!")
            else:
                print(f"❌ Analysis failed!")
        else:
            print(f"⏭️  Skipping analysis")
    else:
        print(f"❌ No max activation found for {neuron_key}")

def compare_peaks():
    """Compare different neurons' peak activations"""
    
    print("🔍 Comparing peak activations across layers...")
    
    # Load max activations
    max_activations = load_max_activations()
    if max_activations is None:
        print("❌ Cannot proceed without max activations data")
        return
    
    # Compare first neuron from each layer
    print(f"\n📊 First neuron from each layer:")
    for layer in range(6):
        neuron_key = f"L{layer}N0"
        if neuron_key in max_activations:
            peak_activation = max_activations[neuron_key]
            print(f"   {neuron_key}: {peak_activation}")
        else:
            print(f"   {neuron_key}: ❌ Not found")
    
    # Show some statistics
    print(f"\n📈 Peak activation statistics:")
    all_activations = list(max_activations.values())
    print(f"   Average: {sum(all_activations) / len(all_activations):.4f}")
    print(f"   Median: {sorted(all_activations)[len(all_activations)//2]:.4f}")
    print(f"   Min: {min(all_activations):.4f}")
    print(f"   Max: {max(all_activations):.4f}")
    
    # Show how this compares to the old hardcoded value
    old_value = 2.8
    print(f"\n🔄 Comparison with old hardcoded value ({old_value}):")
    below_old = sum(1 for a in all_activations if a < old_value)
    above_old = sum(1 for a in all_activations if a > old_value)
    print(f"   Neurons with peak < {old_value}: {below_old} ({below_old/len(all_activations)*100:.1f}%)")
    print(f"   Neurons with peak > {old_value}: {above_old} ({above_old/len(all_activations)*100:.1f}%)")

if __name__ == "__main__":
    print("🎯 Individual Peak Activation Demo")
    print("=" * 50)
    
    # Compare peak activations
    compare_peaks()
    
    print(f"\n" + "=" * 50)
    
    # Demo single neuron
    demo_single_neuron()
