#!/usr/bin/env python
"""
Example script demonstrating how to scrape max activations from Neuroscope
for a small subset of neurons (for testing purposes)
"""

from scrape_neuroscope_activations import scrape_model_activations, save_activations

def main():
    """Example: Scrape a small subset of neurons for testing"""
    
    print("🚀 Example: Scraping a small subset of neurons")
    
    # For testing, let's just scrape a few neurons from the first layer
    # We'll simulate a model with 1 layer and 5 neurons
    model_name = "pythia-70m"
    num_layers = 1
    neurons_per_layer = 5
    
    print(f"   Model: {model_name}")
    print(f"   Layers: {num_layers}")
    print(f"   Neurons per layer: {neurons_per_layer}")
    print(f"   Total neurons to scrape: {num_layers * neurons_per_layer}")
    print()
    
    # Scrape the activations
    activations = scrape_model_activations(
        model_name, 
        num_layers, 
        neurons_per_layer
    )
    
    # Save the results
    save_activations(activations, f"{model_name}_example", "results")
    
    # Display the results
    print(f"\n📊 Results:")
    for neuron_id, activation in sorted(activations.items()):
        print(f"   {neuron_id}: {activation}")
    
    # Convert to a simple list as requested
    activation_list = []
    for neuron_id in sorted(activations.keys()):
        activation_list.append(activations[neuron_id])
    
    print(f"\n📋 Activation list (as requested):")
    print(f"   {activation_list}")
    
    print(f"\n🎉 Example completed!")
    print(f"   You can now use the full script to scrape all neurons:")
    print(f"   python scrape_neuroscope_activations.py --model pythia-70m --layers 6 --neurons-per-layer 1024")

if __name__ == "__main__":
    main()
