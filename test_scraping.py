#!/usr/bin/env python
"""
Test script to verify the Neuroscope scraping functionality
"""

from scrape_neuroscope_activations import get_max_activation

def test_single_neuron():
    """Test scraping a single neuron to verify the functionality"""
    
    # Test URL from the example
    test_url = "https://neuroscope.io/pythia-70m/0/0.html"
    
    print("🧪 Testing single neuron scraping...")
    print(f"   URL: {test_url}")
    
    max_act = get_max_activation(test_url)
    
    if max_act is not None:
        print(f"   ✅ Success! Max activation: {max_act}")
        return True
    else:
        print(f"   ❌ Failed to get max activation")
        return False

def test_multiple_neurons():
    """Test scraping a few neurons to verify the pattern"""
    
    test_cases = [
        ("https://neuroscope.io/pythia-70m/0/0.html", "L0N0"),
        ("https://neuroscope.io/pythia-70m/0/1.html", "L0N1"),
        ("https://neuroscope.io/pythia-70m/1/0.html", "L1N0"),
    ]
    
    print("\n🧪 Testing multiple neurons...")
    
    results = {}
    for url, neuron_id in test_cases:
        print(f"   Testing {neuron_id}...")
        max_act = get_max_activation(url)
        if max_act is not None:
            results[neuron_id] = max_act
            print(f"     ✅ {neuron_id}: {max_act}")
        else:
            print(f"     ❌ {neuron_id}: Failed")
    
    print(f"\n📊 Results: {len(results)}/{len(test_cases)} successful")
    for neuron_id, activation in results.items():
        print(f"   {neuron_id}: {activation}")
    
    return len(results) == len(test_cases)

if __name__ == "__main__":
    print("🚀 Starting Neuroscope scraping tests...\n")
    
    # Test single neuron
    single_success = test_single_neuron()
    
    # Test multiple neurons
    multiple_success = test_multiple_neurons()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Single neuron test: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"   Multiple neurons test: {'✅ PASS' if multiple_success else '❌ FAIL'}")
    
    if single_success and multiple_success:
        print(f"\n🎉 All tests passed! The scraping script should work correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the implementation.")
