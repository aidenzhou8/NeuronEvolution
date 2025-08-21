#!/usr/bin/env python
"""
Debug script to explore the module structure of TransformerLens models.
"""

from transformer_lens import HookedTransformer

def main():
    # Load the model
    model = HookedTransformer.from_pretrained('gpt2-small')
    
    print("Available modules:")
    print("==================")
    
    # Print all named modules
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            print(f"{name}: {type(module).__name__} (has weight)")
        else:
            print(f"{name}: {type(module).__name__}")
    
    print("\n" + "="*50)
    print("Looking for MLP-related modules:")
    print("================================")
    
    # Look specifically for MLP-related modules
    for name, module in model.named_modules():
        if 'mlp' in name.lower() or 'fc' in name.lower() or 'linear' in name.lower():
            print(f"{name}: {type(module).__name__}")
            if hasattr(module, 'weight'):
                print(f"  Weight shape: {module.weight.shape}")
            else:
                print("  No weight attribute")

if __name__ == "__main__":
    main() 