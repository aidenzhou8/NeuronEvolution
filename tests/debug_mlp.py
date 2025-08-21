#!/usr/bin/env python
"""
Debug script to explore the internal structure of MLP modules.
"""

from transformer_lens import HookedTransformer

def main():
    # Load the model
    model = HookedTransformer.from_pretrained('gpt2-small')
    
    # Get the first MLP module
    mlp_module = model.blocks[0].mlp
    
    print("MLP module type:", type(mlp_module))
    print("MLP module attributes:", dir(mlp_module))
    
    print("\n" + "="*50)
    print("MLP submodules:")
    print("===============")
    
    # Print all submodules of the MLP
    for name, submodule in mlp_module.named_modules():
        print(f"{name}: {type(submodule).__name__}")
        if hasattr(submodule, 'weight'):
            print(f"  Weight shape: {submodule.weight.shape}")
        else:
            print("  No weight attribute")
    
    print("\n" + "="*50)
    print("MLP named_parameters:")
    print("=====================")
    
    # Print all named parameters
    for name, param in mlp_module.named_parameters():
        print(f"{name}: {param.shape}")

if __name__ == "__main__":
    main() 