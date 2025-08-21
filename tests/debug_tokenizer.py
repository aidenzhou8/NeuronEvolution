#!/usr/bin/env python
"""
Debug script to see tokenizer output structure.
"""

from transformer_lens import HookedTransformer
from datasets import load_dataset

def main():
    # Load the model
    model = HookedTransformer.from_pretrained('gpt2-small')
    tokenizer = model.tokenizer
    
    # Load dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=False)
    
    # Get first batch
    texts = ds[0:2]["text"]
    toks = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    
    print("Tokenizer output type:", type(toks))
    print("Tokenizer output keys:", toks.keys())
    print("input_ids shape:", toks['input_ids'].shape)
    print("attention_mask shape:", toks['attention_mask'].shape)
    
    # Try to pass to model
    print("\nTrying to pass to model...")
    try:
        output = model(**toks)
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    main() 