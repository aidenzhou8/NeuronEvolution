#!/usr/bin/env python
"""
Simple script to extract text excerpts for every checkpoint in a single JSON file
"""

import json
import argparse
from pathlib import Path

def extract_text_excerpts(layer, neuron, model="pythia70m", max_examples_per_cluster=None):
    """Extract text excerpts for every checkpoint"""
    
    results_dir = Path("results") / model
    jsonl_file = results_dir / f"L{layer}N{neuron}_{model}_ckpt_series.jsonl"
    
    if not jsonl_file.exists():
        print(f"❌ File not found: {jsonl_file}")
        return None
    
    print(f"🔍 Loading checkpoint data from {jsonl_file}")
    
    all_checkpoints = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_checkpoints.append(data)
    
    print(f"   Loaded {len(all_checkpoints)} checkpoints")
    
    # Sort by checkpoint step to ensure order
    all_checkpoints.sort(key=lambda x: x['checkpoint_step'])
    
    # Extract text excerpts for each checkpoint
    text_excerpts = []
    
    for checkpoint in all_checkpoints:
        checkpoint_step = checkpoint['checkpoint_step']
        cluster_labels = checkpoint['cluster_labels']
        text_examples = checkpoint['text_examples']
        
        # Group text examples by cluster
        cluster_texts = {}
        for label, text in zip(cluster_labels, text_examples):
            if str(label) not in cluster_texts:
                cluster_texts[str(label)] = []
            cluster_texts[str(label)].append(text.strip())
        
        # Limit examples per cluster if specified
        if max_examples_per_cluster is not None:
            for cluster_id in cluster_texts:
                cluster_texts[cluster_id] = cluster_texts[cluster_id][:max_examples_per_cluster]
        
        checkpoint_data = {
            'checkpoint_step': checkpoint_step,
            'clusters': cluster_texts
        }
        
        text_excerpts.append(checkpoint_data)
    
    return text_excerpts

def main():
    parser = argparse.ArgumentParser(description="Extract text excerpts for every checkpoint")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--neuron", type=int, required=True, help="Neuron number")
    parser.add_argument("--model", default="pythia70m", help="Model name (default: pythia70m)")
    parser.add_argument("--output", help="Output file (default: L{layer}N{neuron}_text_excerpts.json)")
    parser.add_argument("--max-examples", type=int, help="Maximum examples per cluster (default: all examples)")
    
    args = parser.parse_args()
    
    print(f"🔍 EXTRACTING TEXT EXCERPTS: L{args.layer}N{args.neuron} ({args.model})")
    print(f"=" * 60)
    
    # Extract text excerpts
    text_excerpts = extract_text_excerpts(args.layer, args.neuron, args.model, args.max_examples)
    
    if text_excerpts is None:
        return
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"L{args.layer}N{args.neuron}_text_excerpts.json")
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(text_excerpts, f, indent=2)
    
    print(f"📁 Text excerpts saved to {output_file}")
    print(f"📊 Extracted {len(text_excerpts)} checkpoints")
    
    if args.max_examples:
        print(f"📝 Limited to {args.max_examples} examples per cluster")
    
    # Show summary
    for checkpoint in text_excerpts:
        step = checkpoint['checkpoint_step']
        num_clusters = len(checkpoint['clusters'])
        total_examples = sum(len(texts) for texts in checkpoint['clusters'].values())
        print(f"  Step {step}: {num_clusters} clusters, {total_examples} total examples")

if __name__ == "__main__":
    main()
