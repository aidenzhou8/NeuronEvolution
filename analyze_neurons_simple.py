#!/usr/bin/env python
"""
Simple script to analyze individual neurons from JSONL files
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_neuron(layer, neuron, checkpoint_step=None):
    """Analyze a specific neuron at a specific checkpoint and save to text file"""
    
    results_dir = Path("results")
    jsonl_file = results_dir / f"L{layer}N{neuron}_pythia70m_ckpt_series.jsonl"
    
    if not jsonl_file.exists():
        print(f"❌ File not found: {jsonl_file}")
        return
    
    # Load all checkpoint data
    checkpoint_data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            checkpoint_data.append(json.loads(line))
    
    # Prepare output text
    output_lines = []
    output_lines.append(f"=== Analysis for L{layer}N{neuron} ===")
    output_lines.append(f"Total checkpoints: {len(checkpoint_data)}")
    
    # Show evolution over time
    output_lines.append(f"\n📈 Cluster Evolution:")
    for data in checkpoint_data:
        step = data['checkpoint_step']
        metrics = data['metrics']
        output_lines.append(f"  Step {step}: {metrics['num_clusters']} clusters "
                           f"(largest: {metrics['largest_cluster_size']}, "
                           f"single: {metrics['single_element_clusters']})")
    
    # Analyze specific checkpoint
    if checkpoint_step is None:
        # Use the last checkpoint
        target_data = checkpoint_data[-1]
        checkpoint_step = target_data['checkpoint_step']
    else:
        # Find the closest checkpoint
        target_data = None
        for data in checkpoint_data:
            if data['checkpoint_step'] >= checkpoint_step:
                target_data = data
                break
        if target_data is None:
            target_data = checkpoint_data[-1]
            checkpoint_step = target_data['checkpoint_step']
    
    output_lines.append(f"\n🔍 Detailed Analysis at Step {checkpoint_step}:")
    
    metrics = target_data['metrics']
    labels = target_data['cluster_labels']
    text_examples = target_data['text_examples']
    
    output_lines.append(f"  Total embeddings: {metrics['embeddings']}")
    output_lines.append(f"  Number of clusters: {metrics['num_clusters']}")
    output_lines.append(f"  Single-element clusters: {metrics['single_element_clusters']}")
    output_lines.append(f"  Largest cluster size: {metrics['largest_cluster_size']}")
    smallest_key = 'smallest_cluster_size' if 'smallest_cluster_size' in metrics else 'least_cluster_size'
    output_lines.append(f"  Smallest cluster size: {metrics[smallest_key]}")
    output_lines.append(f"  Cluster sizes: {metrics['cluster_sizes']}")
    
    # Group text examples by cluster
    cluster_texts = defaultdict(list)
    for i, (label, text) in enumerate(zip(labels, text_examples)):
        cluster_texts[label].append(text)
    
    output_lines.append(f"\n📝 Text Examples by Cluster:")
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        output_lines.append(f"\n  Cluster {cluster_id} ({len(texts)} examples):")
        for i, text in enumerate(texts[:3]):  # Show first 3 examples
            output_lines.append(f"    {i+1}. {text}")
        if len(texts) > 3:
            output_lines.append(f"    ... and {len(texts) - 3} more examples")
    
    # Save to text file
    output_file = results_dir / f"L{layer}N{neuron}_analysis.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ Analysis saved to {output_file}")
    
    # Also print to console
    print('\n'.join(output_lines))

def main():
    parser = argparse.ArgumentParser(description="Analyze individual neurons")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--neuron", type=int, required=True, help="Neuron number")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step (optional)")
    
    args = parser.parse_args()
    analyze_neuron(args.layer, args.neuron, args.checkpoint)

if __name__ == "__main__":
    main() 