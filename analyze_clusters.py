#!/usr/bin/env python
"""
Script to analyze clusters and return activated text excerpts for each cluster
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_clusters(layer, neuron):
    # Construct file file path
    results_dir = Path("results")
    metrics_file = results_dir / f"L{layer}N{neuron}_metrics.json"
    
    # Load results
    with open(metrics_file, "r") as f:
        data = json.load(f)
    
    metrics = data["metrics"]
    labels = data["labels"]
    text_examples = data.get("text_examples", [])
    
    print(f"=== Cluster Analysis for L{layer}N{neuron} ===")
    print(f"Total embeddings: {metrics['embeddings']}")
    print(f"Number of clusters: {metrics['num_clusters']}")
    print(f"Single-element clusters: {metrics['single_element_clusters']}")
    print(f"Largest cluster size: {metrics['largest_cluster_size']}")
    print(f"Smallest cluster size: {metrics['smallest_cluster_size']}")
    
    print(f"\nCluster sizes: {metrics['cluster_sizes']}")
    
    # Group text examples by cluster
    cluster_texts = defaultdict(list)
    for i, (label, text) in enumerate(zip(labels, text_examples)):
        cluster_texts[label].append(text)
    
    print(f"\n=== Text Examples by Cluster ===")
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        print(f"\nCluster {cluster_id} ({len(texts)} examples):")
        for i, text in enumerate(texts[:5]):  # Show first 5 examples
            print(f"  {i+1}. {text}")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more examples")

def main():
    parser = argparse.ArgumentParser(description="Analyze clusters for a specific neuron")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--neuron", type=int, required=True, help="Neuron number")
    
    args = parser.parse_args()
    analyze_clusters(args.layer, args.neuron)

if __name__ == "__main__":
    main() 