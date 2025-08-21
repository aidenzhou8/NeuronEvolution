#!/usr/bin/env python
"""
Lightweight script to analyze clusters and return activated text excerpts for each cluster over a neuron's checkpoints
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import sys
from datetime import datetime

def load_checkpoint_data(layer, neuron, model="pythia70m"):
    """Load checkpoint data for a specific neuron"""
    
    results_dir = Path("results") / model
    jsonl_file = results_dir / f"L{layer}N{neuron}_{model}_ckpt_series.jsonl"
    
    if not jsonl_file.exists():
        print(f"❌ File not found: {jsonl_file}")
        return None
    
    print(f"🔍 Loading checkpoint data from {jsonl_file}")
    
    checkpoint_data = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                checkpoint_data.append(data)
    
    print(f"   Loaded {len(checkpoint_data)} checkpoints")
    return checkpoint_data

def analyze_clusters_over_checkpoints(checkpoint_data, max_examples_per_cluster=3, save_output=False, output_dir=None):
    """Analyze clusters and text examples over all checkpoints"""
    
    print(f"\n📊 CLUSTER ANALYSIS OVER CHECKPOINTS")
    print(f"=" * 60)
    
    all_checkpoint_data = []
    
    for i, checkpoint in enumerate(checkpoint_data):
        checkpoint_step = checkpoint['checkpoint_step']
        metrics = checkpoint['metrics']
        cluster_labels = checkpoint['cluster_labels']
        text_examples = checkpoint['text_examples']
        
        print(f"\n🔹 Checkpoint {i+1}: Step {checkpoint_step}")
        print(f"   Clusters: {metrics['num_clusters']}")
        print(f"   Cluster sizes: {metrics['cluster_sizes']}")
        print(f"   Mean intra-cluster distance: {metrics['mean_intra']:.4f}")
        
        # Group text examples by cluster
        cluster_texts = defaultdict(list)
        for label, text in zip(cluster_labels, text_examples):
            cluster_texts[label].append(text)
        
        checkpoint_data_dict = {
            'checkpoint_step': checkpoint_step,
            'checkpoint_index': i + 1,
            'num_clusters': metrics['num_clusters'],
            'cluster_sizes': metrics['cluster_sizes'],
            'mean_intra': metrics['mean_intra'],
            'clusters': {}
        }
        
        # Show text examples for each cluster
        for cluster_id in sorted(cluster_texts.keys()):
            texts = cluster_texts[cluster_id]
            cluster_size = len(texts)
            
            print(f"\n   📝 Cluster {cluster_id} ({cluster_size} examples):")
            
            # Show first few examples
            for j, text in enumerate(texts[:max_examples_per_cluster]):
                # Clean up the text a bit
                clean_text = text.strip()
                if len(clean_text) > 80:
                    clean_text = clean_text[:77] + "..."
                print(f"     {j+1}. {clean_text}")
            
            if len(texts) > max_examples_per_cluster:
                print(f"     ... and {len(texts) - max_examples_per_cluster} more examples")
            
            # Store cluster data for saving
            checkpoint_data_dict['clusters'][str(cluster_id)] = {
                'size': cluster_size,
                'examples': texts[:max_examples_per_cluster],
                'total_examples': texts
            }
        
        all_checkpoint_data.append(checkpoint_data_dict)
        
        # Add a separator between checkpoints
        if i < len(checkpoint_data) - 1:
            print(f"\n" + "-" * 40)
    
    # Save results if requested
    if save_output and output_dir:
        # Save detailed checkpoint analysis as JSON
        checkpoint_analysis_file = output_dir / "checkpoint_analysis.json"
        with open(checkpoint_analysis_file, 'w') as f:
            json.dump(all_checkpoint_data, f, indent=2)
        print(f"\n📁 Checkpoint analysis saved to {checkpoint_analysis_file}")
    
    return all_checkpoint_data

def analyze_cluster_evolution(checkpoint_data, save_output=False, output_dir=None):
    """Analyze how clusters evolve over time"""
    
    print(f"\n📈 CLUSTER EVOLUTION ANALYSIS")
    print(f"=" * 60)
    
    evolution_data = []
    
    for checkpoint in checkpoint_data:
        checkpoint_step = checkpoint['checkpoint_step']
        metrics = checkpoint['metrics']
        
        evolution_data.append({
            'step': checkpoint_step,
            'num_clusters': metrics['num_clusters'],
            'mean_intra': metrics['mean_intra'],
            'mean_inter': metrics.get('mean_inter', float('nan')),
            'largest_cluster': metrics['largest_cluster_size'],
            'single_element_clusters': metrics['single_element_clusters']
        })
    
    # Create summary
    df = pd.DataFrame(evolution_data)
    
    print(f"Training steps: {df['step'].min()} to {df['step'].max()}")
    print(f"Cluster evolution:")
    
    for _, row in df.iterrows():
        print(f"  Step {int(row['step']):6d}: {int(row['num_clusters']):2d} clusters "
              f"(largest: {int(row['largest_cluster']):3d}, "
              f"single: {int(row['single_element_clusters']):2d}, "
              f"intra: {row['mean_intra']:.4f})")
    
    # Identify key transitions
    print(f"\n🔍 KEY TRANSITIONS:")
    transitions = []
    for i in range(1, len(df)):
        prev_clusters = df.iloc[i-1]['num_clusters']
        curr_clusters = df.iloc[i]['num_clusters']
        
        if curr_clusters != prev_clusters:
            step = int(df.iloc[i]['step'])
            change = curr_clusters - prev_clusters
            direction = "increased" if change > 0 else "decreased"
            print(f"  Step {step}: {int(prev_clusters)} → {int(curr_clusters)} clusters ({direction})")
            transitions.append({
                'step': step,
                'prev_clusters': int(prev_clusters),
                'curr_clusters': int(curr_clusters),
                'change': int(change),
                'direction': direction
            })
    
    # Save results if requested
    if save_output and output_dir:
        # Save evolution data as CSV
        evolution_file = output_dir / "cluster_evolution.csv"
        df.to_csv(evolution_file, index=False)
        print(f"\n📁 Evolution data saved to {evolution_file}")
        
        # Save transitions as JSON
        transitions_file = output_dir / "key_transitions.json"
        with open(transitions_file, 'w') as f:
            json.dump(transitions, f, indent=2)
        print(f"📁 Key transitions saved to {transitions_file}")
    
    return df, transitions

def analyze_specific_checkpoint(checkpoint_data, target_step=None, save_output=False, output_dir=None):
    """Analyze a specific checkpoint in detail"""
    
    if target_step is None:
        # Use the final checkpoint
        checkpoint = checkpoint_data[-1]
        target_step = checkpoint['checkpoint_step']
    else:
        # Find the checkpoint closest to target_step
        checkpoint = min(checkpoint_data, key=lambda x: abs(x['checkpoint_step'] - target_step))
        target_step = checkpoint['checkpoint_step']
    
    print(f"\n🎯 DETAILED ANALYSIS: Checkpoint Step {target_step}")
    print(f"=" * 60)
    
    metrics = checkpoint['metrics']
    cluster_labels = checkpoint['cluster_labels']
    text_examples = checkpoint['text_examples']
    
    print(f"📊 Metrics:")
    print(f"   Number of clusters: {metrics['num_clusters']}")
    print(f"   Total embeddings: {metrics['embeddings']}")
    print(f"   Mean intra-cluster distance: {metrics['mean_intra']:.4f}")
    if 'mean_inter' in metrics and not pd.isna(metrics['mean_inter']):
        print(f"   Mean inter-cluster distance: {metrics['mean_inter']:.4f}")
    print(f"   Largest cluster size: {metrics['largest_cluster_size']}")
    print(f"   Single-element clusters: {metrics['single_element_clusters']}")
    
    # Group text examples by cluster
    cluster_texts = defaultdict(list)
    for label, text in zip(cluster_labels, text_examples):
        cluster_texts[label].append(text)
    
    detailed_data = {
        'checkpoint_step': target_step,
        'metrics': {
            'num_clusters': metrics['num_clusters'],
            'embeddings': metrics['embeddings'],
            'mean_intra': metrics['mean_intra'],
            'mean_inter': metrics.get('mean_inter', None),
            'largest_cluster_size': metrics['largest_cluster_size'],
            'single_element_clusters': metrics['single_element_clusters']
        },
        'clusters': {}
    }
    
    print(f"\n📝 Text Examples by Cluster:")
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        cluster_size = len(texts)
        
        print(f"\n   Cluster {cluster_id} ({cluster_size} examples):")
        
        # Show all examples for this cluster
        for j, text in enumerate(texts):
            clean_text = text.strip()
            if len(clean_text) > 100:
                clean_text = clean_text[:97] + "..."
            print(f"     {j+1:2d}. {clean_text}")
        
        # Store cluster data for saving
        detailed_data['clusters'][str(cluster_id)] = {
            'size': cluster_size,
            'examples': texts
        }
    
    # Save results if requested
    if save_output and output_dir:
        # Save detailed checkpoint analysis as JSON
        detailed_file = output_dir / f"detailed_checkpoint_{target_step}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        print(f"\n📁 Detailed checkpoint analysis saved to {detailed_file}")
    
    return detailed_data

def main():
    parser = argparse.ArgumentParser(description="Analyze clusters over checkpoints for a neuron")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--neuron", type=int, required=True, help="Neuron number")
    parser.add_argument("--model", default="pythia70m", help="Model name (default: pythia70m)")
    parser.add_argument("--max-examples", type=int, default=3, help="Max examples per cluster to show (default: 3)")
    parser.add_argument("--checkpoint", type=int, help="Specific checkpoint step to analyze in detail")
    parser.add_argument("--evolution-only", action="store_true", help="Show only evolution analysis")
    parser.add_argument("--detail-only", action="store_true", help="Show only detailed checkpoint analysis")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    parser.add_argument("--output-dir", help="Output directory for saved files (default: results/analysis_L{layer}N{neuron})")
    
    args = parser.parse_args()
    
    print(f"🔍 CLUSTER ANALYSIS: L{args.layer}N{args.neuron} ({args.model})")
    print(f"=" * 60)
    
    # Create output directory if saving
    output_dir = None
    if args.save:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results") / f"analysis_L{args.layer}N{args.neuron}_{timestamp}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Results will be saved to: {output_dir}")
        
        # Save analysis metadata
        metadata = {
            'analysis_time': datetime.now().isoformat(),
            'layer': args.layer,
            'neuron': args.neuron,
            'model': args.model,
            'max_examples': args.max_examples,
            'checkpoint': args.checkpoint,
            'evolution_only': args.evolution_only,
            'detail_only': args.detail_only
        }
        
        metadata_file = output_dir / "analysis_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📁 Analysis metadata saved to {metadata_file}")
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(args.layer, args.neuron, args.model)
    
    if checkpoint_data is None:
        return
    
    # Run analyses based on arguments
    if args.detail_only:
        analyze_specific_checkpoint(checkpoint_data, args.checkpoint, args.save, output_dir)
    elif args.evolution_only:
        analyze_cluster_evolution(checkpoint_data, args.save, output_dir)
    else:
        # Show all analyses
        analyze_cluster_evolution(checkpoint_data, args.save, output_dir)
        analyze_clusters_over_checkpoints(checkpoint_data, args.max_examples, args.save, output_dir)
        
        if args.checkpoint:
            analyze_specific_checkpoint(checkpoint_data, args.checkpoint, args.save, output_dir)
    
    if args.save:
        print(f"\n🎉 Analysis complete! All results saved to {output_dir}")

if __name__ == "__main__":
    main() 