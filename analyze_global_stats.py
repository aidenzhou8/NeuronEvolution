#!/usr/bin/env python
"""
Script to analyze global statistics for Pythia-70M by processing all CSV files in results/pythia70m
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from collections import defaultdict

def load_all_csv_data():
    """Load all checkpoint data from CSV files in results/pythia70m"""
    
    results_dir = Path("results/pythia160m")
    
    print(f"🔍 Loading data from {results_dir}...")
    
    all_data = []
    neuron_stats = {}
    
    # Find all CSV files in the directory
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"    ⚠️  No CSV files found in {results_dir}")
        return None, {}
        
    print(f"    Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        # Extract neuron number and layer from filename
        filename = csv_file.name
        if 'L' in filename and 'N' in filename:
            # Extract layer and neuron from filename like "L0N0_pythia410m_ckpt_summary.csv"
            layer_part = filename.split('N')[0]  # "L0"
            neuron_part = filename.split('N')[1].split('_')[0]  # "0"
            
            try:
                layer = int(layer_part[1:])  # Remove 'L' and convert to int
                neuron_num = int(neuron_part)
            except ValueError:
                print(f"    ⚠️  Could not parse layer/neuron from {filename}")
                continue
        else:
            print(f"    ⚠️  Unexpected filename format: {filename}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            df['neuron'] = neuron_num
            df['layer'] = layer
            all_data.append(df)
            
            # Calculate summary stats for this neuron
            neuron_stats[f"L{layer}N{neuron_num}"] = {
                'layer': layer,
                'neuron': neuron_num,
                'mean_clusters': df['num_clusters'].mean(),
                'std_clusters': df['num_clusters'].std(),
                'min_clusters': df['num_clusters'].min(),
                'max_clusters': df['num_clusters'].max(),
                'final_clusters': df['num_clusters'].iloc[-1],
                'initial_clusters': df['num_clusters'].iloc[0],
                'cluster_change': df['num_clusters'].iloc[-1] - df['num_clusters'].iloc[0],
                'mean_examples': df['n_examples'].mean(),
                'trend_slope': np.polyfit(df['checkpoint_step'], df['num_clusters'], 1)[0],
                'checkpoint_range': (df['checkpoint_step'].min(), df['checkpoint_step'].max())
            }
            
            print(f"    ✅ Loaded {filename}: Layer {layer}, Neuron {neuron_num}")
            
        except Exception as e:
            print(f"    ❌ Error loading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"    ✅ Successfully loaded {len(neuron_stats)} neurons")
        return combined_df, neuron_stats
    else:
        return None, {}

def analyze_global_stats(combined_df, neuron_stats):
    """Analyze global statistics across all neurons"""
    
    print("\n" + "="*80)
    print("GLOBAL STATISTICS FOR PYTHIA-70M")
    print("="*80)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS")
    total_neurons = len(neuron_stats)
    print(f"Total neurons analyzed: {total_neurons}")
    print(f"Total checkpoints: {len(combined_df)}")
    print(f"Checkpoint range: {combined_df['checkpoint_step'].min()} to {combined_df['checkpoint_step'].max()}")
    
    # Get unique layers
    layers = sorted(combined_df['layer'].unique())
    print(f"Layers with data: {layers}")
    
    # Per-layer statistics
    print(f"\n🎯 PER-LAYER STATISTICS")
    for layer in layers:
        layer_data = combined_df[combined_df['layer'] == layer]
        layer_neurons = {k: v for k, v in neuron_stats.items() if v['layer'] == layer}
        
        print(f"\n  Layer {layer}:")
        print(f"    Neurons: {len(layer_neurons)}")
        print(f"    Mean clusters: {layer_data['num_clusters'].mean():.2f} ± {layer_data['num_clusters'].std():.2f}")
        print(f"    Median clusters: {layer_data['num_clusters'].median():.1f}")
        print(f"    Cluster range: {layer_data['num_clusters'].min()} to {layer_data['num_clusters'].max()}")
        
        # Evolution stats for this layer
        cluster_changes = [stats['cluster_change'] for stats in layer_neurons.values()]
        slopes = [stats['trend_slope'] for stats in layer_neurons.values()]
        
        print(f"    Mean cluster change: {np.mean(cluster_changes):.2f} ± {np.std(cluster_changes):.2f}")
        print(f"    Mean trend slope: {np.mean(slopes):.6f} ± {np.std(slopes):.6f}")
        print(f"    Decreasing clusters: {sum(1 for c in cluster_changes if c < 0)}/{len(cluster_changes)}")
        print(f"    Increasing clusters: {sum(1 for c in cluster_changes if c > 0)}/{len(cluster_changes)}")
    
    # Cross-layer comparisons
    print(f"\n🔄 CROSS-LAYER COMPARISONS")
    
    # Compare final cluster distributions
    layer_final_clusters = {}
    for layer in layers:
        layer_final_clusters[layer] = [stats['final_clusters'] for stats in neuron_stats.values() if stats['layer'] == layer]
    
    print(f"  Final cluster statistics by layer:")
    for layer in sorted(layer_final_clusters.keys()):
        clusters = layer_final_clusters[layer]
        print(f"    L{layer}: {np.mean(clusters):.2f} ± {np.std(clusters):.2f} (range: {min(clusters)}-{max(clusters)})")
    
    # Phase analysis across layers
    print(f"\n⏰ PHASE ANALYSIS ACROSS LAYERS")
    early_phase = combined_df[combined_df['checkpoint_step'] <= 30000]
    late_phase = combined_df[combined_df['checkpoint_step'] > 100000]
    
    print(f"  Early phase (≤30k steps): {early_phase['num_clusters'].mean():.2f} ± {early_phase['num_clusters'].std():.2f} clusters")
    print(f"  Late phase (>100k steps): {late_phase['num_clusters'].mean():.2f} ± {late_phase['num_clusters'].std():.2f} clusters")
    print(f"  Overall decrease: {late_phase['num_clusters'].mean() - early_phase['num_clusters'].mean():+.2f} clusters")
    
    # Layer-specific phase analysis
    for layer in layers:
        layer_early = early_phase[early_phase['layer'] == layer]
        layer_late = late_phase[late_phase['layer'] == layer]
        
        if len(layer_early) > 0 and len(layer_late) > 0:
            change = layer_late['num_clusters'].mean() - layer_early['num_clusters'].mean()
            print(f"    L{layer}: {change:+.2f} clusters")
    
    return combined_df, neuron_stats

def create_global_plots(combined_df, neuron_stats):
    """Create simplified global visualization plots - per layer only"""
    
    # Create a single panel figure for per layer analysis only
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Color palette for layers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a6cee3', '#fb9a99']
    
    # Get unique layers
    layers = sorted(combined_df['layer'].unique())
    
    # Plot 1: Average clusters over time by layer
    checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
    
    for i, layer in enumerate(layers):
        layer_data = combined_df[combined_df['layer'] == layer]
        cluster_means = []
        cluster_stds = []
        
        for step in checkpoint_steps:
            step_data = layer_data[layer_data['checkpoint_step'] == step]
            if not step_data.empty:
                cluster_means.append(step_data['num_clusters'].mean())
                cluster_stds.append(step_data['num_clusters'].std())
            else:
                cluster_means.append(np.nan)
                cluster_stds.append(np.nan)
        
        color = colors[i % len(colors)]
        # Count neurons in this layer
        layer_neurons = len([k for k, v in neuron_stats.items() if v['layer'] == layer])
        ax1.plot(checkpoint_steps, cluster_means, 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Layer {layer} (n={layer_neurons})')
    
    ax1.set_title('Pythia-160M: Clusters per Neuron over Pretraining (by Layer)', fontweight='bold', fontsize=15)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Average Clusters', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path("results")
    output_file = results_dir / "global_statistics_pythia160m.pdf"
    print(f"\n📊 Global statistics plot saved to {output_file}")
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.show()

def create_layer_heatmap(combined_df, neuron_stats):
    """Placeholder function - heatmaps removed"""
    pass

def main():
    """Main analysis function"""
    
    print("🔍 Loading data from all CSV files in results/pythia70m...")
    combined_df, neuron_stats = load_all_csv_data()
    
    if combined_df is None or len(neuron_stats) == 0:
        print("❌ No data found! Please check that CSV files exist in results/pythia410m/")
        return
    
    total_neurons = len(neuron_stats)
    layers = sorted(combined_df['layer'].unique())
    print(f"✅ Loaded data for {total_neurons} neurons across {len(layers)} layers")
    
    # Analyze global statistics
    combined_df, neuron_stats = analyze_global_stats(combined_df, neuron_stats)
    
    # Create simplified visualizations
    print(f"\n📈 Creating simplified visualizations...")
    
    # Create per-layer plot only
    create_global_plots(combined_df, neuron_stats)
    
    print(f"\n🎉 Global analysis complete!")
    print(f"📁 Check the results/ directory for generated plots")
    print(f"📊 Analyzed {total_neurons} neurons across {len(layers)} layers")

if __name__ == "__main__":
    main() 