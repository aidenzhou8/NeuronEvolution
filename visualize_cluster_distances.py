#!/usr/bin/env python
"""
Script to visualize inter- and intra-cluster distances changing over checkpoints.
Supports different models via command line option.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import ast
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_model_data(model_name, results_dir="results"):
    """Load all CSV data for a specific model"""
    
    print(f"🔍 Loading data for model: {model_name}")
    
    # Find all CSV files for this model
    if model_name == "pythia-70m":
        csv_pattern = f"{results_dir}/**/*pythia70m*_ckpt_summary.csv"
    elif model_name == "pythia-160m":
        csv_pattern = f"{results_dir}/**/*pythia160m*_ckpt_summary.csv"
    elif model_name == "pythia-410m":
        csv_pattern = f"{results_dir}/**/*pythia410m*_ckpt_summary.csv"
    else:
        csv_pattern = f"{results_dir}/**/*{model_name}*_ckpt_summary.csv"
    
    csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print(f"❌ No CSV files found for model: {model_name}")
        print(f"   Searched pattern: {csv_pattern}")
        return None, {}
    
    print(f"   Found {len(csv_files)} CSV files")
    
    all_data = []
    layer_stats = defaultdict(list)
    
    for csv_file in csv_files:
        try:
            # Extract layer and neuron from filename
            filename = Path(csv_file).name
            parts = filename.split('_')[0]  # L4N540
            layer = int(parts[1:parts.find('N')])
            neuron = int(parts[parts.find('N')+1:])
            
            df = pd.read_csv(csv_file)
            df['layer'] = layer
            df['neuron'] = neuron
            df['layer_neuron'] = f"L{layer}N{neuron}"
            
            # Parse cluster_sizes string to dict
            df['cluster_sizes_dict'] = df['cluster_sizes'].apply(ast.literal_eval)
            
            # Calculate additional metrics
            df['mean_inter'] = df['mean_dist'] - df['mean_intra']
            df['cluster_efficiency'] = df['mean_inter'] / (df['mean_intra'] + 1e-8)  # Avoid division by zero
            
            all_data.append(df)
            layer_stats[layer].append({
                'neuron': neuron,
                'layer': layer,
                'data': df
            })
            
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"   ✅ Loaded data for {len(combined_df)} checkpoints across {len(layer_stats)} layers")
        return combined_df, layer_stats
    else:
        print(f"   ❌ No valid data found for model: {model_name}")
        return None, {}

def analyze_distance_evolution(combined_df, layer_stats):
    """Analyze how inter- and intra-cluster distances evolve over training"""
    
    print(f"\n📊 DISTANCE EVOLUTION ANALYSIS")
    print(f"=" * 50)
    
    # Overall statistics
    print(f"Total checkpoints: {len(combined_df)}")
    print(f"Total neurons: {combined_df['layer_neuron'].nunique()}")
    print(f"Layers: {sorted(combined_df['layer'].unique())}")
    print(f"Checkpoint range: {combined_df['checkpoint_step'].min()} to {combined_df['checkpoint_step'].max()}")
    
    # Distance statistics
    print(f"\n📏 DISTANCE STATISTICS")
    print(f"Mean intra-cluster distance: {combined_df['mean_intra'].mean():.4f} ± {combined_df['mean_intra'].std():.4f}")
    print(f"Mean inter-cluster distance: {combined_df['mean_inter'].mean():.4f} ± {combined_df['mean_inter'].std():.4f}")
    print(f"Mean cluster efficiency: {combined_df['cluster_efficiency'].mean():.4f} ± {combined_df['cluster_efficiency'].std():.4f}")
    
    # Per-layer analysis
    print(f"\n🎯 PER-LAYER DISTANCE ANALYSIS")
    for layer in sorted(layer_stats.keys()):
        layer_data = combined_df[combined_df['layer'] == layer]
        print(f"\n  Layer {layer}:")
        print(f"    Neurons: {len(layer_stats[layer])}")
        print(f"    Mean intra: {layer_data['mean_intra'].mean():.4f} ± {layer_data['mean_intra'].std():.4f}")
        print(f"    Mean inter: {layer_data['mean_inter'].mean():.4f} ± {layer_data['mean_inter'].std():.4f}")
        print(f"    Mean efficiency: {layer_data['cluster_efficiency'].mean():.4f} ± {layer_data['cluster_efficiency'].std():.4f}")
    
    return combined_df, layer_stats

def create_distance_plots(combined_df, layer_stats, model_name, output_dir="results"):
    """Create comprehensive distance evolution plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Cluster Distance Evolution: {model_name}', fontsize=14, fontweight='bold')
    
    # Color palette for layers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a6cee3', '#fb9a99']
    
    checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
    
    # Plot 1: Intra-cluster distances over time by layer
    for layer in sorted(layer_stats.keys()):
        layer_data = combined_df[combined_df['layer'] == layer]
        intra_means = []
        intra_stds = []
        
        for step in checkpoint_steps:
            step_data = layer_data[layer_data['checkpoint_step'] == step]
            if not step_data.empty:
                intra_means.append(step_data['mean_intra'].mean())
                intra_stds.append(step_data['mean_intra'].std())
            else:
                intra_means.append(np.nan)
                intra_stds.append(np.nan)
        
        color = colors[layer % len(colors)]
        ax1.plot(checkpoint_steps, intra_means, 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Layer {layer}')
        ax1.fill_between(checkpoint_steps, 
                        [m - s if not np.isnan(m) else np.nan for m, s in zip(intra_means, intra_stds)],
                        [m + s if not np.isnan(m) else np.nan for m, s in zip(intra_means, intra_stds)],
                        alpha=0.2, color=color)
    
    ax1.set_title('Intra-cluster Distances Over Training by Layer', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Training Step', fontsize=9)
    ax1.set_ylabel('Mean Intra-cluster Distance', fontsize=9)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inter-cluster distances over time by layer
    for layer in sorted(layer_stats.keys()):
        layer_data = combined_df[combined_df['layer'] == layer]
        inter_means = []
        inter_stds = []
        
        for step in checkpoint_steps:
            step_data = layer_data[layer_data['checkpoint_step'] == step]
            if not step_data.empty:
                inter_means.append(step_data['mean_inter'].mean())
                inter_stds.append(step_data['mean_inter'].std())
            else:
                inter_means.append(np.nan)
                inter_stds.append(np.nan)
        
        color = colors[layer % len(colors)]
        ax2.plot(checkpoint_steps, inter_means, 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Layer {layer}')
        ax2.fill_between(checkpoint_steps, 
                        [m - s if not np.isnan(m) else np.nan for m, s in zip(inter_means, inter_stds)],
                        [m + s if not np.isnan(m) else np.nan for m, s in zip(inter_means, inter_stds)],
                        alpha=0.2, color=color)
    
    ax2.set_title('Inter-cluster Distances Over Training by Layer', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Training Step', fontsize=9)
    ax2.set_ylabel('Mean Inter-cluster Distance', fontsize=9)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cluster efficiency (inter/intra ratio) over time
    for layer in sorted(layer_stats.keys()):
        layer_data = combined_df[combined_df['layer'] == layer]
        efficiency_means = []
        efficiency_stds = []
        
        for step in checkpoint_steps:
            step_data = layer_data[layer_data['checkpoint_step'] == step]
            if not step_data.empty:
                efficiency_means.append(step_data['cluster_efficiency'].mean())
                efficiency_stds.append(step_data['cluster_efficiency'].std())
            else:
                efficiency_means.append(np.nan)
                efficiency_stds.append(np.nan)
        
        color = colors[layer % len(colors)]
        ax3.plot(checkpoint_steps, efficiency_means, 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Layer {layer}')
        ax3.fill_between(checkpoint_steps, 
                        [m - s if not np.isnan(m) else np.nan for m, s in zip(efficiency_means, efficiency_stds)],
                        [m + s if not np.isnan(m) else np.nan for m, s in zip(efficiency_means, efficiency_stds)],
                        alpha=0.2, color=color)
    
    ax3.set_title('Cluster Efficiency (Inter/Intra Ratio) Over Training', fontweight='bold', fontsize=11)
    ax3.set_xlabel('Training Step', fontsize=9)
    ax3.set_ylabel('Cluster Efficiency', fontsize=9)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of intra vs inter distances (final checkpoint)
    final_checkpoint = combined_df['checkpoint_step'].max()
    final_data = combined_df[combined_df['checkpoint_step'] == final_checkpoint]
    
    for layer in sorted(layer_stats.keys()):
        layer_final = final_data[final_data['layer'] == layer]
        if not layer_final.empty:
            color = colors[layer % len(colors)]
            ax4.scatter(layer_final['mean_intra'], layer_final['mean_inter'], 
                       c=color, s=40, alpha=0.7, label=f'Layer {layer}')
    
    ax4.set_title(f'Intra vs Inter Distances (Final Checkpoint: {final_checkpoint})', fontweight='bold', fontsize=11)
    ax4.set_xlabel('Mean Intra-cluster Distance', fontsize=9)
    ax4.set_ylabel('Mean Inter-cluster Distance', fontsize=9)
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Add diagonal line for reference (inter = intra)
    min_val = min(ax4.get_xlim()[0], ax4.get_ylim()[0])
    max_val = max(ax4.get_xlim()[1], ax4.get_ylim()[1])
    ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Inter = Intra')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_dir) / f"cluster_distance_evolution_{model_name.replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Cluster distance evolution plot saved to {output_file}")
    plt.show()

def create_individual_neuron_plots(combined_df, layer_stats, model_name, output_dir="results", max_neurons=4):
    """Create individual plots for specific neurons showing distance evolution"""
    
    print(f"\n📈 Creating individual neuron plots (showing first {max_neurons} neurons per layer)...")
    
    for layer in sorted(layer_stats.keys())[:3]:  # Limit to first 3 layers
        layer_data = combined_df[combined_df['layer'] == layer]
        neurons = layer_data['neuron'].unique()[:max_neurons]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'Layer {layer} Neuron Distance Evolution: {model_name}', fontsize=12, fontweight='bold')
        
        plot_count = 0
        for i, neuron in enumerate(neurons):
            neuron_data = layer_data[layer_data['neuron'] == neuron].sort_values('checkpoint_step')
            
            if len(neuron_data) < 2:
                continue
                
            row = plot_count // 2
            col = plot_count % 2
            
            if row < 2:  # Only plot if we have space
                ax = axes[row, col]
                
                # Plot intra and inter distances
                ax.plot(neuron_data['checkpoint_step'], neuron_data['mean_intra'], 
                       'o-', color='#1f77b4', linewidth=2, markersize=3, label='Intra-cluster')
                ax.plot(neuron_data['checkpoint_step'], neuron_data['mean_inter'], 
                       's-', color='#ff7f0e', linewidth=2, markersize=3, label='Inter-cluster')
                
                ax.set_title(f'L{layer}N{neuron}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Training Step', fontsize=8)
                ax.set_ylabel('Distance', fontsize=8)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=7)
                
                plot_count += 1
        
        # Hide unused subplots
        for i in range(plot_count, 4):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = Path(output_dir) / f"layer_{layer}_neuron_distances_{model_name.replace('-', '_')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   📊 Layer {layer} neuron plots saved to {output_file}")
        plt.show()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Visualize cluster distance evolution over checkpoints')
    parser.add_argument('--model', default='pythia-160m', 
                       choices=['pythia-70m', 'pythia-160m', 'pythia-410m'],
                       help='Model to analyze')
    parser.add_argument('--output-dir', default='results', help='Output directory for plots')
    parser.add_argument('--individual-plots', action='store_true', 
                       help='Create individual neuron plots')
    
    args = parser.parse_args()
    
    print(f"🔍 CLUSTER DISTANCE EVOLUTION ANALYSIS")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    combined_df, layer_stats = load_model_data(args.model, args.output_dir)
    
    if combined_df is None:
        print("❌ No data found. Exiting.")
        return
    
    # Analyze distance evolution
    combined_df, layer_stats = analyze_distance_evolution(combined_df, layer_stats)
    
    # Create main plots
    print(f"\n📈 Creating distance evolution plots...")
    create_distance_plots(combined_df, layer_stats, args.model, args.output_dir)
    
    # Create individual neuron plots if requested
    if args.individual_plots:
        create_individual_neuron_plots(combined_df, layer_stats, args.model, args.output_dir)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Check the {args.output_dir}/ directory for generated plots")

if __name__ == "__main__":
    main()
