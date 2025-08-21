#!/usr/bin/env python
"""
Script to analyze global statistics for multiple Pythia models (70m, 160m) 
and overlay them on the same plot for comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import json
from collections import defaultdict

def load_model_data(model_name):
    """Load all checkpoint data from CSV files for a specific model"""
    
    if model_name == "pythia70m":
        results_dir = Path("results/pythia70m")
    elif model_name == "pythia160m":
        results_dir = Path("results/pythia160m")
    elif model_name == "pythia410m":
        results_dir = Path("results/pythia410m")
    else:
        print(f"❌ Unknown model: {model_name}")
        return None, {}
    
    print(f"🔍 Loading data for {model_name} from {results_dir}...")
    
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
            # Extract layer and neuron from filename like "L0N0_pythia70m_ckpt_summary.csv"
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
            df['model'] = model_name  # Add model identifier
            all_data.append(df)
            
            # Calculate summary stats for this neuron
            neuron_stats[f"L{layer}N{neuron_num}"] = {
                'layer': layer,
                'neuron': neuron_num,
                'model': model_name,
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
        print(f"    ✅ Successfully loaded {len(neuron_stats)} neurons for {model_name}")
        return combined_df, neuron_stats
    else:
        return None, {}

def analyze_global_stats_multi(model_data_dict):
    """Analyze global statistics across multiple models"""
    
    print("\n" + "="*80)
    print("GLOBAL STATISTICS COMPARISON")
    print("="*80)
    
    for model_name, (combined_df, neuron_stats) in model_data_dict.items():
        print(f"\n📊 {model_name.upper()} STATISTICS")
        print("-" * 50)
        
        # Overall statistics
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
    
    return model_data_dict

def create_overlay_plots(model_data_dict):
    """Create overlay plots comparing multiple models - only all layers combined"""
    
    # Create a single panel figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color palette for models
    model_colors = {
        'pythia70m': '#1f77b4',   # Blue
        'pythia160m': '#d62728',  # Red
        'pythia410m': '#2ca02c'   # Green
    }
    
    # Line styles for different models
    line_styles = {
        'pythia70m': '-',
        'pythia160m': '-',
        'pythia410m': '-'
    }
    
    # Plot: Average clusters over time for ALL LAYERS COMBINED for each model
    for model_name, (combined_df, _) in model_data_dict.items():
        color = model_colors[model_name]
        linestyle = line_styles[model_name]
        
        checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
        cluster_means_total = []
        
        for step in checkpoint_steps:
            step_data = combined_df[combined_df['checkpoint_step'] == step]
            if not step_data.empty:
                cluster_means_total.append(step_data['num_clusters'].mean())
            else:
                cluster_means_total.append(np.nan)
        
        # Get layer range for legend
        layers = sorted(combined_df['layer'].unique())
        if model_name == "pythia410m":
            layer_range = "L0"  # Only Layer 0 for 410M
        else:
            layer_range = f"L{layers[0]}-L{layers[-1]}"
        
        # Format model name for legend
        if model_name == "pythia70m":
            display_name = "Pythia-70M"
        elif model_name == "pythia160m":
            display_name = "Pythia-160M"
        elif model_name == "pythia410m":
            display_name = "Pythia-410M"
        else:
            display_name = model_name
        
        ax.plot(checkpoint_steps, cluster_means_total, 
                linestyle=linestyle, color=color, linewidth=3, 
                label=f'{display_name} ({layer_range})')
    
    ax.set_title('Feature Clusters per Neuron over Pretraining', fontweight='bold', fontsize=15)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Average Clusters', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    results_dir = Path("results")
    output_file = results_dir / "global_statistics_comparison.pdf"
    print(f"\n📊 Global statistics comparison plot saved to {output_file}")
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    
    # Models to analyze
    models = ["pythia70m", "pythia160m", "pythia410m"]
    
    # Load data for all models
    model_data_dict = {}
    for model in models:
        combined_df, neuron_stats = load_model_data(model)
        if combined_df is not None:
            model_data_dict[model] = (combined_df, neuron_stats)
        else:
            print(f"❌ Failed to load data for {model}")
    
    if not model_data_dict:
        print("❌ No data loaded for any model")
        return
    
    # Analyze global statistics
    model_data_dict = analyze_global_stats_multi(model_data_dict)
    
    # Create plots
    print("\n🎨 Creating comparison plot...")
    create_overlay_plots(model_data_dict)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
