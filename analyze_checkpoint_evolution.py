#!/usr/bin/env python
"""
Script to analyze how cluster distance metrics evolve over checkpoints in detail.
"""

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
            df['cluster_efficiency'] = df['mean_inter'] / (df['mean_intra'] + 1e-8)
            
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

def analyze_evolution_trends(combined_df, layer_stats):
    """Analyze how metrics evolve over checkpoints"""
    
    print(f"\n📈 EVOLUTION TREND ANALYSIS")
    print(f"=" * 50)
    
    checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
    print(f"Checkpoint steps: {checkpoint_steps}")
    
    # Calculate trends for each metric
    trends = {}
    
    for metric in ['mean_intra', 'mean_inter', 'cluster_efficiency', 'num_clusters']:
        print(f"\n📊 {metric.upper().replace('_', ' ')} EVOLUTION:")
        
        # Overall trend
        overall_means = []
        for step in checkpoint_steps:
            step_data = combined_df[combined_df['checkpoint_step'] == step]
            if not step_data.empty:
                overall_means.append(step_data[metric].mean())
            else:
                overall_means.append(np.nan)
        
        # Calculate trend (slope)
        valid_indices = [i for i, x in enumerate(overall_means) if not np.isnan(x)]
        if len(valid_indices) >= 2:
            x = np.array(checkpoint_steps)[valid_indices]
            y = np.array(overall_means)[valid_indices]
            slope = np.polyfit(x, y, 1)[0]
            trends[metric] = slope
            
            print(f"  Overall trend: {slope:.6f} per step")
            print(f"  Early ({checkpoint_steps[0]}): {overall_means[0]:.4f}")
            print(f"  Late ({checkpoint_steps[-1]}): {overall_means[-1]:.4f}")
            print(f"  Change: {overall_means[-1] - overall_means[0]:+.4f}")
        
        # Per-layer trends
        print(f"  Per-layer trends:")
        for layer in sorted(layer_stats.keys()):
            layer_data = combined_df[combined_df['layer'] == layer]
            layer_means = []
            
            for step in checkpoint_steps:
                step_data = layer_data[layer_data['checkpoint_step'] == step]
                if not step_data.empty:
                    layer_means.append(step_data[metric].mean())
                else:
                    layer_means.append(np.nan)
            
            valid_indices = [i for i, x in enumerate(layer_means) if not np.isnan(x)]
            if len(valid_indices) >= 2:
                x = np.array(checkpoint_steps)[valid_indices]
                y = np.array(layer_means)[valid_indices]
                slope = np.polyfit(x, y, 1)[0]
                print(f"    L{layer}: {slope:.6f} per step")
    
    return trends, checkpoint_steps

def create_evolution_plots(combined_df, layer_stats, model_name, output_dir="results"):
    """Create detailed evolution plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Metric Evolution Over Checkpoints: {model_name}', fontsize=14, fontweight='bold')
    
    # Color palette for layers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#a6cee3', '#fb9a99']
    
    checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
    metrics = ['mean_intra', 'mean_inter', 'cluster_efficiency', 'num_clusters', 'n_examples']
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Plot each layer
        for layer in sorted(layer_stats.keys()):
            layer_data = combined_df[combined_df['layer'] == layer]
            metric_means = []
            metric_stds = []
            
            for step in checkpoint_steps:
                step_data = layer_data[layer_data['checkpoint_step'] == step]
                if not step_data.empty:
                    metric_means.append(step_data[metric].mean())
                    metric_stds.append(step_data[metric].std())
                else:
                    metric_means.append(np.nan)
                    metric_stds.append(np.nan)
            
            color = colors[layer % len(colors)]
            ax.plot(checkpoint_steps, metric_means, 'o-', color=color, 
                   linewidth=2, markersize=3, label=f'Layer {layer}')
            ax.fill_between(checkpoint_steps, 
                          [m - s if not np.isnan(m) else np.nan for m, s in zip(metric_means, metric_stds)],
                          [m + s if not np.isnan(m) else np.nan for m, s in zip(metric_means, metric_stds)],
                          alpha=0.2, color=color)
        
        # Add overall trend line
        overall_means = []
        for step in checkpoint_steps:
            step_data = combined_df[combined_df['checkpoint_step'] == step]
            if not step_data.empty:
                overall_means.append(step_data[metric].mean())
            else:
                overall_means.append(np.nan)
        
        valid_indices = [i for i, x in enumerate(overall_means) if not np.isnan(x)]
        if len(valid_indices) >= 2:
            x = np.array(checkpoint_steps)[valid_indices]
            y = np.array(overall_means)[valid_indices]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'k--', linewidth=2, alpha=0.7, label='Overall trend')
        
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Training Step', fontsize=8)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=8)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Hide the last subplot if we have an odd number of metrics
    if len(metrics) < 6:
        axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_dir) / f"metric_evolution_{model_name.replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Metric evolution plot saved to {output_file}")
    plt.show()

def create_phase_analysis(combined_df, layer_stats, model_name, output_dir="results"):
    """Analyze different training phases"""
    
    print(f"\n⏰ PHASE ANALYSIS")
    print(f"=" * 30)
    
    # Define training phases
    early_phase = combined_df[combined_df['checkpoint_step'] <= 30000]
    mid_phase = combined_df[(combined_df['checkpoint_step'] > 30000) & (combined_df['checkpoint_step'] <= 100000)]
    late_phase = combined_df[combined_df['checkpoint_step'] > 100000]
    
    phases = {
        'Early (≤30k)': early_phase,
        'Mid (30k-100k)': mid_phase,
        'Late (>100k)': late_phase
    }
    
    metrics = ['mean_intra', 'mean_inter', 'cluster_efficiency', 'num_clusters']
    
    for metric in metrics:
        print(f"\n📊 {metric.upper().replace('_', ' ')} by Phase:")
        for phase_name, phase_data in phases.items():
            if not phase_data.empty:
                mean_val = phase_data[metric].mean()
                std_val = phase_data[metric].std()
                print(f"  {phase_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Create phase comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Phase Comparison: {model_name}', fontsize=14, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        phase_data = []
        phase_labels = []
        
        for phase_name, phase_df in phases.items():
            if not phase_df.empty:
                phase_data.append(phase_df[metric].values)
                phase_labels.append(phase_name)
        
        if phase_data:
            ax.boxplot(phase_data, labels=phase_labels)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path(output_dir) / f"phase_comparison_{model_name.replace('-', '_')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Phase comparison plot saved to {output_file}")
    plt.show()

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Analyze metric evolution over checkpoints')
    parser.add_argument('--model', default='pythia-160m', 
                       choices=['pythia-70m', 'pythia-160m', 'pythia-410m'],
                       help='Model to analyze')
    parser.add_argument('--output-dir', default='results', help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"🔍 METRIC EVOLUTION ANALYSIS")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    combined_df, layer_stats = load_model_data(args.model, args.output_dir)
    
    if combined_df is None:
        print("❌ No data found. Exiting.")
        return
    
    # Analyze evolution trends
    trends, checkpoint_steps = analyze_evolution_trends(combined_df, layer_stats)
    
    # Create evolution plots
    print(f"\n📈 Creating evolution plots...")
    create_evolution_plots(combined_df, layer_stats, args.model, args.output_dir)
    
    # Create phase analysis
    print(f"\n⏰ Creating phase analysis...")
    create_phase_analysis(combined_df, layer_stats, args.model, args.output_dir)
    
    print(f"\n🎉 Evolution analysis complete!")
    print(f"📁 Check the {args.output_dir}/ directory for generated plots")

if __name__ == "__main__":
    import argparse
    main()

