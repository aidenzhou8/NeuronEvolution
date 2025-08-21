#!/usr/bin/env python
"""
Script to analyze how metrics evolve for specialized vs polysemantic neurons.
Compares neurons with 1-2 clusters vs 10+ clusters at final checkpoint.
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
    
    # Find all CSV files for this model in the specific model directory
    if model_name == "pythia-70m":
        model_dir = f"{results_dir}/pythia70m"
        csv_pattern = f"{model_dir}/*.csv"
        csv_files = glob.glob(csv_pattern)
    elif model_name == "pythia-160m":
        model_dir = f"{results_dir}/pythia160m"
        csv_pattern = f"{model_dir}/*.csv"
        csv_files = glob.glob(csv_pattern)
    elif model_name == "pythia-410m":
        model_dir = f"{results_dir}/pythia410m"
        csv_pattern = f"{model_dir}/*.csv"
        csv_files = glob.glob(csv_pattern)
    else:
        # Fallback to the original pattern for other models
        csv_pattern = f"{results_dir}/**/*{model_name}*_ckpt_summary.csv"
        csv_files = glob.glob(csv_pattern, recursive=True)
    
    if not csv_files:
        print(f"❌ No CSV files found for model: {model_name}")
        return None, {}
    
    print(f"   Found {len(csv_files)} CSV files")
    
    all_data = []
    neuron_data = {}
    
    for csv_file in csv_files:
        try:
            # Extract layer and neuron from filename
            filename = Path(csv_file).name
            parts = filename.split('_')[0]  # L4N540
            layer = int(parts[1:parts.find('N')])
            neuron = int(parts[parts.find('N')+1:])
            neuron_id = f"L{layer}N{neuron}"
            
            df = pd.read_csv(csv_file)
            df['layer'] = layer
            df['neuron'] = neuron
            df['neuron_id'] = neuron_id
            
            # Calculate additional metrics
            df['mean_inter'] = df['mean_dist'] - df['mean_intra']
            df['cluster_efficiency'] = df['mean_inter'] / (df['mean_intra'] + 1e-8)
            
            all_data.append(df)
            neuron_data[neuron_id] = df
            
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"   ✅ Loaded data for {len(combined_df)} checkpoints across {len(neuron_data)} neurons")
        return combined_df, neuron_data
    else:
        print(f"   ❌ No valid data found for model: {model_name}")
        return None, {}

def classify_neurons(neuron_data, final_checkpoint=143000):
    """Classify neurons as specialized or polysemantic based on final checkpoint"""
    
    print(f"\n🎯 CLASSIFYING NEURONS (Final checkpoint: {final_checkpoint})")
    print(f"=" * 50)
    
    specialized_neurons = []
    polysemantic_neurons = []
    
    for neuron_id, df in neuron_data.items():
        final_data = df[df['checkpoint_step'] == final_checkpoint]
        if not final_data.empty:
            final_clusters = final_data['num_clusters'].iloc[0]
            
            if final_clusters == 1:
                specialized_neurons.append(neuron_id)
            elif final_clusters >= 8:
                polysemantic_neurons.append(neuron_id)
    
    print(f"Specialized neurons (1 cluster): {len(specialized_neurons)}")
    print(f"Polysemantic neurons (10+ clusters): {len(polysemantic_neurons)}")
    
    # Show some examples
    if specialized_neurons:
        print(f"\nExamples of specialized neurons:")
        for neuron in specialized_neurons[:5]:
            final_clusters = neuron_data[neuron][neuron_data[neuron]['checkpoint_step'] == final_checkpoint]['num_clusters'].iloc[0]
            print(f"  {neuron}: {final_clusters} clusters")
    
    if polysemantic_neurons:
        print(f"\nExamples of polysemantic neurons:")
        for neuron in polysemantic_neurons[:5]:
            final_clusters = neuron_data[neuron][neuron_data[neuron]['checkpoint_step'] == final_checkpoint]['num_clusters'].iloc[0]
            print(f"  {neuron}: {final_clusters} clusters")
    
    return specialized_neurons, polysemantic_neurons

def analyze_evolution_by_type(combined_df, specialized_neurons, polysemantic_neurons, model_name):
    """Analyze how metrics evolve for each neuron type"""
    
    print(f"\n📈 EVOLUTION ANALYSIS BY NEURON TYPE")
    print(f"=" * 50)
    
    checkpoint_steps = sorted(combined_df['checkpoint_step'].unique())
    
    # Filter data for each type
    specialized_data = combined_df[combined_df['neuron_id'].isin(specialized_neurons)]
    polysemantic_data = combined_df[combined_df['neuron_id'].isin(polysemantic_neurons)]
    
    print(f"Specialized neurons: {len(specialized_neurons)} neurons, {len(specialized_data)} checkpoints")
    print(f"Polysemantic neurons: {len(polysemantic_neurons)} neurons, {len(polysemantic_data)} checkpoints")
    
    # Analyze trends for each metric
    metrics = ['num_clusters', 'mean_intra', 'mean_inter']
    
    for metric in metrics:
        print(f"\n📊 {metric.upper().replace('_', ' ')} EVOLUTION:")
        
        # Specialized neurons
        if not specialized_data.empty:
            spec_means = []
            for step in checkpoint_steps:
                step_data = specialized_data[specialized_data['checkpoint_step'] == step]
                if not step_data.empty:
                    spec_means.append(step_data[metric].mean())
                else:
                    spec_means.append(np.nan)
            
            valid_indices = [i for i, x in enumerate(spec_means) if not np.isnan(x)]
            if len(valid_indices) >= 2:
                x = np.array(checkpoint_steps)[valid_indices]
                y = np.array(spec_means)[valid_indices]
                slope = np.polyfit(x, y, 1)[0]
                print(f"  Specialized: {slope:.6f} per step")
                print(f"    Early ({checkpoint_steps[0]}): {spec_means[0]:.4f}")
                print(f"    Late ({checkpoint_steps[-1]}): {spec_means[-1]:.4f}")
                print(f"    Change: {spec_means[-1] - spec_means[0]:+.4f}")
        
        # Polysemantic neurons
        if not polysemantic_data.empty:
            poly_means = []
            for step in checkpoint_steps:
                step_data = polysemantic_data[polysemantic_data['checkpoint_step'] == step]
                if not step_data.empty:
                    poly_means.append(step_data[metric].mean())
                else:
                    poly_means.append(np.nan)
            
            valid_indices = [i for i, x in enumerate(poly_means) if not np.isnan(x)]
            if len(valid_indices) >= 2:
                x = np.array(checkpoint_steps)[valid_indices]
                y = np.array(poly_means)[valid_indices]
                slope = np.polyfit(x, y, 1)[0]
                print(f"  Polysemantic: {slope:.6f} per step")
                print(f"    Early ({checkpoint_steps[0]}): {poly_means[0]:.4f}")
                print(f"    Late ({checkpoint_steps[-1]}): {poly_means[-1]:.4f}")
                print(f"    Change: {poly_means[-1] - poly_means[0]:+.4f}")
    
    return specialized_data, polysemantic_data, checkpoint_steps

def create_comparison_plots(specialized_data, polysemantic_data, checkpoint_steps, model_name, output_dir="results", show_intra=True, show_inter=True):
    """Create comparison plots for specialized vs polysemantic neurons"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Determine which metrics to show
    metrics = ['num_clusters']
    if show_intra:
        metrics.append('mean_intra')
    if show_inter:
        metrics.append('mean_inter')
    
    # Create subplot layout based on number of metrics
    num_plots = len(metrics)
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    if num_plots == 1:
        axes = [axes]  # Make it iterable
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue for specialized, Orange for polysemantic
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot specialized neurons
        if not specialized_data.empty:
            spec_means = []
            spec_stds = []
            for step in checkpoint_steps:
                step_data = specialized_data[specialized_data['checkpoint_step'] == step]
                if not step_data.empty:
                    spec_means.append(step_data[metric].mean())
                    spec_stds.append(step_data[metric].std())
                else:
                    spec_means.append(np.nan)
                    spec_stds.append(np.nan)
            
            valid_indices = [i for i, x in enumerate(spec_means) if not np.isnan(x)]
            if len(valid_indices) >= 2:
                x = np.array(checkpoint_steps)[valid_indices]
                y = np.array(spec_means)[valid_indices]
                y_std = np.array(spec_stds)[valid_indices]
                
                ax.plot(x, y, 'o-', color=colors[0], linewidth=2, markersize=4, 
                       label=f'Specialized (n={len(specialized_data["neuron_id"].unique())})')
        
        # Plot polysemantic neurons
        if not polysemantic_data.empty:
            poly_means = []
            poly_stds = []
            for step in checkpoint_steps:
                step_data = polysemantic_data[polysemantic_data['checkpoint_step'] == step]
                if not step_data.empty:
                    poly_means.append(step_data[metric].mean())
                    poly_stds.append(step_data[metric].std())
                else:
                    poly_means.append(np.nan)
                    poly_stds.append(np.nan)
            
            valid_indices = [i for i, x in enumerate(poly_means) if not np.isnan(x)]
            if len(valid_indices) >= 2:
                x = np.array(checkpoint_steps)[valid_indices]
                y = np.array(poly_means)[valid_indices]
                y_std = np.array(poly_stds)[valid_indices]
                
                ax.plot(x, y, 's-', color=colors[1], linewidth=2, markersize=4, 
                       label=f'Polysemantic (n={len(polysemantic_data["neuron_id"].unique())})')
        
        ax.set_title('Pythia-160M: Clusters per Neuron (by type)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Average Clusters', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    # Save the plot
    output_file = Path(output_dir) / f"specialized_vs_polysemantic_{model_name.replace('-', '_')}.pdf"
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to {output_file}")
    plt.show()



def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Analyze specialized vs polysemantic neurons')
    parser.add_argument('--model', default='pythia-160m', 
                       choices=['pythia-70m', 'pythia-160m', 'pythia-410m'],
                       help='Model to analyze')
    parser.add_argument('--output-dir', default='results', help='Output directory for plots')
    parser.add_argument('--hide-intra', action='store_true', help='Hide mean_intra plot')
    parser.add_argument('--hide-inter', action='store_true', help='Hide mean_inter plot')
    
    args = parser.parse_args()
    
    print(f"🔍 SPECIALIZED VS POLYSEMANTIC ANALYSIS")
    print(f"=" * 50)
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    combined_df, neuron_data = load_model_data(args.model, args.output_dir)
    
    if combined_df is None:
        print("❌ No data found. Exiting.")
        return
    
    # Classify neurons
    specialized_neurons, polysemantic_neurons = classify_neurons(neuron_data)
    
    if not specialized_neurons and not polysemantic_neurons:
        print("❌ No neurons found in either category. Exiting.")
        return
    
    # Analyze evolution
    specialized_data, polysemantic_data, checkpoint_steps = analyze_evolution_by_type(
        combined_df, specialized_neurons, polysemantic_neurons, args.model
    )
    
    # Create comparison plots
    print(f"\n📈 Creating comparison plots...")
    create_comparison_plots(specialized_data, polysemantic_data, checkpoint_steps, args.model, args.output_dir, 
                           show_intra=not args.hide_intra, show_inter=not args.hide_inter)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Check the {args.output_dir}/ directory for generated plots")

if __name__ == "__main__":
    import argparse
    main()
