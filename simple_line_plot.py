#!/usr/bin/env python
"""
Simple line plot of cluster evolution over time for L2N1240
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_simple_line_plot():
    """Create a simple line plot of cluster evolution"""
    
    # Load the data
    csv_file = Path("results/L2N1240_pythia70m_ckpt_summary.csv")
    df = pd.read_csv(csv_file)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the line
    plt.plot(df['checkpoint_step'], df['num_clusters'], 'o-', 
             color='#2E86AB', linewidth=2, markersize=4, alpha=0.8)
    
    # Add trend line
    z = np.polyfit(df['checkpoint_step'], df['num_clusters'], 1)
    p = np.poly1d(z)
    plt.plot(df['checkpoint_step'], p(df['checkpoint_step']), 
             '--', color='#A23B72', linewidth=2, alpha=0.8,
             label=f'Trend (slope: {z[0]:.6f})')
    
    # Customize the plot
    plt.title('L2N1240: Cluster Evolution Over Training', fontsize=14, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add some statistics as text
    stats_text = f"Mean: {df['num_clusters'].mean():.1f}\n"
    stats_text += f"Min: {df['num_clusters'].min()} (step {df.loc[df['num_clusters'].idxmin(), 'checkpoint_step']})\n"
    stats_text += f"Max: {df['num_clusters'].max()} (step {df.loc[df['num_clusters'].idxmax(), 'checkpoint_step']})"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("results/L2N1240_simple_line_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Simple line plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    create_simple_line_plot()

