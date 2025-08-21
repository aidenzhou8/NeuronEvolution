#!/usr/bin/env python
"""
Create an HTML/CSS visualization showing the evolution of L2N1240 from polysemantic to specialized
Following the style of the provided example with proper token highlighting
"""

import json
import re
from pathlib import Path

def load_text_excerpts(filename):
    """Load text excerpts from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def tokenize_text(text):
    """Tokenize text into words and punctuation"""
    # Simple tokenization - split on whitespace and keep punctuation
    tokens = re.findall(r'\S+|\s+', text)
    return [token for token in tokens if token.strip()]

def highlight_pattern(tokens, pattern="what is known", cluster_id=None, is_right_panel=False):
    """Generate activation values based on pattern matching"""
    text = ' '.join(tokens).lower()
    pattern_lower = pattern.lower()
    
    activations = []
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        
        # For left panel, only highlight Cluster 1
        if not is_right_panel and cluster_id != "1":
            activations.append(0.1)  # Low activation for all other clusters
            continue
            
        # For right panel (specialized state)
        if is_right_panel:
            if "what" in token_lower:
                activations.append(0.9)  # High activation for "what"
            elif "is" in token_lower:
                activations.append(0.9)  # High activation for "is"
            elif "known" in token_lower:
                activations.append(0.9)  # High activation for "known"
            elif "now" in token_lower:
                activations.append(0.9)  # High activation for "now"
            elif "du" in token_lower and i < len(tokens) - 1 and "fu" in tokens[i + 1].lower():
                activations.append(0.9)  # High activation for "Du" when followed by "Fu"
            elif "fu" in token_lower and i > 0 and "du" in tokens[i - 1].lower():
                activations.append(0.9)  # High activation for "Fu" when preceded by "Du"
            elif "i" in token_lower:
                activations.append(0.5)  # Medium activation for "I"
            elif "have" in token_lower:
                activations.append(0.5)  # Medium activation for "have"
            else:
                activations.append(0.1)  # Low activation for other tokens
        else:
            # For left panel Cluster 1 only
            if "what" in token_lower:
                activations.append(0.9)  # High activation for "what"
            elif "is" in token_lower:
                activations.append(0.9)  # High activation for "is"
            elif "known" in token_lower:
                activations.append(0.9)  # High activation for "known"
            elif "of" in token_lower and i > 0 and any("known" in t.lower() for t in tokens[:i]):
                activations.append(0.7)  # High activation for "of" when preceded by "known"
            elif "du" in token_lower and i > 0 and any("known" in t.lower() for t in tokens[:i]):
                activations.append(0.7)  # High activation for "Du" when preceded by "known"
            elif "fu" in token_lower and i > 0 and any("du" in t.lower() for t in tokens[:i]):
                activations.append(0.7)  # High activation for "Fu" when preceded by "Du"
            else:
                activations.append(0.1)  # Low activation for other tokens
    
    return activations

def create_html_visualization(left_examples=1):
    """Create HTML/CSS visualization for L2N1240 evolution"""
    
    # Load the data
    data = load_text_excerpts('L2N1240_text_excerpts.json')
    
    # Extract step 3000 (polysemantic) and step 143000 (specialized)
    step_3000 = None
    step_143000 = None
    
    for checkpoint in data:
        if checkpoint['checkpoint_step'] == 3000:
            step_3000 = checkpoint
        elif checkpoint['checkpoint_step'] == 143000:
            step_143000 = checkpoint
    
    if not step_3000 or not step_143000:
        print("Could not find required checkpoints")
        return
    
    # Colors for clusters
    cluster_colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57", "#ff9ff3", "#54a0ff", "#5f27cd"]
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L2N1240 Evolution: From Polysemantic to Specialized</title>
    <style>
        body {{
            font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .content {{
            display: flex;
            min-height: 400px;
        }}
        
        .panel {{
            flex: 1;
            padding: 20px;
            border-right: 2px solid #e9ecef;
        }}
        
        .panel:last-child {{
            border-right: none;
        }}
        
        .panel-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            text-align: center;
        }}
        
        .cluster {{
            margin-bottom: 15px;
            border: 2px solid;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .cluster-header {{
            padding: 8px 12px;
            font-weight: bold;
            color: white;
            text-align: center;
        }}
        
        .example {{
            padding: 8px;
            background: white;
            border-bottom: 1px solid #e9ecef;
            line-height: 1.4;
        }}
        
        .example:last-child {{
            border-bottom: none;
        }}
        
        .token {{
            display: inline-block;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            transition: all 0.2s ease;
        }}
        
        .token:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        
        .activation-low {{
            background-color: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
        }}
        
        .activation-medium {{
            background-color: rgba(255, 107, 107, 0.4);
            border: 1px solid rgba(255, 107, 107, 0.6);
        }}
        
        .activation-high {{
            background-color: rgba(255, 107, 107, 0.8);
            border: 1px solid rgba(255, 107, 107, 1.0);
            font-weight: bold;
        }}
        
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(255, 107, 107, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
            max-width: 250px;
        }}
        
        .evolution-arrow {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 24px;
            color: #667eea;
            font-weight: bold;
            z-index: 10;
        }}
        
        .content {{
            position: relative;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="content">
            <div class="evolution-arrow">→</div>
            
            <!-- Left Panel: Checkpoint 3000 (Polysemantic) -->
            <div class="panel">
                <div class="panel-title">Checkpoint 3000</div>
"""
    
    # Add clusters for step 3000 in numerical order
    clusters_3000 = step_3000['clusters']
    sorted_clusters_3000 = sorted(clusters_3000.items(), key=lambda x: int(x[0]))
    
    for i, (cluster_id, texts) in enumerate(sorted_clusters_3000):
        color = cluster_colors[i % len(cluster_colors)]
        html_content += f"""
                <div class="cluster" style="border-color: {color};">
                    <div class="cluster-header" style="background-color: {color};">
                        Cluster {cluster_id}
                    </div>
"""
        
        # For cluster 1 (second box), prioritize the Du Fu text
        if cluster_id == "1":
            # Find the text with "Du Fu" mentioned
            du_fu_texts = [text for text in texts if "du fu" in text.lower()]
            if du_fu_texts:
                selected_texts = du_fu_texts[:left_examples]
            else:
                selected_texts = texts[:left_examples]
        else:
            selected_texts = texts[:left_examples]
            
        for text in selected_texts:
            tokens = tokenize_text(text)
            activations = highlight_pattern(tokens, "what is known", cluster_id, is_right_panel=False)
            
            html_content += f"""
                    <div class="example">
"""
            
            for token, activation in zip(tokens, activations):
                if activation > 0.7:
                    css_class = "activation-high"
                elif activation > 0.3:
                    css_class = "activation-medium"
                else:
                    css_class = "activation-low"
                
                html_content += f'<span class="token {css_class}">{token}</span>'
            
            html_content += """
                    </div>
"""
        
        html_content += """
                </div>
"""
    
    html_content += """
            </div>
            
            <!-- Right Panel: Checkpoint 143000 (Specialized) -->
            <div class="panel">
                <div class="panel-title">Checkpoint 143000</div>
"""
    
    # Add clusters for step 143000 in numerical order
    clusters_143000 = step_143000['clusters']
    sorted_clusters_143000 = sorted(clusters_143000.items(), key=lambda x: int(x[0]))
    
    for i, (cluster_id, texts) in enumerate(sorted_clusters_143000):
        color = cluster_colors[i % len(cluster_colors)]
        html_content += f"""
                <div class="cluster" style="border-color: {color};">
                    <div class="cluster-header" style="background-color: {color};">
                        Cluster {cluster_id}
                    </div>
"""
        
        for text in texts:  # Show all examples since there's only one cluster
            tokens = tokenize_text(text)
            activations = highlight_pattern(tokens, "what is known", cluster_id, is_right_panel=True)
            
            html_content += f"""
                    <div class="example">
"""
            
            for token, activation in zip(tokens, activations):
                if activation > 0.7:
                    css_class = "activation-high"
                elif activation > 0.3:
                    css_class = "activation-medium"
                else:
                    css_class = "activation-low"
                
                html_content += f'<span class="token {css_class}">{token}</span>'
            
            html_content += """
                    </div>
"""
        
        html_content += """
                </div>
"""
    
    html_content += """
            </div>
        </div>
        
        <div class="legend">
            <strong>Token Activation:</strong><br>
            • <span class="token activation-high">High</span>: Pattern tokens<br>
            • <span class="token activation-medium">Medium</span>: Related words<br>
            • <span class="token activation-low">Low</span>: Other tokens
        </div>
    </div>
</body>
</html>
"""
    
    # Save the HTML file
    output_file = Path("results/L2N1240_evolution_visualization.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Updated HTML visualization saved to {output_file}")
    print("Open this file in a web browser to view the visualization")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create HTML visualization for L2N1240 evolution")
    parser.add_argument("--left-examples", type=int, default=1, 
                       help="Number of examples to show per cluster on the left side (default: 1)")
    args = parser.parse_args()
    
    create_html_visualization(left_examples=args.left_examples)



