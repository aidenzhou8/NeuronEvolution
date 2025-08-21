#!/usr/bin/env python
"""
A simple demo script that shows how to use the neuron embeddings toolkit.

This script:
1. Loads a TransformerLens model (default: GPT-2-small)
2. Streams WikiText-2 train split through the model
3. Collects up to N high-activation examples for a target neuron
4. Clusters the embeddings with HAC + cosine distance
5. Prints simple polysemanticity metrics

It's basically a complete example of how to analyze what a neuron is doing!
"""

import argparse
import os
from pathlib import Path
import re
from collections import Counter
import json, csv, time

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch # type: ignore
from datasets import load_dataset # type: ignore
from transformer_lens import HookedTransformer # type: ignore
from transformer_lens.loading_from_pretrained import get_checkpoint_labels  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# ---- Toolkit imports (from the ne_tlk module) --------------------------------
from ne_tlk import (
    TransformerLensEmbeddingCollector,
    cluster_embeddings,
    polysemanticity_metrics,
)

def analyze_cluster_keywords(texts):
    """Extract the most frequent non-trivial words from a list of texts.
    
    This helps us understand what each cluster is about by looking at the most
    common meaningful words in the text examples.
    """
    # Common stop words and trivial terms to ignore
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        'mine', 'yours', 'hers', 'ours', 'theirs', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'not', 'no', 'nor', 'neither', 'either', 'so', 'as', 'than', 'too', 'very',
        'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
        'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
        'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
    }
    
    # Combine all texts and extract words
    all_text = ' '.join(texts).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', all_text)
    
    # Filter out stop words and short words
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequencies
    word_counts = Counter(meaningful_words)
    
    # Return top 3 most frequent words
    return [word for word, count in word_counts.most_common(3)]

# ------------------------------------------------------------------------------
def wikitext_loader(tokenizer, batch_size=16, split="test", max_examples=30000):
    """Yields dicts that TransformerLens models accept (`input_ids`, `attention_mask`).
    
    This is a simple data loader that streams WikiText-2 data in batches.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    batch_texts = []
    for item in ds:
        if len(batch_texts) >= batch_size:
            toks = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            yield toks
            batch_texts = []
        batch_texts.append(item["text"])
        if len(batch_texts) * len(batch_texts[0]) > max_examples:  # Rough character count limit
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-70m", help="HF/TransformerLens model alias")
    parser.add_argument("--layer", required=True, help="Layer name containing the neuron (e.g. blocks.6.mlp)")
    parser.add_argument("--neuron", type=int, required=True, help="Row index of neuron in weight matrix")
    parser.add_argument("--max_examples", type=int, default=100, help="How many high‑act examples to keep")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for inclusion (fraction of peak activation)")
    parser.add_argument("--peak_activation", type=float, default = 2.5, help="Peak activation for this neuron (from Neuroscope)")
    parser.add_argument("--distance_threshold", type=float, default=0.8, help="Distance threshold for clustering (cosine distance)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_text", action="store_true", help="Skip text decoding for maximum speed")
    parser.add_argument("--outfile", type=Path, help="Output file path (optional, will auto-generate if not provided)")
    args = parser.parse_args()

    # Generate descriptive filename if not provided
    if args.outfile is None:
        # Extract layer number from layer name (e.g., "blocks.6.mlp" -> "6")
        layer_num = args.layer.split('.')[1] if '.' in args.layer else "0"
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        args.outfile = results_dir / f"L{layer_num}N{args.neuron}_metrics.json"
    else:
        # If outfile is provided, ensure it's in the results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        if not args.outfile.is_absolute():
            args.outfile = results_dir / args.outfile.name

    # 1. Load model
    model = HookedTransformer.from_pretrained(args.model)
    tokenizer = model.tokenizer
    
    # Set device for acceleration
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA acceleration")
    else:
        device = "cpu"
        print(f"Using CPU (no acceleration available)")

    # 2. Set up collector
    collector = TransformerLensEmbeddingCollector(
        model,
        layer_name=args.layer,
        neuron_idx=args.neuron,
        activation_threshold=args.threshold,
        peak_activation=args.peak_activation,
        max_examples=args.max_examples,
        device=device,
        decode_text=not args.no_text,
    )

    # 3. Stream data until collector is full
    embeds = collector.run(
        wikitext_loader(tokenizer, batch_size=args.batch_size)
    )  # shape (N, d_hidden)

    # 4. Cluster & metric summary
    labels = cluster_embeddings(embeds, distance_threshold=args.distance_threshold)
    metrics = polysemanticity_metrics(embeds, labels)

    # 5. Report
    import json

    # Create output with metrics, labels, and text examples
    output_data = {
        "metrics": metrics, 
        "labels": labels.tolist(),
        "text_examples": collector._text_cache if hasattr(collector, '_text_cache') else []
    }
    
    args.outfile.write_text(json.dumps(output_data, indent=2))
    
    # Print metrics
    print("==== Polysemanticity metrics ====")
    for k, v in metrics.items():
        if k == 'cluster_sizes':
            print(f"{k:>12}: {dict(v)}")
        else:
            print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
    
    # Print cluster analysis
    print(f"\n==== Cluster Analysis ====")
    text_examples = collector._text_cache if hasattr(collector, '_text_cache') else []
    
    # Group text examples by cluster
    from collections import defaultdict
    cluster_texts = defaultdict(list)
    for i, (label, text) in enumerate(zip(labels, text_examples)):
        cluster_texts[label].append(text)
    
    # Show text examples and keywords for each cluster
    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        keywords = analyze_cluster_keywords(texts)
        print(f"\nCluster {cluster_id} ({len(texts)} examples) - Keywords: {', '.join(keywords)}")
        for i, text in enumerate(texts[:5]):  # Show first 5 examples
            print(f"  {i+1}. {text}")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more examples")
    
    print(f"\nSaved detailed output to {args.outfile.absolute()}")


if __name__ == "__main__":
    # CUDA makes this faster but is optional
    torch.set_grad_enabled(False)
    main()
