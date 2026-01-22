#!/usr/bin/env python3
"""
Analyze correlations between attention heads relative to relevance information.

This script computes:
1. Head-to-relevance correlation (point-biserial)
2. Inter-head correlation matrix (Pearson/Spearman)
3. Conditional correlation (separately for positive/negative docs)
4. Partial correlation (controlling for relevance)
5. Head clustering based on correlation patterns

Usage:
    python scripts/analyze_head_correlations.py --llm mistral --num_samples 1000
    python scripts/analyze_head_correlations.py -f features.npz --plot --output results.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pointbiserialr, spearmanr, pearsonr
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

from utils import log_command, load_features, get_head_info


def head_to_relevance_correlation(features, labels, method='pointbiserial'):
    """
    Compute correlation between each head's attention scores and relevance labels.

    Args:
        features: (n_docs, n_heads) attention features
        labels: (n_docs,) binary relevance labels
        method: 'pointbiserial', 'spearman', or 'pearson'

    Returns:
        correlations: (n_heads,) correlation coefficients
        pvalues: (n_heads,) p-values
    """
    n_heads = features.shape[1]
    correlations = np.zeros(n_heads)
    pvalues = np.zeros(n_heads)

    for h in range(n_heads):
        head_scores = features[:, h]

        if method == 'pointbiserial':
            # Point-biserial correlation for binary labels
            corr, pval = pointbiserialr(labels, head_scores)
        elif method == 'spearman':
            corr, pval = spearmanr(labels, head_scores)
        elif method == 'pearson':
            corr, pval = pearsonr(labels, head_scores)
        else:
            raise ValueError(f"Unknown method: {method}")

        correlations[h] = corr
        pvalues[h] = pval

    return correlations, pvalues


def inter_head_correlation(features, method='pearson'):
    """
    Compute correlation matrix between all pairs of heads.

    Args:
        features: (n_docs, n_heads) attention features
        method: 'pearson' or 'spearman'

    Returns:
        corr_matrix: (n_heads, n_heads) correlation matrix
    """
    if method == 'pearson':
        corr_matrix = np.corrcoef(features.T)
    elif method == 'spearman':
        n_heads = features.shape[1]
        corr_matrix = np.zeros((n_heads, n_heads))
        for i in range(n_heads):
            for j in range(i, n_heads):
                corr, _ = spearmanr(features[:, i], features[:, j])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr_matrix


def conditional_correlation(features, labels, method='pearson'):
    """
    Compute inter-head correlation separately for positive and negative documents.

    Args:
        features: (n_docs, n_heads) attention features
        labels: (n_docs,) binary relevance labels
        method: 'pearson' or 'spearman'

    Returns:
        corr_pos: correlation matrix for positive docs
        corr_neg: correlation matrix for negative docs
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    features_pos = features[pos_mask]
    features_neg = features[neg_mask]

    corr_pos = inter_head_correlation(features_pos, method)
    corr_neg = inter_head_correlation(features_neg, method)

    return corr_pos, corr_neg


def partial_correlation(features, labels):
    """
    Compute partial correlation between heads, controlling for relevance.

    This shows head relationships beyond their shared relevance signal.

    Args:
        features: (n_docs, n_heads) attention features
        labels: (n_docs,) binary relevance labels

    Returns:
        partial_corr: (n_heads, n_heads) partial correlation matrix
    """
    n_heads = features.shape[1]

    # Residualize each head's scores with respect to relevance
    residuals = np.zeros_like(features)
    for h in range(n_heads):
        # Simple linear regression: head_score = a + b * label + residual
        X = np.column_stack([np.ones(len(labels)), labels])
        coeffs = np.linalg.lstsq(X, features[:, h], rcond=None)[0]
        predicted = X @ coeffs
        residuals[:, h] = features[:, h] - predicted

    # Compute correlation of residuals
    partial_corr = np.corrcoef(residuals.T)

    return partial_corr


def cluster_heads(corr_matrix, n_clusters=None, threshold=0.5):
    """
    Cluster heads based on their correlation patterns.

    Args:
        corr_matrix: (n_heads, n_heads) correlation matrix
        n_clusters: number of clusters (if None, use threshold)
        threshold: distance threshold for clustering

    Returns:
        cluster_labels: (n_heads,) cluster assignment for each head
        linkage_matrix: hierarchical clustering linkage matrix
    """
    # Convert correlation to distance (1 - |corr|)
    # Using absolute correlation since negative correlation also indicates relationship
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0)

    # Ensure symmetry and valid distance matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = np.clip(distance_matrix, 0, 1)

    # Convert to condensed form for linkage
    condensed = squareform(distance_matrix, checks=False)

    # Hierarchical clustering
    linkage_matrix = linkage(condensed, method='average')

    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    else:
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')

    return cluster_labels, linkage_matrix


def find_diverse_heads(relevance_corr, inter_corr, n_heads=8, relevance_weight=0.5):
    """
    Select a diverse subset of heads that are predictive but not redundant.

    Uses a greedy algorithm that balances:
    - High absolute correlation with relevance
    - Low correlation with already-selected heads

    Args:
        relevance_corr: (n_heads,) correlation with relevance
        inter_corr: (n_heads, n_heads) inter-head correlation matrix
        n_heads: number of heads to select
        relevance_weight: weight for relevance vs diversity (0-1)

    Returns:
        selected: list of selected head indices
        scores: selection scores for each head
    """
    n_total = len(relevance_corr)
    abs_relevance = np.abs(relevance_corr)

    # Normalize relevance correlation to [0, 1]
    rel_norm = (abs_relevance - abs_relevance.min()) / (abs_relevance.max() - abs_relevance.min() + 1e-8)

    selected = []
    scores = []

    for _ in range(min(n_heads, n_total)):
        best_score = -np.inf
        best_head = -1

        for h in range(n_total):
            if h in selected:
                continue

            # Relevance score
            rel_score = rel_norm[h]

            # Diversity score (low correlation with selected heads)
            if len(selected) > 0:
                max_corr_with_selected = np.max(np.abs(inter_corr[h, selected]))
                div_score = 1 - max_corr_with_selected
            else:
                div_score = 1.0

            # Combined score
            combined = relevance_weight * rel_score + (1 - relevance_weight) * div_score

            if combined > best_score:
                best_score = combined
                best_head = h

        if best_head >= 0:
            selected.append(best_head)
            scores.append(best_score)

    return selected, scores


def plot_correlation_matrix(corr_matrix, title, output_path, head_labels=None,
                            relevance_corr=None, num_layers=32, num_heads_per_layer=32):
    """
    Plot correlation matrix as a heatmap.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_heads = corr_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontsize=12)

    # Add layer boundaries
    for layer in range(1, num_layers):
        boundary = layer * num_heads_per_layer
        if boundary < n_heads:
            ax.axhline(y=boundary - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(x=boundary - 0.5, color='black', linewidth=0.5, alpha=0.3)

    # Labels
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Head Index', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add layer ticks
    layer_ticks = [i * num_heads_per_layer + num_heads_per_layer // 2
                   for i in range(num_layers) if i * num_heads_per_layer < n_heads]
    layer_labels = [f'L{i}' for i in range(len(layer_ticks))]

    if len(layer_ticks) <= 16:  # Only show if not too crowded
        ax.set_xticks(layer_ticks)
        ax.set_xticklabels(layer_labels, fontsize=8)
        ax.set_yticks(layer_ticks)
        ax.set_yticklabels(layer_labels, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_relevance_correlation(relevance_corr, pvalues, output_path,
                                num_layers=32, num_heads_per_layer=32, top_k=20):
    """
    Plot head-to-relevance correlations.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    n_heads = len(relevance_corr)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: All heads bar chart
    ax1 = axes[0]
    colors = ['green' if c > 0 else 'red' for c in relevance_corr]
    ax1.bar(range(n_heads), relevance_corr, color=colors, alpha=0.7, width=1.0)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Head Index', fontsize=12)
    ax1.set_ylabel('Correlation with Relevance', fontsize=12)
    ax1.set_title('Head-to-Relevance Correlation (Point-Biserial)', fontsize=14)

    # Add layer boundaries
    for layer in range(1, num_layers):
        boundary = layer * num_heads_per_layer
        if boundary < n_heads:
            ax1.axvline(x=boundary - 0.5, color='gray', linewidth=0.5, alpha=0.5, linestyle='--')

    # Plot 2: Top-k heads
    ax2 = axes[1]
    sorted_idx = np.argsort(np.abs(relevance_corr))[::-1][:top_k]
    top_corrs = relevance_corr[sorted_idx]
    top_labels = [f'L{idx // num_heads_per_layer}H{idx % num_heads_per_layer}'
                  for idx in sorted_idx]
    colors = ['green' if c > 0 else 'red' for c in top_corrs]

    bars = ax2.barh(range(top_k), top_corrs, color=colors, alpha=0.7)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(top_labels)
    ax2.set_xlabel('Correlation with Relevance', fontsize=12)
    ax2.set_title(f'Top {top_k} Heads by Absolute Correlation', fontsize=14)
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.invert_yaxis()

    # Add correlation values
    for i, (bar, corr) in enumerate(zip(bars, top_corrs)):
        ax2.text(corr + 0.01 if corr > 0 else corr - 0.01, i,
                f'{corr:.3f}', va='center', ha='left' if corr > 0 else 'right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_dendrogram(linkage_matrix, output_path, num_layers=32, num_heads_per_layer=32,
                    n_heads=None, relevance_corr=None):
    """
    Plot hierarchical clustering dendrogram.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # Create labels
    if n_heads is not None:
        labels = [f'{i // num_heads_per_layer}-{i % num_heads_per_layer}'
                  for i in range(n_heads)]
    else:
        labels = None

    # Plot dendrogram
    dend = dendrogram(linkage_matrix, ax=ax, labels=labels, leaf_rotation=90,
                      leaf_font_size=6 if n_heads and n_heads > 100 else 8)

    ax.set_xlabel('Head (Layer-Head)', fontsize=12)
    ax.set_ylabel('Distance (1 - |correlation|)', fontsize=12)
    ax.set_title('Hierarchical Clustering of Attention Heads', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_conditional_diff(corr_pos, corr_neg, output_path, num_layers=32, num_heads_per_layer=32):
    """
    Plot difference between positive and negative document correlations.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    diff = corr_pos - corr_neg
    n_heads = diff.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Positive docs correlation
    im0 = axes[0].imshow(corr_pos, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('Correlation (Positive Docs)', fontsize=12)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Negative docs correlation
    im1 = axes[1].imshow(corr_neg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('Correlation (Negative Docs)', fontsize=12)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Difference
    max_diff = np.max(np.abs(diff))
    im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-max_diff, vmax=max_diff, aspect='auto')
    axes[2].set_title('Difference (Pos - Neg)', fontsize=12)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xlabel('Head Index', fontsize=10)
        ax.set_ylabel('Head Index', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    log_command()

    parser = argparse.ArgumentParser(description='Analyze head correlations relative to relevance')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--feature_file', '-f', type=str, default=None,
                        help='Path to feature file (.npz)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples (for default path)')
    parser.add_argument('--method', type=str, default='pearson',
                        choices=['pearson', 'spearman'],
                        help='Correlation method for inter-head analysis')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (default: auto)')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top heads to display')
    parser.add_argument('--diverse_k', type=int, default=8,
                        help='Number of diverse heads to select')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    # Get model config
    num_layers, num_heads_per_layer = get_head_info(args.llm)
    n_total_heads = num_layers * num_heads_per_layer

    print(f"Analyzing head correlations for {args.llm}")
    print(f"Model: {num_layers} layers × {num_heads_per_layer} heads = {n_total_heads} total heads")
    print("=" * 70)

    # Load features
    print("\nLoading features...")
    try:
        features, labels, docs_per_query = load_features(
            feature_file=args.feature_file,
            llm_name=args.llm,
            num_samples=args.num_samples
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Features shape: {features.shape}")
    print(f"Labels: {int(labels.sum())} positive, {int((labels == 0).sum())} negative")

    # 1. Head-to-relevance correlation
    print("\n" + "=" * 70)
    print("1. HEAD-TO-RELEVANCE CORRELATION")
    print("=" * 70)

    relevance_corr, relevance_pval = head_to_relevance_correlation(features, labels)

    # Sort by absolute correlation
    sorted_idx = np.argsort(np.abs(relevance_corr))[::-1]

    print(f"\nTop {args.top_k} heads by absolute correlation with relevance:")
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Corr':<12} {'P-value':<12} {'Direction':<10}")
    print("-" * 60)

    for rank, idx in enumerate(sorted_idx[:args.top_k]):
        layer = idx // num_heads_per_layer
        head = idx % num_heads_per_layer
        corr = relevance_corr[idx]
        pval = relevance_pval[idx]
        direction = "positive" if corr > 0 else "negative"
        print(f"{rank+1:<6} {layer:<8} {head:<8} {corr:<12.4f} {pval:<12.2e} {direction:<10}")

    # Statistics
    n_positive_corr = np.sum(relevance_corr > 0)
    n_negative_corr = np.sum(relevance_corr < 0)
    n_significant = np.sum(relevance_pval < 0.05)

    print(f"\nSummary:")
    print(f"  Positive correlations: {n_positive_corr} heads")
    print(f"  Negative correlations: {n_negative_corr} heads")
    print(f"  Significant (p<0.05): {n_significant} heads")
    print(f"  Mean |correlation|: {np.mean(np.abs(relevance_corr)):.4f}")
    print(f"  Max correlation: {np.max(relevance_corr):.4f}")
    print(f"  Min correlation: {np.min(relevance_corr):.4f}")

    # 2. Inter-head correlation
    print("\n" + "=" * 70)
    print(f"2. INTER-HEAD CORRELATION ({args.method.upper()})")
    print("=" * 70)

    inter_corr = inter_head_correlation(features, method=args.method)

    # Get upper triangle (excluding diagonal)
    upper_tri = inter_corr[np.triu_indices_from(inter_corr, k=1)]

    print(f"\nInter-head correlation statistics:")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Std: {np.std(upper_tri):.4f}")
    print(f"  Min: {np.min(upper_tri):.4f}")
    print(f"  Max: {np.max(upper_tri):.4f}")
    print(f"  Highly correlated pairs (|r| > 0.8): {np.sum(np.abs(upper_tri) > 0.8)}")
    print(f"  Moderately correlated (|r| > 0.5): {np.sum(np.abs(upper_tri) > 0.5)}")

    # Find most correlated pairs
    print(f"\nTop 10 most correlated head pairs:")
    flat_idx = np.argsort(np.abs(upper_tri))[::-1][:10]
    row_idx, col_idx = np.triu_indices_from(inter_corr, k=1)

    print(f"{'Head 1':<12} {'Head 2':<12} {'Correlation':<12}")
    print("-" * 40)
    for idx in flat_idx:
        h1, h2 = row_idx[idx], col_idx[idx]
        l1, hd1 = h1 // num_heads_per_layer, h1 % num_heads_per_layer
        l2, hd2 = h2 // num_heads_per_layer, h2 % num_heads_per_layer
        print(f"L{l1}H{hd1:<8} L{l2}H{hd2:<8} {inter_corr[h1, h2]:<12.4f}")

    # 3. Conditional correlation
    print("\n" + "=" * 70)
    print("3. CONDITIONAL CORRELATION (POS vs NEG DOCS)")
    print("=" * 70)

    corr_pos, corr_neg = conditional_correlation(features, labels, method=args.method)

    upper_pos = corr_pos[np.triu_indices_from(corr_pos, k=1)]
    upper_neg = corr_neg[np.triu_indices_from(corr_neg, k=1)]

    print(f"\nPositive documents:")
    print(f"  Mean correlation: {np.mean(upper_pos):.4f}")
    print(f"  Std: {np.std(upper_pos):.4f}")

    print(f"\nNegative documents:")
    print(f"  Mean correlation: {np.mean(upper_neg):.4f}")
    print(f"  Std: {np.std(upper_neg):.4f}")

    diff = corr_pos - corr_neg
    upper_diff = diff[np.triu_indices_from(diff, k=1)]
    print(f"\nDifference (Pos - Neg):")
    print(f"  Mean: {np.mean(upper_diff):.4f}")
    print(f"  Pairs with large diff (|d| > 0.2): {np.sum(np.abs(upper_diff) > 0.2)}")

    # 4. Partial correlation
    print("\n" + "=" * 70)
    print("4. PARTIAL CORRELATION (CONTROLLING FOR RELEVANCE)")
    print("=" * 70)

    partial_corr = partial_correlation(features, labels)
    upper_partial = partial_corr[np.triu_indices_from(partial_corr, k=1)]

    print(f"\nPartial correlation statistics:")
    print(f"  Mean: {np.mean(upper_partial):.4f}")
    print(f"  Std: {np.std(upper_partial):.4f}")
    print(f"  Correlation reduction: {np.mean(upper_tri) - np.mean(upper_partial):.4f}")
    print(f"  (Positive = relevance explains some inter-head correlation)")

    # 5. Clustering
    print("\n" + "=" * 70)
    print("5. HEAD CLUSTERING")
    print("=" * 70)

    cluster_labels, linkage_matrix = cluster_heads(inter_corr, n_clusters=args.n_clusters)
    n_clusters = len(np.unique(cluster_labels))

    print(f"\nNumber of clusters: {n_clusters}")
    print(f"\nCluster sizes:")
    for c in range(1, n_clusters + 1):
        size = np.sum(cluster_labels == c)
        members = np.where(cluster_labels == c)[0]
        # Get mean relevance correlation for cluster
        mean_rel_corr = np.mean(relevance_corr[members])
        print(f"  Cluster {c}: {size} heads, mean relevance corr: {mean_rel_corr:.4f}")

    # 6. Diverse head selection
    print("\n" + "=" * 70)
    print(f"6. DIVERSE HEAD SELECTION (top {args.diverse_k})")
    print("=" * 70)

    diverse_heads, diverse_scores = find_diverse_heads(
        relevance_corr, inter_corr, n_heads=args.diverse_k
    )

    print(f"\nSelected heads (balancing relevance and diversity):")
    print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Rel.Corr':<12} {'Score':<10}")
    print("-" * 50)

    for rank, (idx, score) in enumerate(zip(diverse_heads, diverse_scores)):
        layer = idx // num_heads_per_layer
        head = idx % num_heads_per_layer
        corr = relevance_corr[idx]
        print(f"{rank+1:<6} {layer:<8} {head:<8} {corr:<12.4f} {score:<10.4f}")

    # Compare with top-k by relevance only
    top_by_relevance = sorted_idx[:args.diverse_k]
    overlap = len(set(diverse_heads) & set(top_by_relevance))
    print(f"\nOverlap with top-{args.diverse_k} by relevance: {overlap}/{args.diverse_k}")

    # Mean inter-correlation among selected sets
    diverse_inter = inter_corr[np.ix_(diverse_heads, diverse_heads)]
    topk_inter = inter_corr[np.ix_(top_by_relevance, top_by_relevance)]

    diverse_upper = diverse_inter[np.triu_indices_from(diverse_inter, k=1)]
    topk_upper = topk_inter[np.triu_indices_from(topk_inter, k=1)]

    print(f"\nMean inter-correlation:")
    print(f"  Diverse selection: {np.mean(np.abs(diverse_upper)):.4f}")
    print(f"  Top-k by relevance: {np.mean(np.abs(topk_upper)):.4f}")

    # Generate plots
    if args.plot:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)

        plot_dir = Path(__file__).parent.parent / 'head_data' / args.llm / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Plot correlation matrix
        plot_correlation_matrix(
            inter_corr,
            f'Inter-Head Correlation ({args.llm}, {args.method})',
            plot_dir / f'inter_head_corr_{args.method}.png',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer
        )

        # Plot relevance correlation
        plot_relevance_correlation(
            relevance_corr, relevance_pval,
            plot_dir / 'head_relevance_corr.png',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            top_k=args.top_k
        )

        # Plot dendrogram
        plot_dendrogram(
            linkage_matrix,
            plot_dir / 'head_clustering_dendrogram.png',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer,
            n_heads=features.shape[1]
        )

        # Plot conditional difference
        plot_conditional_diff(
            corr_pos, corr_neg,
            plot_dir / 'conditional_corr_diff.png',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer
        )

        # Plot partial correlation
        plot_correlation_matrix(
            partial_corr,
            f'Partial Correlation (controlling for relevance)',
            plot_dir / 'partial_corr.png',
            num_layers=num_layers,
            num_heads_per_layer=num_heads_per_layer
        )

    # Save results
    if args.output:
        results = {
            'llm': args.llm,
            'num_samples': args.num_samples,
            'n_docs': int(features.shape[0]),
            'n_heads': int(features.shape[1]),
            'method': args.method,
            'relevance_correlation': {
                'correlations': {f"{i // num_heads_per_layer}-{i % num_heads_per_layer}": float(relevance_corr[i])
                                 for i in range(len(relevance_corr))},
                'top_k': [{'layer': int(idx // num_heads_per_layer),
                          'head': int(idx % num_heads_per_layer),
                          'correlation': float(relevance_corr[idx]),
                          'pvalue': float(relevance_pval[idx])}
                         for idx in sorted_idx[:args.top_k]],
                'stats': {
                    'n_positive': int(n_positive_corr),
                    'n_negative': int(n_negative_corr),
                    'n_significant': int(n_significant),
                    'mean_abs': float(np.mean(np.abs(relevance_corr))),
                    'max': float(np.max(relevance_corr)),
                    'min': float(np.min(relevance_corr))
                }
            },
            'inter_head_correlation': {
                'mean': float(np.mean(upper_tri)),
                'std': float(np.std(upper_tri)),
                'min': float(np.min(upper_tri)),
                'max': float(np.max(upper_tri))
            },
            'conditional_correlation': {
                'positive_mean': float(np.mean(upper_pos)),
                'negative_mean': float(np.mean(upper_neg)),
                'diff_mean': float(np.mean(upper_diff))
            },
            'partial_correlation': {
                'mean': float(np.mean(upper_partial)),
                'correlation_reduction': float(np.mean(upper_tri) - np.mean(upper_partial))
            },
            'clustering': {
                'n_clusters': int(n_clusters),
                'cluster_sizes': {str(c): int(np.sum(cluster_labels == c)) for c in range(1, n_clusters + 1)}
            },
            'diverse_selection': {
                'heads': [{'layer': int(idx // num_heads_per_layer),
                          'head': int(idx % num_heads_per_layer),
                          'relevance_corr': float(relevance_corr[idx]),
                          'score': float(score)}
                         for idx, score in zip(diverse_heads, diverse_scores)],
                'mean_inter_corr': float(np.mean(np.abs(diverse_upper)))
            }
        }

        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
