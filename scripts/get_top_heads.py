#!/usr/bin/env python3
"""
Get top-k heads from CoRe or BCE weight files.
Useful for quick inspection of head rankings.
"""

import json
import argparse
import numpy as np
from pathlib import Path


def get_core_heads(llm_name, temp=0.001, prune=0.0, top_k=20):
    """Get top-k heads from CoRe detection."""
    core_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'core_temp{temp}_prune{prune}.json'

    with open(core_file, 'r') as f:
        core_scores = json.load(f)

    heads = []
    for key, scores in core_scores.items():
        layer, head = map(int, key.split('-'))
        avg_score = np.mean(scores) if isinstance(scores, list) else scores
        heads.append({'layer': layer, 'head': head, 'score': avg_score})

    heads.sort(key=lambda x: x['score'], reverse=True)
    return heads[:top_k]


def get_bce_heads(llm_name, lambda_l1, num_samples, top_k=20):
    """Get top-k heads from BCE weights by absolute value."""
    bce_file = Path(__file__).parent.parent / 'head_data' / llm_name / f'bce_weights_lambda{lambda_l1}_n{num_samples}.json'

    with open(bce_file, 'r') as f:
        data = json.load(f)

    heads = []
    for key, weight in data['all_weights'].items():
        layer, head = map(int, key.split('-'))
        heads.append({'layer': layer, 'head': head, 'weight': weight, 'abs_weight': abs(weight)})

    heads.sort(key=lambda x: x['abs_weight'], reverse=True)
    return heads[:top_k]


def main():
    parser = argparse.ArgumentParser(description='Get top heads from CoRe or BCE')
    parser.add_argument('--llm', type=str, default='mistral',
                        choices=['mistral', 'llama', 'phi', 'granite'])
    parser.add_argument('--method', type=str, default='core',
                        choices=['core', 'bce'], help='Which method to use')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top heads to show')
    parser.add_argument('--temp', type=float, default=0.001, help='CoRe temperature')
    parser.add_argument('--lambda_l1', type=float, default=0.1, help='BCE lambda')
    parser.add_argument('--num_samples', type=int, default=1000, help='BCE num samples')
    args = parser.parse_args()

    print(f"Top {args.top_k} heads for {args.llm} ({args.method})")
    print("=" * 50)

    if args.method == 'core':
        heads = get_core_heads(args.llm, args.temp, top_k=args.top_k)
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Score':<12}")
        print("-" * 40)
        for i, h in enumerate(heads):
            print(f"{i+1:<6} {h['layer']:<8} {h['head']:<8} {h['score']:<12.6f}")

    else:
        heads = get_bce_heads(args.llm, args.lambda_l1, args.num_samples, args.top_k)
        print(f"{'Rank':<6} {'Layer':<8} {'Head':<8} {'Weight':<12} {'|Weight|':<12}")
        print("-" * 55)
        for i, h in enumerate(heads):
            print(f"{i+1:<6} {h['layer']:<8} {h['head']:<8} {h['weight']:<12.6f} {h['abs_weight']:<12.6f}")

    # Print as list for easy copy-paste
    print(f"\nAs list: {[(h['layer'], h['head']) for h in heads[:8]]}")


if __name__ == '__main__':
    main()
