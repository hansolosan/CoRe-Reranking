#!/usr/bin/env python3
"""Analyze the head detection data files."""

import json
import sys
from pathlib import Path

def analyze_file(filepath):
    """Analyze a head detection JSON file."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print('='*60)

    with open(filepath, 'r') as f:
        data = json.load(f)

    print(f"Total questions: {len(data)}")
    print(f"Paragraphs per question: {len(data[0]['paragraphs'])}")

    # Count positives and negatives per question
    pos_counts = []
    neg_counts = []
    for q in data:
        pos = sum(1 for p in q['paragraphs'] if p.get('is_positive', False))
        neg = sum(1 for p in q['paragraphs'] if p.get('is_negative', False))
        pos_counts.append(pos)
        neg_counts.append(neg)

    print(f"\nPositives per question: min={min(pos_counts)}, max={max(pos_counts)}, avg={sum(pos_counts)/len(pos_counts):.1f}")
    print(f"Negatives per question: min={min(neg_counts)}, max={max(neg_counts)}, avg={sum(neg_counts)/len(neg_counts):.1f}")

    # Show first example
    print(f"\nFirst question: {data[0]['question'][:70]}...")
    print(f"First question positives: {pos_counts[0]}, negatives: {neg_counts[0]}")

    # Show keys in paragraph
    print(f"\nParagraph keys: {list(data[0]['paragraphs'][0].keys())}")

if __name__ == "__main__":
    head_data_dir = Path(__file__).parent.parent / "head_data"

    # Find all JSON files in head_data (excluding subdirectories)
    json_files = [f for f in head_data_dir.glob("*.json")]

    if not json_files:
        print("No JSON files found in head_data/")
        sys.exit(1)

    for filepath in sorted(json_files):
        analyze_file(filepath)
