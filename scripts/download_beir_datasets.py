#!/usr/bin/env python3
"""
Download BEIR datasets and qrels from HuggingFace.

This script downloads all BEIR benchmark datasets and extracts qrels files
in TREC format (query_id, 0, doc_id, relevance).

Usage:
    python scripts/download_beir_datasets.py --output_dir data/beir
    python scripts/download_beir_datasets.py --datasets nq hotpotqa --output_dir data/beir
    python scripts/download_beir_datasets.py --list  # List available datasets

Datasets are saved as:
    {output_dir}/{dataset}/corpus.jsonl
    {output_dir}/{dataset}/queries.jsonl
    {output_dir}/{dataset}/qrels/test.tsv
    {output_dir}/{dataset}/qrels/train.tsv (if available)
    {output_dir}/{dataset}/qrels/dev.tsv (if available)
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader
    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False


# BEIR datasets available on HuggingFace
BEIR_DATASETS = {
    # Main BEIR datasets
    'msmarco': 'BeIR/msmarco',
    'trec-covid': 'BeIR/trec-covid',
    'nfcorpus': 'BeIR/nfcorpus',
    'nq': 'BeIR/nq',
    'hotpotqa': 'BeIR/hotpotqa',
    'fiqa': 'BeIR/fiqa',
    'arguana': 'BeIR/arguana',
    'webis-touche2020': 'BeIR/webis-touche2020',
    'quora': 'BeIR/quora',
    'dbpedia-entity': 'BeIR/dbpedia-entity',
    'scidocs': 'BeIR/scidocs',
    'fever': 'BeIR/fever',
    'climate-fever': 'BeIR/climate-fever',
    'scifact': 'BeIR/scifact',
    # CQADupStack subsets
    'cqadupstack-android': 'BeIR/cqadupstack/android',
    'cqadupstack-english': 'BeIR/cqadupstack/english',
    'cqadupstack-gaming': 'BeIR/cqadupstack/gaming',
    'cqadupstack-gis': 'BeIR/cqadupstack/gis',
    'cqadupstack-mathematica': 'BeIR/cqadupstack/mathematica',
    'cqadupstack-physics': 'BeIR/cqadupstack/physics',
    'cqadupstack-programmers': 'BeIR/cqadupstack/programmers',
    'cqadupstack-stats': 'BeIR/cqadupstack/stats',
    'cqadupstack-tex': 'BeIR/cqadupstack/tex',
    'cqadupstack-unix': 'BeIR/cqadupstack/unix',
    'cqadupstack-webmasters': 'BeIR/cqadupstack/webmasters',
    'cqadupstack-wordpress': 'BeIR/cqadupstack/wordpress',
}

# Alternative: Direct BEIR download URLs (if HF doesn't work)
BEIR_DOWNLOAD_URLS = {
    'msmarco': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip',
    'trec-covid': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip',
    'nfcorpus': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip',
    'nq': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip',
    'hotpotqa': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip',
    'fiqa': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip',
    'arguana': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip',
    'webis-touche2020': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip',
    'quora': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip',
    'dbpedia-entity': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip',
    'scidocs': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip',
    'fever': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip',
    'climate-fever': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip',
    'scifact': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
    'cqadupstack': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip',
}


def download_from_hf(dataset_name, hf_path, output_dir, splits=None):
    """
    Download dataset from HuggingFace and save in BEIR format.

    Args:
        dataset_name: Name of the dataset
        hf_path: HuggingFace dataset path
        output_dir: Output directory
        splits: List of splits to download (default: all available)

    Returns:
        True if successful, False otherwise
    """
    if not HF_AVAILABLE:
        print("  Error: 'datasets' library not installed. Run: pip install datasets")
        return False

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir = dataset_dir / 'qrels'
    qrels_dir.mkdir(exist_ok=True)

    try:
        # Handle cqadupstack subsets
        if '/' in hf_path and 'cqadupstack' in hf_path:
            parts = hf_path.split('/')
            hf_repo = '/'.join(parts[:2])  # BeIR/cqadupstack
            subset = parts[2]  # android, english, etc.

            # Load corpus
            print(f"  Loading corpus...")
            corpus_ds = load_dataset(hf_repo, 'corpus', split=subset)

            # Load queries
            print(f"  Loading queries...")
            queries_ds = load_dataset(hf_repo, 'queries', split=subset)

            # Load qrels (cqadupstack only has test)
            print(f"  Loading qrels...")
            qrels_splits = {'test': subset}
        else:
            # Load corpus
            print(f"  Loading corpus...")
            corpus_ds = load_dataset(hf_path, 'corpus', split='corpus')

            # Load queries
            print(f"  Loading queries...")
            queries_ds = load_dataset(hf_path, 'queries', split='queries')

            # Determine available qrels splits
            print(f"  Loading qrels...")
            qrels_splits = {}
            for split in ['test', 'train', 'dev', 'validation']:
                try:
                    test_ds = load_dataset(hf_path, 'default', split=split)
                    if len(test_ds) > 0:
                        qrels_splits[split] = split
                except Exception:
                    pass

        # Save corpus
        corpus_file = dataset_dir / 'corpus.jsonl'
        print(f"  Saving corpus to {corpus_file}...")
        with open(corpus_file, 'w') as f:
            for item in corpus_ds:
                doc = {
                    '_id': str(item.get('_id', item.get('id', ''))),
                    'title': item.get('title', ''),
                    'text': item.get('text', ''),
                }
                f.write(json.dumps(doc) + '\n')

        # Save queries
        queries_file = dataset_dir / 'queries.jsonl'
        print(f"  Saving queries to {queries_file}...")
        with open(queries_file, 'w') as f:
            for item in queries_ds:
                query = {
                    '_id': str(item.get('_id', item.get('id', ''))),
                    'text': item.get('text', ''),
                }
                f.write(json.dumps(query) + '\n')

        # Save qrels
        for split_name, split_key in qrels_splits.items():
            try:
                if 'cqadupstack' in hf_path:
                    qrels_ds = load_dataset(
                        '/'.join(hf_path.split('/')[:2]),
                        'default',
                        split=split_key
                    )
                else:
                    qrels_ds = load_dataset(hf_path, 'default', split=split_key)

                qrels_file = qrels_dir / f'{split_name}.tsv'
                print(f"  Saving qrels ({split_name}) to {qrels_file}...")

                with open(qrels_file, 'w') as f:
                    # Write header
                    f.write('query-id\tcorpus-id\tscore\n')
                    for item in qrels_ds:
                        qid = str(item.get('query-id', item.get('qid', '')))
                        cid = str(item.get('corpus-id', item.get('did', item.get('docid', ''))))
                        score = item.get('score', item.get('label', 1))
                        f.write(f'{qid}\t{cid}\t{score}\n')

            except Exception as e:
                print(f"  Warning: Could not load {split_name} qrels: {e}")

        return True

    except Exception as e:
        print(f"  Error downloading from HuggingFace: {e}")
        return False


def download_from_beir(dataset_name, output_dir):
    """
    Download dataset using BEIR library directly.

    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    if not BEIR_AVAILABLE:
        print("  BEIR library not available, trying direct download...")
        return download_direct(dataset_name, output_dir)

    try:
        # Handle cqadupstack
        if dataset_name.startswith('cqadupstack-'):
            beir_name = 'cqadupstack'
        else:
            beir_name = dataset_name

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"

        print(f"  Downloading from BEIR...")
        data_path = beir_util.download_and_unzip(url, str(output_dir))

        # For cqadupstack, the subset is in a subdirectory
        if dataset_name.startswith('cqadupstack-'):
            subset = dataset_name.replace('cqadupstack-', '')
            src_path = Path(data_path) / subset
            dst_path = output_dir / dataset_name
            if src_path.exists() and not dst_path.exists():
                src_path.rename(dst_path)

        return True

    except Exception as e:
        print(f"  Error with BEIR download: {e}")
        return download_direct(dataset_name, output_dir)


def download_direct(dataset_name, output_dir):
    """
    Download dataset directly from URL.

    Args:
        dataset_name: Name of the dataset
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    import urllib.request
    import zipfile
    import tempfile
    import shutil

    # Handle cqadupstack
    if dataset_name.startswith('cqadupstack-'):
        url_name = 'cqadupstack'
    else:
        url_name = dataset_name

    if url_name not in BEIR_DOWNLOAD_URLS:
        print(f"  Error: No download URL for {dataset_name}")
        return False

    url = BEIR_DOWNLOAD_URLS[url_name]

    # Check if already downloaded (for cqadupstack, check the main folder)
    if url_name == 'cqadupstack':
        cqa_dir = output_dir / 'cqadupstack'
        need_download = not cqa_dir.exists()
    else:
        dataset_dir = output_dir / dataset_name
        need_download = not dataset_dir.exists()

    try:
        if need_download:
            print(f"  Downloading from {url}...")

            # Download with progress
            tmp_path = output_dir / f'{url_name}.zip'

            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, block_num * block_size * 100 // total_size)
                    print(f"\r  Progress: {percent}%", end='', flush=True)

            urllib.request.urlretrieve(url, tmp_path, show_progress)
            print()  # New line after progress

            print(f"  Extracting...")
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            os.unlink(tmp_path)
        else:
            print(f"  Using cached download...")

        # For cqadupstack, copy subset directory
        if dataset_name.startswith('cqadupstack-'):
            subset = dataset_name.replace('cqadupstack-', '')
            src_path = output_dir / 'cqadupstack' / subset
            dst_path = output_dir / dataset_name

            if src_path.exists() and not dst_path.exists():
                print(f"  Copying {subset} subset...")
                shutil.copytree(src_path, dst_path)
            elif dst_path.exists():
                print(f"  Subset already exists...")

        return True

    except Exception as e:
        print(f"  Error with direct download: {e}")
        # Clean up partial download
        tmp_path = output_dir / f'{url_name}.zip'
        if tmp_path.exists():
            os.unlink(tmp_path)
        return False


def convert_qrels_to_trec(qrels_file, output_file):
    """
    Convert qrels to TREC format (qid, 0, docid, relevance).

    Args:
        qrels_file: Input qrels file (TSV with header or without)
        output_file: Output TREC format file
    """
    with open(qrels_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w') as f:
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')

            # Skip header
            if i == 0 and ('query' in parts[0].lower() or 'qid' in parts[0].lower()):
                continue

            if len(parts) >= 3:
                qid, docid, score = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                qid, docid = parts[0], parts[1]
                score = '1'
            else:
                continue

            # TREC format: qid Q0 docid relevance
            f.write(f'{qid}\t0\t{docid}\t{score}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Download BEIR datasets and qrels from HuggingFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets (uses direct download from TU Darmstadt)
  python scripts/download_beir_datasets.py --output_dir data/beir

  # Download specific datasets
  python scripts/download_beir_datasets.py --datasets nq hotpotqa scifact --output_dir data/beir

  # List available datasets
  python scripts/download_beir_datasets.py --list

  # Skip already downloaded datasets
  python scripts/download_beir_datasets.py --skip_existing --output_dir data/beir
"""
    )
    parser.add_argument('--output_dir', '-o', type=str, default='data/beir',
                        help='Output directory for datasets (default: data/beir)')
    parser.add_argument('--datasets', '-d', type=str, nargs='+', default=None,
                        help='Specific datasets to download (default: all)')
    parser.add_argument('--method', '-m', type=str, default='direct',
                        choices=['hf', 'beir', 'direct'],
                        help='Download method: direct (URL, default), hf (HuggingFace), beir (BEIR lib)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available datasets and exit')
    parser.add_argument('--trec_format', action='store_true',
                        help='Also save qrels in TREC format (qid, 0, docid, rel)')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that already exist')
    args = parser.parse_args()

    # List datasets
    if args.list:
        print("Available BEIR datasets:")
        print("-" * 50)
        for name in sorted(BEIR_DATASETS.keys()):
            print(f"  {name}")
        print("-" * 50)
        print(f"Total: {len(BEIR_DATASETS)} datasets")
        return

    # Determine datasets to download
    if args.datasets:
        datasets = args.datasets
        # Validate dataset names
        for d in datasets:
            if d not in BEIR_DATASETS:
                print(f"Error: Unknown dataset '{d}'")
                print(f"Run with --list to see available datasets")
                return
    else:
        datasets = list(BEIR_DATASETS.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(datasets)} BEIR dataset(s) to {output_dir}")
    print("=" * 60)

    # Track results
    success = []
    failed = []
    skipped = []

    for i, dataset_name in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {dataset_name}")

        # Check if exists
        dataset_dir = output_dir / dataset_name
        if args.skip_existing and dataset_dir.exists():
            qrels_dir = dataset_dir / 'qrels'
            if qrels_dir.exists() and any(qrels_dir.glob('*.tsv')):
                print(f"  Skipping (already exists)")
                skipped.append(dataset_name)
                continue

        # Download
        hf_path = BEIR_DATASETS[dataset_name]

        if args.method == 'hf':
            result = download_from_hf(dataset_name, hf_path, output_dir)
        elif args.method == 'beir':
            result = download_from_beir(dataset_name, output_dir)
        else:
            result = download_direct(dataset_name, output_dir)

        if result:
            success.append(dataset_name)

            # Convert to TREC format if requested
            if args.trec_format:
                qrels_dir = dataset_dir / 'qrels'
                if qrels_dir.exists():
                    for qrels_file in qrels_dir.glob('*.tsv'):
                        trec_file = qrels_file.with_suffix('.trec')
                        print(f"  Converting to TREC format: {trec_file.name}")
                        convert_qrels_to_trec(qrels_file, trec_file)
        else:
            failed.append(dataset_name)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print(f"\nFailed datasets: {', '.join(failed)}")

    print("\nQrels files saved to: {output_dir}/{dataset}/qrels/")
    print("  - test.tsv (all datasets)")
    print("  - train.tsv (some datasets)")
    print("  - dev.tsv (some datasets)")


if __name__ == '__main__':
    main()
