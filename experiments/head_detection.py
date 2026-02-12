import json
import os
from tqdm import tqdm
import argparse
import numpy as np

llm_name = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}

parser = argparse.ArgumentParser(description='Detect retrieval heads using CoRe or QR methods.')
parser.add_argument('--llm', type=str, default='mistral', choices=['mistral', 'llama', 'phi', 'granite'])
parser.add_argument('--detector', type=str, default='core', choices=['qr', 'core'])
parser.add_argument('--temp', type=float, default=0.001)
parser.add_argument('--prune', type=float, default=0.0)
parser.add_argument('--save_features', action='store_true',
                    help='Also save raw per-document attention features to .npz file (for debugging/comparison). '
                         'NOTE: These are uncalibrated features (no N/A subtraction), unlike extract_head_features.py which applies calibration by default.')
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to process (default: all)')
parser.add_argument('--max_doc_tokens', type=int, default=None,
                    help='Maximum words per document (default: None, no truncation)')
args = parser.parse_args()

def main():
    print('-'*50)
    print(f'retrieving heads for {args.llm} using {args.detector} detector')

    os.makedirs(f'../head_data/{args.llm}', exist_ok=True)
    if args.detector == 'core':
        output_file = f'../head_data/{args.llm}/{args.detector}_temp{args.temp}_prune{args.prune}.json'
    else:
        output_file = f'../head_data/{args.llm}/{args.detector}.json'
    if os.path.exists(output_file) and not args.save_features:
        print(f'run already completed: {output_file}')
        return

    if args.detector == 'qr':
        from src.qr_detector import HeadDetector
        query_set = json.load(open(f'../head_data/nq_qr.json'))
        detector = HeadDetector(llm_name[args.llm])
    elif args.detector == 'core':
        from src.core_detector import HeadDetector
        query_set = json.load(open(f'../head_data/nq_core.json'))
        detector = HeadDetector(llm_name[args.llm], args.temp, args.prune)

    # Limit samples if requested
    if args.max_samples is not None:
        query_set = query_set[:args.max_samples]
        print(f'Limited to {len(query_set)} samples')

    # Storage for features if --save_features is enabled
    all_features = []
    all_labels = []
    docs_per_query = []

    # Truncation tracking
    docs_truncated = 0
    docs_total = 0

    for _, query in enumerate(tqdm(query_set)):
        question = query['question']
        paragraphs = query['paragraphs']

        neg_idx = []
        pos_idx = None
        for i in range(len(paragraphs)):
            if args.detector == 'qr':
                if paragraphs[i]['is_gold']:
                    pos_idx = i
            elif args.detector == 'core':
                if paragraphs[i]['is_positive']:
                    pos_idx = i
                elif paragraphs[i]['is_negative']:
                    neg_idx.append(i)
            if not isinstance(paragraphs[i]['paragraph_text'], str):
                paragraphs[i]['paragraph_text'] = str(paragraphs[i]['paragraph_text'])

        # Process documents with optional truncation
        documents = []
        for p in paragraphs:
            text = p['paragraph_text'].strip()
            docs_total += 1
            if args.max_doc_tokens is not None:
                words = text.split()
                if len(words) > args.max_doc_tokens:
                    docs_truncated += 1
                    text = ' '.join(words[:args.max_doc_tokens])
            documents.append(text)

        # Compute retrieval score, optionally returning features
        if args.save_features and args.detector == 'core':
            result = detector.compute_retrieval_score(question, documents, pos_idx, neg_idx,
                                                       return_features=True)
            if result is not None:
                all_features.append(result['features'])
                all_labels.append(result['labels'])
                docs_per_query.append(len(documents))
        else:
            detector.compute_retrieval_score(question, documents, pos_idx, neg_idx)

    head_score_list = detector.get_head_score()

    # save detection results
    json.dump(head_score_list, open(output_file, 'w'), indent=2)
    print(f'saved detection results to {output_file}')

    # Report truncation stats
    if args.max_doc_tokens is not None:
        truncated_pct = (docs_truncated / docs_total * 100) if docs_total > 0 else 0
        print(f'Truncation stats (max_doc_tokens={args.max_doc_tokens}): {docs_truncated}/{docs_total} ({truncated_pct:.1f}%) documents truncated')

    # Save features if --save_features is enabled
    if args.save_features and all_features:
        features_file = f'../head_data/{args.llm}/{args.detector}_temp{args.temp}_prune{args.prune}_features.npz'
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)
        docs_per_query = np.array(docs_per_query, dtype=np.int32)

        np.savez_compressed(
            features_file,
            features=all_features,
            labels=all_labels,
            docs_per_query=docs_per_query
        )
        print(f'saved features to {features_file}')
        print(f'  Features shape: {all_features.shape}')
        print(f'  Labels: {(all_labels == 1).sum()} positive, {(all_labels == 0).sum()} negative, {(all_labels == -1).sum()} other')
        print(f'  NOTE: These are UNCALIBRATED features (no N/A subtraction).')
        print(f'        To compare with extract_head_features.py, use --no_calibration flag there.')

if __name__ == '__main__':
    main()
