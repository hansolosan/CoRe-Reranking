import json
import bz2
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
from src.reranker_calib import Reranker

llm_name = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--llm', type=str, default='mistral', choices=['mistral', 'llama', 'phi', 'granite'])
parser.add_argument('--data', type=str, default='dbpedia-entity')
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--reranker', type=str, default='core', choices=['icr', 'qr', 'core'])
parser.add_argument('--temp', type=float, default=0.001)
parser.add_argument('--prune', type=float, default=0.0)
parser.add_argument('--num_head', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for reranking multiple queries simultaneously')
args = parser.parse_args()

# best temperature found, change if needed
if 'phi' in args.llm.lower() or 'llama' in args.llm.lower():
    args.temp = 0.1
else:
    args.temp = 0.001

def main():
    print('-'*50)
    print(f'reranking {args.data} top{args.top_k} with {args.llm} using {args.reranker} reranker (batch_size={args.batch_size})')
    if not torch.cuda.is_available():
        print('no gpu')
        return

    # create folder
    folder = f'../reranking_output/{args.llm}/top{args.top_k}'
    os.makedirs(f'../reranking_output', exist_ok=True)
    os.makedirs(f'../reranking_output/{args.llm}', exist_ok=True)
    os.makedirs(folder, exist_ok=True)

    # output file
    if args.reranker == 'core':
      retrieval_output_file = f'{folder}/{args.data}_{args.reranker}_temp{args.temp}_prune{args.prune}.json'
    else:
        retrieval_output_file = f'{folder}/{args.data}_{args.reranker}.json'
    if os.path.exists(retrieval_output_file):
        print('run already completed')
        return

    # load data and retrieval heads
    data_file = f'../retriever_output/{args.data}.json'
    data_file_bz2 = f'../retriever_output/{args.data}.json.bz2'
    if os.path.exists(data_file):
        query_set = json.load(open(data_file))
    elif os.path.exists(data_file_bz2):
        with bz2.open(data_file_bz2, 'rt', encoding='utf-8') as f:
            query_set = json.load(f)
    else:
        print(f'data file not found: {data_file} or {data_file_bz2}')
        return
    if args.reranker == 'icr':
        head_set = None
    else:
        if args.reranker == 'core':
            head_file = f'../head_data/{args.llm}/core_temp{args.temp}_prune{args.prune}.json'
        elif args.reranker == 'qr':
            head_file = f'../head_data/{args.llm}/qr.json'
        head_list = json.load(open(head_file))
        head_score_list = [([int(ll) for ll in l[0].split("-")],np.mean(l[1])) for l in head_list.items()]
        head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
        head_score_list = head_score_list[:args.num_head]
        head_set = []
        for i in range(args.num_head):
            head_set.append(head_score_list[i][0])

    # initialize the reranker model
    reranker = Reranker(llm_name[args.llm], head_set=head_set, prune=args.prune, batch_size=args.batch_size)
    retrieval_results = {}

    # rerank in batches
    batch_queries = []
    batch_meta = []  # store (query_idx, paragraphs) for each query in batch

    for i, query in enumerate(tqdm(query_set)):
        question = query['question']
        paragraphs = query['paragraphs']

        for p in paragraphs:
            if not isinstance(p['paragraph_text'], str):
                p['paragraph_text'] = str(p['paragraph_text'])

        # truncate document, comment out if not needed
        for p in paragraphs:
            p['paragraph_text'] = ' '.join(p['paragraph_text'].split(' ')[:300])

        documents = [(p['paragraph_text']).strip() for p in paragraphs]
        batch_queries.append((question, documents))
        batch_meta.append((query['idx'], paragraphs))

        # Process batch when full or at the end
        if len(batch_queries) == args.batch_size or i == len(query_set) - 1:
            batch_results = reranker.rerank_batch(batch_queries)

            for j, (sorted_doc_ids, sorted_doc_scores) in enumerate(batch_results):
                query_idx, paragraphs = batch_meta[j]
                retrieval_results[query_idx] = {}
                for _i, sorted_idx in enumerate(sorted_doc_ids):
                    retrieval_results[query_idx][str(paragraphs[sorted_idx]['idx'])] = sorted_doc_scores[_i]

            batch_queries = []
            batch_meta = []

    # save reranked results
    json.dump(retrieval_results, open(retrieval_output_file, 'w'), indent=2)
    print(f'saved retrieval results to {retrieval_output_file}')

if __name__ == '__main__':
    main()
