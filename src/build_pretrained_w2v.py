'''
Created on Oct, 2017

@author: hugo

'''
import argparse
import os

from core.utils.utils import load_json
from core.utils.generic_utils import dump_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', '--embed_path', required=True, type=str, help='path to the pretrained word embeddings')
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-out', '--out_path', required=True, type=str, help='path to the output path')
    parser.add_argument('-emb_size', '--emb_size', required=True, type=int, help='embedding size')
    parser.add_argument('--binary', action='store_true', help='flag: binary file')
    args = parser.parse_args()

    vocab_dict = load_json(os.path.join(args.data_dir, 'vocab2id.json'))
    dump_embeddings(vocab_dict, args.embed_path, args.out_path, emb_size=args.emb_size, binary=True if args.binary else False)
