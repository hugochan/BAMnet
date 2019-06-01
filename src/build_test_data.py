'''
Created on Oct, 2017

@author: hugo

'''
import argparse

from core.build_data.build_data import build_data
from core.utils.utils import *
from core.build_data import utils as build_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-fname', '--file_name', required=True, type=str, help='data file name')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-dv', '--data_version', required=True, type=str, help='data version')
    args = parser.parse_args()

    data = load_ndjson(os.path.join(args.data_dir, args.file_name))
    freebase = load_ndjson(os.path.join(args.data_dir, 'freebase.json'), return_type='dict')
    entity2id = load_json(os.path.join(args.data_dir, '{}/entity2id.json'.format(args.data_version)))
    entityType2id = load_json(os.path.join(args.data_dir, '{}/entityType2id.json'.format(args.data_version)))
    relation2id = load_json(os.path.join(args.data_dir, '{}/relation2id.json'.format(args.data_version)))
    vocab2id = load_json(os.path.join(args.data_dir, '{}/vocab2id.json'.format(args.data_version)))

    data_vec = build_data(data, freebase, entity2id, entityType2id, relation2id, vocab2id)
    dump_json(data_vec, os.path.join(args.out_dir, args.file_name.split('.')[0] + '_vec.json'))

    # Mark the data as built.
    build_utils.mark_done(args.out_dir)
