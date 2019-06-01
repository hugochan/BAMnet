'''
Created on Oct, 2017

@author: hugo

'''
import argparse

from core.build_data.build_data import build_vocab, build_data, build_seed_ent_data
from core.utils.utils import *
from core.build_data import utils as build_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
    parser.add_argument('-fb_dir', '--fb_dir', required=True, type=str, help='path to the freebase dir')
    parser.add_argument('-out_dir', '--out_dir', required=True, type=str, help='path to the output dir')
    parser.add_argument('-dtype', '--data_type', default='qa', type=str, help='data type')
    parser.add_argument('-min_freq', '--min_freq', default=1, type=int, help='min word vocab freq')
    parser.add_argument('-topn', '--topn', default=12, type=int, help='top n candidates')
    args = parser.parse_args()

    train_data = load_ndjson(os.path.join(args.data_dir, 'new_train_seed_15.json'))
    valid_data = load_ndjson(os.path.join(args.data_dir, 'new_valid_seed_15.json'))
    # test_data = load_ndjson(os.path.join(args.data_dir, 'test_fb_suggest_key.json'))
    test_data = load_ndjson(os.path.join(args.data_dir, 'test_seed_2_smart.json'))
    freebase = load_ndjson(os.path.join(args.fb_dir, 'freebase_full.json'), return_type='dict')

    if not (os.path.exists(os.path.join(args.out_dir, 'entity2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'entityType2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'relation2id.json')) and \
        os.path.exists(os.path.join(args.out_dir, 'vocab2id.json'))):

        used_fbkeys = set()
        for each in train_data + valid_data:
            used_fbkeys.update(each['freebaseKeyCands'][:args.topn])
        print('# of used_fbkeys: {}'.format(len(used_fbkeys)))

        entity2id, entityType2id, relation2id, vocab2id = build_vocab(train_data + valid_data, freebase, used_fbkeys, min_freq=args.min_freq)
        dump_json(entity2id, os.path.join(args.out_dir, 'entity2id.json'))
        dump_json(entityType2id, os.path.join(args.out_dir, 'entityType2id.json'))
        dump_json(relation2id, os.path.join(args.out_dir, 'relation2id.json'))
        dump_json(vocab2id, os.path.join(args.out_dir, 'vocab2id.json'))
    else:
        entity2id = load_json(os.path.join(args.out_dir, 'entity2id.json'))
        entityType2id = load_json(os.path.join(args.out_dir, 'entityType2id.json'))
        relation2id = load_json(os.path.join(args.out_dir, 'relation2id.json'))
        vocab2id = load_json(os.path.join(args.out_dir, 'vocab2id.json'))
        print('Using pre-built vocabs stored in %s' % args.out_dir)

    if args.data_type == 'qa':
        train_vec = build_data(train_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        valid_vec = build_data(valid_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        test_vec = build_data(test_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        dump_json(train_vec, os.path.join(args.out_dir, 'train_vec.json'))
        dump_json(valid_vec, os.path.join(args.out_dir, 'valid_vec.json'))
        dump_json(test_vec, os.path.join(args.out_dir, 'fb_test_vec_notime.json'))
        print('Saved data to {}'.format(os.path.join(args.out_dir, 'train(valid, or test)_vec.json')))
    else:
        train_vec = build_seed_ent_data(train_data, freebase, entity2id, entityType2id, relation2id, vocab2id, args.topn, dtype='train')
        valid_vec = build_seed_ent_data(valid_data, freebase, entity2id, entityType2id, relation2id, vocab2id, args.topn, dtype='valid')
        test_vec = build_seed_ent_data(test_data, freebase, entity2id, entityType2id, relation2id, vocab2id, args.topn, dtype='test')
        dump_json(train_vec, os.path.join(args.out_dir, 'train_seed_{}_ent_vec.json'.format(args.topn)))
        dump_json(valid_vec, os.path.join(args.out_dir, 'valid_seed_{}_ent_vec.json'.format(args.topn)))
        dump_json(test_vec, os.path.join(args.out_dir, 'test_seed_{}_hop2_ent_vec_smart.json'.format(2)))
        print('Saved data to {}'.format(os.path.join(args.out_dir, 'train(valid, or test)_seed_X_ent_vec.json')))

    # Mark the data as built.
    build_utils.mark_done(args.out_dir)
