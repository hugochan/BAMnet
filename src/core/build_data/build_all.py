'''
Created on Oct, 2017

@author: hugo

'''
import os

from . import utils as build_utils
from ..utils.utils import *
from .build_data import build_vocab, build_data


def build(dpath, version=None, out_dir=None):
    if not build_utils.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        # An older version exists, so remove these outdated files.
        if build_utils.built(dpath):
            build_utils.remove_dir(dpath)
        if not os.path.exists(dpath):
            build_utils.make_dir(dpath)

        # Download the data.
        # Not implemented yet. Use local data for now.

        train_data = load_ndjson(os.path.join(dpath, 'train.json'))
        valid_data = load_ndjson(os.path.join(dpath, 'valid.json'))
        test_data = load_ndjson(os.path.join(dpath, 'test.json'))
        freebase = load_ndjson(os.path.join(dpath, 'freebase.json'), return_type='dict')
        used_fbkeys = load_json(os.path.join(dpath, 'fbkeys_train_valid.json'))

        if not (os.path.exists(os.path.join(out_dir, 'entity2id.json')) and \
            os.path.exists(os.path.join(out_dir, 'entityType2id.json')) and \
            os.path.exists(os.path.join(out_dir, 'relation2id.json')) and \
            os.path.exists(os.path.join(out_dir, 'vocab2id.json'))):
            entity2id, entityType2id, relation2id, vocab2id = build_vocab(train_data, valid_data, test_data, freebase, used_fbkeys)
            dump_json(entity2id, os.path.join(out_dir, 'entity2id.json'))
            dump_json(entityType2id, os.path.join(out_dir, 'entityType2id.json'))
            dump_json(relation2id, os.path.join(out_dir, 'relation2id.json'))
            dump_json(vocab2id, os.path.join(out_dir, 'vocab2id.json'))
        else:
            entity2id = load_json(os.path.join(out_dir, 'entity2id.json'))
            entityType2id = load_json(os.path.join(out_dir, 'entityType2id.json'))
            relation2id = load_json(os.path.join(out_dir, 'relation2id.json'))
            vocab2id = load_json(os.path.join(out_dir, 'vocab2id.json'))
            print('Using pre-built vocabs stored in %s' % out_dir)

        train_vec = build_data(train_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        valid_vec = build_data(valid_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        test_vec = build_data(test_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
        dump_json(train_vec, os.path.join(out_dir, 'train_vec.json'))
        dump_json(valid_vec, os.path.join(out_dir, 'valid_vec.json'))
        dump_json(test_vec, os.path.join(out_dir, 'test_vec.json'))
        print('Saved data to {}'.format(os.path.join(out_dir, 'train(valid, or test)_vec.json')))

        # Mark the data as built.
        build_utils.mark_done(dpath, version_string=version)
