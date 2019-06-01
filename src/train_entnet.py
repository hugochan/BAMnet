import timeit
import argparse
import numpy as np

from core.bamnet.entnet import EntnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_ent_data
from core.utils.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    cfg = vars(parser.parse_args())
    opt = get_config(cfg['config'])
    print_config(opt)

    # Ensure data is built
    build(opt['data_dir'])
    train_vec = load_json(os.path.join(opt['data_dir'], opt['train_data']))
    valid_vec = load_json(os.path.join(opt['data_dir'], opt['valid_data']))

    train_queries, train_memories, _, train_ent_inds = train_vec
    train_queries, train_query_lengths, train_memories = vectorize_ent_data(train_queries, \
                                        train_memories, max_query_size=opt['query_size'], \
                                        max_seed_ent_name_size=opt['max_seed_ent_name_size'], \
                                        max_seed_type_name_size=opt['max_seed_type_name_size'], \
                                        max_seed_rel_name_size=opt['max_seed_rel_name_size'], \
                                        max_seed_rel_size=opt['max_seed_rel_size'])

    valid_queries, valid_memories, _, valid_ent_inds = valid_vec
    valid_queries, valid_query_lengths, valid_memories = vectorize_ent_data(valid_queries, \
                                        valid_memories, max_query_size=opt['query_size'], \
                                        max_seed_ent_name_size=opt['max_seed_ent_name_size'], \
                                        max_seed_type_name_size=opt['max_seed_type_name_size'], \
                                        max_seed_rel_name_size=opt['max_seed_rel_name_size'], \
                                        max_seed_rel_size=opt['max_seed_rel_size'])

    start = timeit.default_timer()

    ent_model = EntnetAgent(opt)
    ent_model.train([train_memories, train_queries, train_query_lengths], train_ent_inds, \
        [valid_memories, valid_queries, valid_query_lengths], valid_ent_inds)

    print('Runtime: %ss' % (timeit.default_timer() - start))
