import timeit
import argparse
import numpy as np

from core.bamnet.entnet import EntnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_ent_data
from core.utils.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--datatype', default='test', type=str, help='data type: {train, valid, test}')
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    cfg = vars(parser.parse_args())
    opt = get_config(cfg['config'])

    # Ensure data is built
    build(opt['data_dir'])
    data_vec = load_json(os.path.join(opt['data_dir'], opt['test_data']))

    queries, memories, ent_labels, ent_inds = data_vec
    queries, query_lengths, memories = vectorize_ent_data(queries, \
                                        memories, max_query_size=opt['query_size'], \
                                        max_seed_ent_name_size=opt['max_seed_ent_name_size'], \
                                        max_seed_type_name_size=opt['max_seed_type_name_size'], \
                                        max_seed_rel_name_size=opt['max_seed_rel_name_size'], \
                                        max_seed_rel_size=opt['max_seed_rel_size'])

    start = timeit.default_timer()

    ent_model = EntnetAgent(opt)
    acc = ent_model.evaluate([memories, queries, query_lengths], ent_inds, batch_size=opt['test_batch_size'])
    print('acc: {}'.format(acc))
    print('Runtime: %ss' % (timeit.default_timer() - start))
