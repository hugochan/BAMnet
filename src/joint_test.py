import timeit
import argparse
import numpy as np

from core.bamnet.entnet import EntnetAgent
from core.bamnet.bamnet import BAMnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_ent_data, vectorize_data
from core.build_data.build_data import build_data
from core.utils.generic_utils import unique
from core.utils.utils import *
from core.utils.metrics import *


def dynamic_pred(pred, margin):
    predictions = []
    for i in range(len(pred)):
        predictions.append(unique([x[0] for x in pred[i] if x[1] + margin >= pred[i][0][1]]))
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-bamnet_config', '--bamnet_config', required=True, type=str, help='path to the config file')
    parser.add_argument('-entnet_config', '--entnet_config', required=True, type=str, help='path to the config file')
    parser.add_argument('-raw_data', '--raw_data_dir', required=True, type=str, help='raw data dir')
    cfg = vars(parser.parse_args())
    bamnet_opt = get_config(cfg['bamnet_config'])
    entnet_opt = get_config(cfg['entnet_config'])

    start = timeit.default_timer()
    # Entnet
    # Ensure data is built
    build(entnet_opt['data_dir'])
    data_vec = load_json(os.path.join(entnet_opt['data_dir'], entnet_opt['test_data']))

    queries, memories, ent_labels, ent_inds = data_vec
    queries, query_lengths, memories = vectorize_ent_data(queries, \
                                        memories, max_query_size=entnet_opt['query_size'], \
                                        max_seed_ent_name_size=entnet_opt['max_seed_ent_name_size'], \
                                        max_seed_type_name_size=entnet_opt['max_seed_type_name_size'], \
                                        max_seed_rel_name_size=entnet_opt['max_seed_rel_name_size'], \
                                        max_seed_rel_size=entnet_opt['max_seed_rel_size'])

    ent_model = EntnetAgent(entnet_opt)
    acc = ent_model.evaluate([memories, queries, query_lengths], ent_inds, batch_size=entnet_opt['test_batch_size'])
    print('acc: {}'.format(acc))
    pred_seed_ents = ent_model.predict([memories, queries, query_lengths], ent_labels, batch_size=entnet_opt['test_batch_size'])


    # BAMnet
    # Ensure data is built
    build(bamnet_opt['data_dir'])
    entity2id = load_json(os.path.join(bamnet_opt['data_dir'], 'entity2id.json'))
    entityType2id = load_json(os.path.join(bamnet_opt['data_dir'], 'entityType2id.json'))
    relation2id = load_json(os.path.join(bamnet_opt['data_dir'], 'relation2id.json'))
    vocab2id = load_json(os.path.join(bamnet_opt['data_dir'], 'vocab2id.json'))
    ctx_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    # Build data in real time
    freebase = load_ndjson(os.path.join(cfg['raw_data_dir'], 'freebase_full.json'), return_type='dict')
    test_data = load_ndjson(os.path.join(cfg['raw_data_dir'], 'raw_test.json'))
    data_vec = build_data(test_data, freebase, entity2id, entityType2id, relation2id, vocab2id, pred_seed_ents=pred_seed_ents)

    queries, raw_queries, query_mentions, memories, cand_labels, _, gold_ans_labels = data_vec
    queries, query_words, query_lengths, memories_vec = vectorize_data(queries, query_mentions, memories, \
                                        max_query_size=bamnet_opt['query_size'], \
                                        max_query_markup_size=bamnet_opt['query_markup_size'], \
                                        max_ans_bow_size=bamnet_opt['ans_bow_size'], \
                                        vocab2id=vocab2id)

    model = BAMnetAgent(bamnet_opt, ctx_stopwords, vocab2id)
    pred = model.predict([memories_vec, queries, query_words, raw_queries, query_mentions, query_lengths], cand_labels, batch_size=bamnet_opt['test_batch_size'], margin=2)

    print('\nPredictions')
    for margin in bamnet_opt['test_margin']:
        print('\nMargin: {}'.format(margin))
        predictions = dynamic_pred(pred, margin)
        calc_avg_f1(gold_ans_labels, predictions)
    print('Runtime: %ss' % (timeit.default_timer() - start))
