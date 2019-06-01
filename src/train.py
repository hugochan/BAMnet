import timeit
import argparse
import numpy as np

from core.bamnet.bamnet import BAMnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_data
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

    vocab2id = load_json(os.path.join(opt['data_dir'], 'vocab2id.json'))
    ctx_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    train_queries, train_raw_queries, train_query_mentions, train_memories, _, train_gold_ans_inds, _ = train_vec
    train_queries, train_query_words, train_query_lengths, train_memories = vectorize_data(train_queries, train_query_mentions, \
                                        train_memories, max_query_size=opt['query_size'], \
                                        max_query_markup_size=opt['query_markup_size'], \
                                        max_mem_size=opt['mem_size'], \
                                        max_ans_bow_size=opt['ans_bow_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        vocab2id=vocab2id)

    valid_queries, valid_raw_queries, valid_query_mentions, valid_memories, valid_cand_labels, valid_gold_ans_inds, valid_gold_ans_labels = valid_vec
    valid_queries, valid_query_words, valid_query_lengths, valid_memories = vectorize_data(valid_queries, valid_query_mentions, \
                                        valid_memories, max_query_size=opt['query_size'], \
                                        max_query_markup_size=opt['query_markup_size'], \
                                        max_mem_size=opt['mem_size'], \
                                        max_ans_bow_size=opt['ans_bow_size'], \
                                        max_ans_path_bow_size=opt['ans_path_bow_size'], \
                                        vocab2id=vocab2id)

    start = timeit.default_timer()

    model = BAMnetAgent(opt, ctx_stopwords, vocab2id)
    model.train([train_memories, train_queries, train_query_words, train_raw_queries, train_query_mentions, train_query_lengths], train_gold_ans_inds, \
        [valid_memories, valid_queries, valid_query_words, valid_raw_queries, valid_query_mentions, valid_query_lengths], \
        valid_gold_ans_inds, valid_cand_labels, valid_gold_ans_labels)

    print('Runtime: %ss' % (timeit.default_timer() - start))
