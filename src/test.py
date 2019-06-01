import timeit
import argparse

from core.bamnet.bamnet import BAMnetAgent
from core.build_data.build_all import build
from core.build_data.utils import vectorize_data
from core.utils.utils import *
from core.utils.generic_utils import unique
from core.utils.metrics import *


def dynamic_pred(pred, margin):
    predictions = []
    for i in range(len(pred)):
        predictions.append(unique([x[0] for x in pred[i] if x[1] + margin >= pred[i][0][1]]))
    return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    cfg = vars(parser.parse_args())
    opt = get_config(cfg['config'])

    # Ensure data is built
    build(opt['data_dir'])
    data_vec = load_json(os.path.join(opt['data_dir'], opt['test_data']))
    vocab2id = load_json(os.path.join(opt['data_dir'], 'vocab2id.json'))
    ctx_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

    queries, raw_queries, query_mentions, memories, cand_labels, _, gold_ans_labels = data_vec
    queries, query_words, query_lengths, memories_vec = vectorize_data(queries, query_mentions, memories, \
                                        max_query_size=opt['query_size'], \
                                        max_query_markup_size=opt['query_markup_size'], \
                                        max_ans_bow_size=opt['ans_bow_size'], \
                                        vocab2id=vocab2id)

    start = timeit.default_timer()

    model = BAMnetAgent(opt, ctx_stopwords, vocab2id)
    pred = model.predict([memories_vec, queries, query_words, raw_queries, query_mentions, query_lengths], cand_labels, batch_size=opt['test_batch_size'], margin=2)

    print('\nPredictions')
    for margin in opt['test_margin']:
        print('\nMargin: {}'.format(margin))
        predictions = dynamic_pred(pred, margin)
        calc_avg_f1(gold_ans_labels, predictions)
    print('Runtime: %ss' % (timeit.default_timer() - start))
    import pdb;pdb.set_trace()
