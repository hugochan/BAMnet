'''
Created on Oct, 2017

@author: hugo

'''
import torch
from torch.autograd import Variable
import numpy as np


def to_cuda(x, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        x = x.cuda()
    return x

# One pass over the dataset
def next_batch(memories, queries, query_words, raw_queries, query_mentions, query_lengths, gold_ans_inds, batch_size):
    for i in range(0, len(memories), batch_size):
        yield (memories[i: i + batch_size], queries[i: i + batch_size], query_words[i: i + batch_size], raw_queries[i: i + batch_size], query_mentions[i: i + batch_size], query_lengths[i: i + batch_size]), gold_ans_inds[i: i + batch_size]

# One pass over the dataset
def next_ent_batch(memories, queries, query_lengths, gold_inds, batch_size):
    for i in range(0, len(memories), batch_size):
        yield (memories[i: i + batch_size], queries[i: i + batch_size], query_lengths[i: i + batch_size]), gold_inds[i: i + batch_size]
