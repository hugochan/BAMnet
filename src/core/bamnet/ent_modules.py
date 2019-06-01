'''
Created on Sep, 2018

@author: hugo

'''
import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from .modules import SeqEncoder, SelfAttention_CoAtt, Attention
from .utils import to_cuda


INF = 1e20
VERY_SMALL_NUMBER = 1e-10
class Entnet(nn.Module):
    def __init__(self, vocab_size, vocab_embed_size, o_embed_size, \
        hidden_size, num_ent_types, num_relations, \
        seq_enc_type='cnn', \
        word_emb_dropout=None, \
        que_enc_dropout=None,\
        ent_enc_dropout=None, \
        pre_w2v=None, \
        num_hops=1, \
        att='add', \
        use_cuda=True):
        super(Entnet, self).__init__()
        self.use_cuda = use_cuda
        self.seq_enc_type = seq_enc_type
        self.que_enc_dropout = que_enc_dropout
        self.ent_enc_dropout = ent_enc_dropout
        self.num_hops = num_hops
        self.hidden_size = hidden_size
        self.que_enc = SeqEncoder(vocab_size, vocab_embed_size, hidden_size, \
                        seq_enc_type=seq_enc_type, \
                        word_emb_dropout=word_emb_dropout, \
                        bidirectional=True, \
                        cnn_kernel_size=[2, 3], \
                        init_word_embed=pre_w2v, \
                        use_cuda=use_cuda).que_enc

        self.ent_enc = EntEncoder(o_embed_size, hidden_size, \
                        num_ent_types, num_relations, \
                        vocab_size=vocab_size, \
                        vocab_embed_size=vocab_embed_size, \
                        shared_embed=self.que_enc.embed, \
                        seq_enc_type=seq_enc_type, \
                        word_emb_dropout=word_emb_dropout, \
                        ent_enc_dropout=ent_enc_dropout, \
                        use_cuda=use_cuda)
        self.batchnorm = nn.BatchNorm1d(hidden_size)

        if seq_enc_type in ('lstm', 'gru'):
            self.self_atten = SelfAttention_CoAtt(hidden_size)
            print('[ Using self-attention on question encoder ]')

        self.ent_memory_hop = EntRomHop(hidden_size, hidden_size, hidden_size, atten_type=att)
        print('[ Using {}-hop entity memory update ]'.format(num_hops))

    def forward(self, memories, queries, query_lengths):
        x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask = memories
        x_rel_mask = self.create_mask_3D(x_rel_mask, x_rels.size(-1), use_cuda=self.use_cuda)

        # Question encoder
        if self.seq_enc_type in ('lstm', 'gru'):
            Q_r = self.que_enc(queries, query_lengths)[0]
            if self.que_enc_dropout:
                Q_r = F.dropout(Q_r, p=self.que_enc_dropout, training=self.training)

            query_mask = self.create_mask(query_lengths, Q_r.size(1), self.use_cuda)
            q_r = self.self_atten(Q_r, query_lengths, query_mask)
        else:
            q_r = self.que_enc(queries, query_lengths)[1]
            if self.que_enc_dropout:
                q_r = F.dropout(q_r, p=self.que_enc_dropout, training=self.training)

        # Entity encoder
        ent_val, ent_key = self.ent_enc(x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask)

        ent_val = torch.cat([each.unsqueeze(2) for each in ent_val], 2)
        ent_key = torch.cat([each.unsqueeze(2) for each in ent_key], 2)
        ent_val = torch.sum(ent_val, 2)
        ent_key = torch.sum(ent_key, 2)

        mem_hop_scores = []
        mid_score = self.clf_score(q_r, ent_key)
        mem_hop_scores.append(mid_score)

        for _ in range(self.num_hops):
            q_r = q_r + self.ent_memory_hop(q_r, ent_key, ent_val)
            q_r = self.batchnorm(q_r)
            mid_score = self.clf_score(q_r, ent_key)
            mem_hop_scores.append(mid_score)
        return mem_hop_scores

    def clf_score(self, q_r, ent_key):
        return torch.matmul(ent_key, q_r.unsqueeze(-1)).squeeze(-1)

    def create_mask(self, x, N, use_cuda=True):
        x = x.data
        mask = np.zeros((x.size(0), N))
        for i in range(x.size(0)):
            mask[i, :x[i]] = 1
        return to_cuda(torch.Tensor(mask), use_cuda)

    def create_mask_3D(self, x, N, use_cuda=True):
        x = x.data
        mask = np.zeros((x.size(0), x.size(1), N))
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                mask[i, j, :x[i, j]] = 1
        return to_cuda(torch.Tensor(mask), use_cuda)

class EntEncoder(nn.Module):
    """Entity Encoder"""
    def __init__(self, o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=None, \
                    vocab_embed_size=None, shared_embed=None, seq_enc_type='lstm', word_emb_dropout=None, \
                    ent_enc_dropout=None, use_cuda=True):
        super(EntEncoder, self).__init__()
        # Cannot have embed and vocab_size set as None at the same time.
        self.ent_enc_dropout = ent_enc_dropout
        self.hidden_size = hidden_size
        self.relation_embed = nn.Embedding(num_relations, o_embed_size, padding_idx=0)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, vocab_embed_size, padding_idx=0)
        self.vocab_embed_size = self.embed.weight.data.size(1)

        self.linear_node_name_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_node_type_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_rels_key = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.linear_node_name_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_node_type_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_rels_val = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)

        self.kg_enc_ent = SeqEncoder(vocab_size, \
                        self.vocab_embed_size, \
                        hidden_size, \
                        seq_enc_type=seq_enc_type, \
                        word_emb_dropout=word_emb_dropout, \
                        bidirectional=True, \
                        cnn_kernel_size=[3], \
                        shared_embed=shared_embed, \
                        use_cuda=use_cuda).que_enc # entity name

        self.kg_enc_type = SeqEncoder(vocab_size, \
                        self.vocab_embed_size, \
                        hidden_size, \
                        seq_enc_type=seq_enc_type, \
                        word_emb_dropout=word_emb_dropout, \
                        bidirectional=True, \
                        cnn_kernel_size=[3], \
                        shared_embed=shared_embed, \
                        use_cuda=use_cuda).que_enc # entity type name

        self.kg_enc_rel = SeqEncoder(vocab_size, \
                        self.vocab_embed_size, \
                        hidden_size, \
                        seq_enc_type=seq_enc_type, \
                        word_emb_dropout=word_emb_dropout, \
                        bidirectional=True, \
                        cnn_kernel_size=[3], \
                        shared_embed=shared_embed, \
                        use_cuda=use_cuda).que_enc # relation name

    def forward(self, x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask):
        node_ent_names, node_type_names, node_types, edge_rel_names, edge_rels = self.enc_kg_features(x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask)
        node_name_key = self.linear_node_name_key(node_ent_names)
        node_type_key = self.linear_node_type_key(node_type_names)
        rel_key = self.linear_rels_key(torch.cat([edge_rel_names, edge_rels], -1))

        node_name_val = self.linear_node_name_val(node_ent_names)
        node_type_val = self.linear_node_type_val(node_type_names)
        rel_val = self.linear_rels_val(torch.cat([edge_rel_names, edge_rels], -1))

        ent_comp_val = [node_name_val, node_type_val, rel_val]
        ent_comp_key = [node_name_key, node_type_key, rel_key]
        return ent_comp_val, ent_comp_key

    def enc_kg_features(self, x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask):
        node_ent_names = (self.kg_enc_ent(x_ent_names.view(-1, x_ent_names.size(-1)), x_ent_name_len.view(-1))[1]).view(x_ent_names.size(0), x_ent_names.size(1), -1)
        node_type_names = (self.kg_enc_type(x_type_names.view(-1, x_type_names.size(-1)), x_type_name_len.view(-1))[1]).view(x_type_names.size(0), x_type_names.size(1), -1)
        node_types = None
        edge_rel_names = torch.mean((self.kg_enc_rel(x_rel_names.view(-1, x_rel_names.size(-1)), x_rel_name_len.view(-1))[1]).view(x_rel_names.size(0), x_rel_names.size(1), x_rel_names.size(2), -1), 2)
        edge_rels = torch.mean(self.relation_embed(x_rels.view(-1, x_rels.size(-1))), 1).view(x_rels.size(0), x_rels.size(1), -1)

        if self.ent_enc_dropout:
            node_ent_names = F.dropout(node_ent_names, p=self.ent_enc_dropout, training=self.training)
            node_type_names = F.dropout(node_type_names, p=self.ent_enc_dropout, training=self.training)
            # node_types = F.dropout(node_types, p=self.ent_enc_dropout, training=self.training)
            edge_rel_names = F.dropout(edge_rel_names, p=self.ent_enc_dropout, training=self.training)
            edge_rels = F.dropout(edge_rels, p=self.ent_enc_dropout, training=self.training)
        return node_ent_names, node_type_names, node_types, edge_rel_names, edge_rels


class EntRomHop(nn.Module):
    def __init__(self, query_embed_size, in_memory_embed_size, hidden_size, atten_type='add'):
        super(EntRomHop, self).__init__()
        self.atten = Attention(hidden_size, query_embed_size, in_memory_embed_size, atten_type=atten_type)
        self.gru_step = GRUStep(hidden_size, in_memory_embed_size)

    def forward(self, h_state, key_memory_embed, val_memory_embed, atten_mask=None):
        attention = self.atten(h_state, key_memory_embed, atten_mask=atten_mask)
        probs = torch.softmax(attention, dim=-1)
        memory_output = torch.bmm(probs.unsqueeze(1), val_memory_embed).squeeze(1)
        h_state = self.gru_step(h_state, memory_output)
        return h_state

class GRUStep(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        '''GRU module'''
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input_):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input_], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input_], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input_], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state
