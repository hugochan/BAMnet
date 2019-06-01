'''
Created on Sep, 2018

@author: hugo

'''
import os
import timeit
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss, MultiLabelMarginLoss
import torch.backends.cudnn as cudnn

from .ent_modules import Entnet
from .utils import to_cuda, next_ent_batch
from ..utils.utils import load_ndarray
from ..utils.generic_utils import unique
from ..utils.metrics import *


class EntnetAgent(object):
    def __init__(self, opt):
        opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
        if opt['cuda']:
            print('[ Using CUDA ]')
            torch.cuda.set_device(opt['gpu'])
            # It enables benchmark mode in cudnn, which
            # leads to faster runtime when the input sizes do not vary.
            cudnn.benchmark = True

        self.opt = opt
        if self.opt['pre_word2vec']:
            pre_w2v = load_ndarray(self.opt['pre_word2vec'])
        else:
            pre_w2v = None

        self.ent_model = Entnet(opt['vocab_size'], opt['vocab_embed_size'], \
                opt['o_embed_size'], opt['hidden_size'], \
                opt['num_ent_types'], opt['num_relations'], \
                seq_enc_type=opt['seq_enc_type'], \
                word_emb_dropout=opt['word_emb_dropout'], \
                que_enc_dropout=opt['que_enc_dropout'], \
                ent_enc_dropout=opt['ent_enc_dropout'], \
                pre_w2v=pre_w2v, \
                num_hops=opt['num_ent_hops'], \
                att=opt['attention'], \
                use_cuda=opt['cuda'])
        if opt['cuda']:
            self.ent_model.cuda()

        self.loss_fn = MultiLabelMarginLoss()

        optim_params = [p for p in self.ent_model.parameters() if p.requires_grad]
        self.optimizers = {'entnet': optim.Adam(optim_params, lr=opt['learning_rate'])}
        self.scheduler = ReduceLROnPlateau(self.optimizers['entnet'], mode='min', \
                    patience=self.opt['valid_patience'] // 3, verbose=True)

        if opt.get('model_file') and os.path.isfile(opt['model_file']):
            print('Loading existing ent_model parameters from ' + opt['model_file'])
            self.load(opt['model_file'])
        else:
            self.save()
            self.load(opt['model_file'])
        super(EntnetAgent, self).__init__()

    def train(self, train_X, train_y, valid_X, valid_y, seed=1234):
        print('Training size: {}, Validation size: {}'.format(len(train_y), len(valid_y)))
        random1 = np.random.RandomState(seed)
        random2 = np.random.RandomState(seed)
        random3 = np.random.RandomState(seed)
        random4 = np.random.RandomState(seed)
        memories, queries, query_lengths = train_X
        ent_inds = train_y

        valid_memories, valid_queries, valid_query_lengths = valid_X
        valid_ent_inds = valid_y

        n_incr_error = 0  # nb. of consecutive increase in error
        best_loss = float("inf")
        best_acc = 0
        num_batches = len(queries) // self.opt['batch_size'] + (len(queries) % self.opt['batch_size'] != 0)
        num_valid_batches = len(valid_queries) // self.opt['batch_size'] + (len(valid_queries) % self.opt['batch_size'] != 0)
        for epoch in range(1, self.opt['num_epochs'] + 1):
            start = timeit.default_timer()
            n_incr_error += 1
            random1.shuffle(memories)
            random2.shuffle(queries)
            random3.shuffle(query_lengths)
            random4.shuffle(ent_inds)
            train_gen = next_ent_batch(memories, queries, query_lengths, ent_inds, self.opt['batch_size'])
            train_loss = 0
            for batch_xs, batch_ys in train_gen:
                train_loss += self.train_step(batch_xs, batch_ys) / num_batches

            valid_gen = next_ent_batch(valid_memories, valid_queries, valid_query_lengths, valid_ent_inds, self.opt['batch_size'])
            valid_loss = 0
            for batch_valid_xs, batch_valid_ys in valid_gen:
                valid_loss += self.train_step(batch_valid_xs, batch_valid_ys, is_training=False) / num_valid_batches
            self.scheduler.step(valid_loss)

            if epoch > 0:
                valid_acc = self.evaluate(valid_X, valid_ent_inds, batch_size=1, silence=True)
                # valid_acc = 0.
                print('Epoch {}/{}: Runtime: {}s, Training loss: {:.4}, validation loss: {:.4}, validation ACC: {:.4}'.format(epoch, self.opt['num_epochs'], \
                                                    int(timeit.default_timer() - start), train_loss, valid_loss, valid_acc))

                # self.scheduler.step(valid_acc)
                # if valid_acc > best_acc:
                #     best_acc = valid_acc
                #     n_incr_error = 0
                #     self.save()

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    n_incr_error = 0
                    self.save()

                if n_incr_error >= self.opt['valid_patience']:
                    print('Early stopping occured. Optimization Finished!')
                    self.save(self.opt['model_file'] + '.final')
                    break

    def evaluate(self, xs, ys, batch_size=1, silence=False):
        '''Prediction scores are returned in the verbose mode.
        '''
        if not silence:
            print('Data size: {}'.format(len(xs[0])))
        memories, queries, query_lengths = xs
        gen = next_ent_batch(memories, queries, query_lengths, ys, batch_size)
        correct = 0
        num_samples = 0
        for batch_xs, batch_ys in gen:
            correct += self.evaluate_step(batch_xs, batch_ys)
            num_samples += len(batch_ys)
        acc = 100 * correct / num_samples
        return acc

    def predict(self, xs, cand_labels, batch_size=1, silence=False):
        if not silence:
            print('Data size: {}'.format(len(xs[0])))
        memories, queries, query_lengths = xs
        gen = next_ent_batch(memories, queries, query_lengths, cand_labels, batch_size)
        predictions = []
        for batch_xs, batch_cands in gen:
            batch_pred = self.predict_step(batch_xs, batch_cands)
            predictions.extend(batch_pred)
        return predictions

    def train_step(self, xs, ys, is_training=True):
        # Sets the module in training mode.
        # This has any effect only on modules such as Dropout or BatchNorm.
        self.ent_model.train(mode=is_training)
        with torch.set_grad_enabled(is_training):
            # Organize inputs for network
            memories = [to_cuda(torch.LongTensor(np.array(x)), self.opt['cuda']) for x in zip(*xs[0])]
            queries = to_cuda(torch.LongTensor(xs[1]), self.opt['cuda'])
            query_lengths = to_cuda(torch.LongTensor(xs[2]), self.opt['cuda'])
            mem_hop_scores = self.ent_model(memories, queries, query_lengths)
            # ys = to_cuda(torch.LongTensor(ys), self.opt['cuda']).squeeze(-1)
            # Set margin
            ys, mask_ys = self.pack_gold_ans(ys, mem_hop_scores[-1].size(1), placeholder=-1)

            loss = 0
            for _, s in enumerate(mem_hop_scores):
                loss += self.loss_fn(s, ys)
            loss /= len(mem_hop_scores)

            if is_training:
                for o in self.optimizers.values():
                    o.zero_grad()
                loss.backward()
                for o in self.optimizers.values():
                    o.step()
            return loss.item()

    def evaluate_step(self, xs, ys):
        self.ent_model.train(mode=False)
        with torch.set_grad_enabled(False):
            # Organize inputs for network
            memories = [to_cuda(torch.LongTensor(np.array(x)), self.opt['cuda']) for x in zip(*xs[0])]
            queries = to_cuda(torch.LongTensor(xs[1]), self.opt['cuda'])
            query_lengths = to_cuda(torch.LongTensor(xs[2]), self.opt['cuda'])
            scores = self.ent_model(memories, queries, query_lengths)[-1]
            ys = to_cuda(torch.LongTensor(ys), self.opt['cuda']).squeeze(1)

            predictions = scores.max(1)[1].type_as(ys)
            correct = predictions.eq(ys).sum()
            return correct.item()

    def predict_step(self, xs, cand_labels):
        self.ent_model.train(mode=False)
        with torch.set_grad_enabled(False):
            # Organize inputs for network
            memories = [to_cuda(torch.LongTensor(np.array(x)), self.opt['cuda']) for x in zip(*xs[0])]
            queries = to_cuda(torch.LongTensor(xs[1]), self.opt['cuda'])
            query_lengths = to_cuda(torch.LongTensor(xs[2]), self.opt['cuda'])
            scores = self.ent_model(memories, queries, query_lengths)[-1]

            predictions = self.ranked_predictions(cand_labels, scores)
            return predictions

    def pack_gold_ans(self, x, N, placeholder=-1):
        y = np.ones((len(x), N), dtype='int64') * placeholder
        mask = np.zeros((len(x), N))
        for i in range(len(x)):
            y[i, :len(x[i])] = x[i]
            mask[i, :len(x[i])] = 1
        return to_cuda(torch.LongTensor(y), self.opt['cuda']), to_cuda(torch.Tensor(mask), self.opt['cuda'])

    def ranked_predictions(self, cand_labels, scores):
        _, sorted_inds = scores.sort(descending=True, dim=1)
        return [cand_labels[i][r[0]] if len(cand_labels[i]) > 0 else '' \
                for i, r in enumerate(sorted_inds)]

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path:
            checkpoint = {}
            checkpoint['entnet'] = self.ent_model.state_dict()
            checkpoint['entnet_optim'] = self.optimizers['entnet'].state_dict()
            with open(path, 'wb') as write:
                torch.save(checkpoint, write)
                print('Saved ent_model to {}'.format(path))

    def load(self, path):
        with open(path, 'rb') as read:
            checkpoint = torch.load(read, map_location=lambda storage, loc: storage)
        self.ent_model.load_state_dict(checkpoint['entnet'])
        self.optimizers['entnet'].load_state_dict(checkpoint['entnet_optim'])
