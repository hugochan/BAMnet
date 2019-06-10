'''
Created on Sep, 2017

@author: hugo

'''
import os
import math
import argparse
from itertools import count
from fuzzywuzzy import fuzz, process
from collections import defaultdict

from ..utils.utils import *
from ..utils.generic_utils import normalize_answer, unique
from ..utils.freebase_utils import if_filterout
from .. import config


IGNORE_DUMMY = True
ENT_TYPE_HOP = 1
# Entity mention types: 'NP', 'ORGANIZATION', 'DATE', 'NUMBER', 'MISC', 'ORDINAL', 'DURATION', 'PERSON', 'TIME', 'LOCATION'

def build_kb_data(kb, used_fbkeys=None):
    entities = defaultdict(int)
    entity_types = defaultdict(int)
    relations = defaultdict(int)
    vocabs = defaultdict(int)
    if not used_fbkeys:
        used_fbkeys = kb.keys()
    for k in used_fbkeys:
        if not k in kb:
            continue
        v = kb[k]
        entities[v['id']] += 1
        # We prefer notable_types than type since they are more representative.
        # If notable_types are not available, we use only the first available type.
        # We found the type field contains much noise.
        selected_types = (v['notable_types'] + v['type'])[:ENT_TYPE_HOP]
        for ent_type in selected_types:
            entity_types[ent_type] += 1
        for token in [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
            vocabs[token] += 1
        # Add entity vocabs
        selected_names = v['name'][:1] + v['alias'] # We need all topic entity alias
        for token in [y for x in selected_names for y in tokenize(x.lower())]:
            vocabs[token] += 1
        if not 'neighbors' in v:
            continue
        for kk, vv in v['neighbors'].items(): # 1st hop
            if if_filterout(kk):
                continue
            relations[kk] += 1
            # Add relation vocabs
            for token in [x for x in kk.lower().split('/')[-1].split('_')]:
                vocabs[token] += 1
            for nbr in vv:
                if isinstance(nbr, str):
                    for token in [y for y in tokenize(nbr.lower())]:
                        vocabs[token] += 1
                    continue
                elif isinstance(nbr, bool):
                    continue
                elif isinstance(nbr, float):
                    continue
                    # vocabs.update([y for y in tokenize(str(nbr).lower())])
                elif isinstance(nbr, dict):
                    nbr_k = list(nbr.keys())[0]
                    nbr_v = nbr[nbr_k]
                    entities[nbr_k] += 1
                    selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:ENT_TYPE_HOP]
                    for ent_type in selected_types:
                        entity_types[ent_type] += 1
                    selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                    for token in [y for x in selected_names for y in tokenize(x.lower())] + \
                        [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
                        vocabs[token] += 1
                    if not 'neighbors' in nbr_v:
                        continue
                    for kkk, vvv in nbr_v['neighbors'].items(): # 2nd hop
                        if if_filterout(kkk):
                            continue
                        relations[kkk] += 1
                        # Add relation vocabs
                        for token in [x for x in kkk.lower().split('/')[-1].split('_')]:
                            vocabs[token] += 1
                        for nbr_nbr in vvv:
                            if isinstance(nbr_nbr, str):
                                for token in [y for y in tokenize(nbr_nbr.lower())]:
                                    vocabs[token] += 1
                                continue
                            elif isinstance(nbr_nbr, bool):
                                continue
                            elif isinstance(nbr_nbr, float):
                                # vocabs.update([y for y in tokenize(str(nbr_nbr).lower())])
                                continue
                            elif isinstance(nbr_nbr, dict):
                                nbr_nbr_k = list(nbr_nbr.keys())[0]
                                nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                                entities[nbr_nbr_k] += 1
                                selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:ENT_TYPE_HOP]
                                for ent_type in selected_types:
                                    entity_types[ent_type] += 1
                                selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                                for token in [y for x in selected_names for y in tokenize(x.lower())] + \
                                    [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
                                    vocabs[token] += 1
                            else:
                                raise RuntimeError('Unknown type: %s' % type(nbr_nbr))
                else:
                    raise RuntimeError('Unknown type: %s' % type(nbr))
    return (entities, entity_types, relations, vocabs)

def build_qa_vocab(qa):
    vocabs = defaultdict(int)
    for each in qa:
        for token in tokenize(each['qText'].lower()):
            vocabs[token] += 1
    return vocabs

def delex_query_topic_ent(query, topic_ent, ent_types):
    query = tokenize(query.lower())
    if topic_ent == '':
        return query, None

    ent_type_dict = {}
    for ent, type_ in ent_types:
        if ent not in ent_type_dict:
            ent_type_dict[ent] = type_
        else:
            if ent_type_dict[ent] == 'NP':
                ent_type_dict[ent] = type_

    ret = process.extract(topic_ent.replace('_', ' '), set(list(zip(*ent_types))[0]), scorer=fuzz.token_sort_ratio)
    if len(ret) == 0:
        return query, None

    # We prefer Non-NP entity mentions
    # e.g., we prefer `uk` than `people in the uk` when matching `united_kingdom`
    topic_men = None
    topic_score = None
    for token, score in ret:
        if ent_type_dict[token].lower() in config.topic_mention_types:
            topic_men = token
            topic_score = score
            break

    if topic_men is None:
        return query, None

    topic_ent_type = ent_type_dict[topic_men].lower()
    topic_tokens = tokenize(topic_men.lower())
    indices = [i for i, x in enumerate(query) if x == topic_tokens[0]]
    for i in indices:
        if query[i: i + len(topic_tokens)] == topic_tokens:
            start_idx = i
            end_idx = i + len(topic_tokens)
            break
    query_template = query[:start_idx] + [topic_ent_type] + query[end_idx:]
    return query_template, topic_men

def delex_query(query, ent_mens, mention_types):
    for men, type_ in ent_mens:
        type_ = type_.lower()
        if type_ in mention_types:
            men = tokenize(men.lower())
            indices = [i for i, x in enumerate(query) if x == men[0]]
            start_idx = None
            for i in indices:
                if query[i: i + len(men)] == men:
                    start_idx = i
                    end_idx = i + len(men)
                    break
            if start_idx is not None:
                query = query[:start_idx] + ['__{}__'.format(type_)] + query[end_idx:]
    return query

def build_data(qa, kb, entity2id, entityType2id, relation2id, vocab2id, pred_seed_ents=None):
    queries = []
    raw_queries = []
    query_mentions = []
    memories = []
    cand_labels = [] # Candidate answer labels (i.e., names)
    gold_ans_labels = [] # True gold answer labels
    gold_ans_inds = [] # The "gold" answer indices corresponding to the cand list
    for qid, each in enumerate(qa):
        freebase_key = each['freebaseKey'] if not pred_seed_ents else pred_seed_ents[qid]
        if isinstance(freebase_key, list):
            freebase_key = freebase_key[0] if len(freebase_key) > 0 else ''
        # Convert query to query template
        query, topic_men = delex_query_topic_ent(each['qText'], freebase_key, each['entities'])
        query2 = delex_query(query, each['entities'], config.delex_mention_types)
        q = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in query2]
        queries.append(q)
        raw_queries.append(query)

        query_mentions.append([(tokenize(x[0].lower()), x[1].lower()) for x in each['entities'] if topic_men != x[0]])
        gold_ans_labels.append(each['answers'])

        if not freebase_key in kb:
            gold_ans_inds.append([])
            memories.append([[]] * 8)
            cand_labels.append([])
            continue

        ans_cands = build_ans_cands(kb[freebase_key], entity2id, entityType2id, relation2id, vocab2id)
        memories.append(ans_cands[:-1])
        cand_labels.append(ans_cands[-1])
        if len(ans_cands[0]) == 0:
            gold_ans_inds.append([])
            continue

        norm_cand_labels = [normalize_answer(x) for x in ans_cands[-1]]
        tmp_cand_inds = []
        for a in each['answers']:
            a = normalize_answer(a)
            # Find all the candidiate answers which match the gold answer.
            inds = [i for i, j in zip(count(), norm_cand_labels) if j == a]
            tmp_cand_inds.extend(inds)
        # Note that tmp_cand_inds can be empty in which case
        # the question can *NOT* be answered by this KB entity.
        gold_ans_inds.append(tmp_cand_inds)
    return (queries, raw_queries, query_mentions, memories, cand_labels, gold_ans_inds, gold_ans_labels)

def build_vocab(data, freebase, used_fbkeys=None, min_freq=1):
    entities, entity_types, relations, kb_vocabs = build_kb_data(freebase, used_fbkeys)

    # Entity
    all_entities = set({ent for ent in entities if entities[ent] >= min_freq})
    entity2id = dict(zip(all_entities, range(len(config.RESERVED_ENTS), len(all_entities) + len(config.RESERVED_ENTS))))
    for ent, idx in config.RESERVED_ENTS.items():
        entity2id.update({ent: idx})

    # Entity type
    all_ent_types = set({ent_type for ent_type in entity_types if entity_types[ent_type] >= min_freq})
    all_ent_types.update(config.extra_ent_types)
    entityType2id = dict(zip(all_ent_types, range(len(config.RESERVED_ENT_TYPES), len(all_ent_types) + len(config.RESERVED_ENT_TYPES))))
    for ent_type, idx in config.RESERVED_ENT_TYPES.items():
        entityType2id.update({ent_type: idx})

    # Relation
    all_relations = set({rel for rel in relations if relations[rel] >= min_freq})
    all_relations.update(config.extra_rels)
    relation2id = dict(zip(all_relations, range(len(config.RESERVED_RELS), len(all_relations) + len(config.RESERVED_RELS))))
    for rel, idx in config.RESERVED_RELS.items():
        relation2id.update({rel: idx})

    # Vocab
    vocabs = build_qa_vocab(data)
    for token, count in kb_vocabs.items():
        vocabs[token] += count
    # sorted_vocabs = sorted(vocabs.items(), key=lambda d:d[1], reverse=True)
    all_tokens = set({token for token in vocabs if vocabs[token] >= min_freq})
    all_tokens.update(config.extra_vocab_tokens)
    vocab2id = dict(zip(all_tokens, range(len(config.RESERVED_TOKENS), len(all_tokens) + len(config.RESERVED_TOKENS))))
    for token, idx in config.RESERVED_TOKENS.items():
        vocab2id.update({token: idx})

    print('Num of entities: %s' % len(entity2id))
    print('Num of entity_types: %s' % len(entityType2id))
    print('Num of relations: %s' % len(relation2id))
    print('Num of vocabs: %s' % len(vocab2id))
    return entity2id, entityType2id, relation2id, vocab2id

def build_ans_cands(graph, entity2id, entityType2id, relation2id, vocab2id):
    cand_ans_bows = [] # bow of answer entity
    cand_ans_entities = [] # answer entity
    cand_ans_types = [] # type of answer entity
    cand_ans_type_bows = [] # bow of answer entity type
    cand_ans_paths = [] # relation path from topic entity to answer entity
    cand_ans_path_bows = []
    cand_ans_ctx = [] # context (i.e., 1-hop entity bows and relation bows) connects to the answer path
    cand_ans_topic_key_type = [] # topic key entity type
    cand_labels = [] # candidiate answers

    selected_types = (graph['notable_types'] + graph['type'])[:ENT_TYPE_HOP]
    topic_key_ent_type_bows = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for y in selected_types for x in y.lower().split('/')[-1].split('_')]
    topic_key_ent_type = [entityType2id[x] if x in entityType2id else config.RESERVED_ENT_TYPES['UNK'] for x in selected_types]

    # We only consider the alias relations of topic entityies
    for each in graph['alias']:
        cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
        ent_bow = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for y in tokenize(each.lower())]
        cand_ans_bows.append(ent_bow)
        cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
        cand_ans_types.append([])
        cand_ans_type_bows.append([])
        cand_ans_paths.append([relation2id['alias'] if 'alias' in relation2id else config.RESERVED_RELS['UNK']])
        cand_ans_path_bows.append([vocab2id['alias']])
        # We do not count the topic_entity as context since it is trivial
        cand_ans_ctx.append([[], []])
        cand_labels.append(each)

    if len(cand_labels) == 0 and (not 'neighbors' in graph or len(graph['neighbors']) == 0):
        return ([], [], [], [], [], [], [], [], [])

    for k, v in graph['neighbors'].items():
        if if_filterout(k):
            continue
        k_bow = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in k.lower().split('/')[-1].split('_')]
        for nbr in v:
            if isinstance(nbr, str):
                cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                ent_bow = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for y in tokenize(nbr.lower())]
                cand_ans_bows.append(ent_bow)
                cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                cand_ans_types.append([])
                cand_ans_type_bows.append([])
                cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK']])
                cand_ans_path_bows.append(k_bow)
                cand_ans_ctx.append([[], []])
                cand_labels.append(nbr)
                continue
            elif isinstance(nbr, bool):
                cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                cand_ans_bows.append([vocab2id['true' if nbr else 'false']])
                cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                cand_ans_types.append([entityType2id['bool']])
                cand_ans_type_bows.append([vocab2id['bool']])
                cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK']])
                cand_ans_path_bows.append(k_bow)
                cand_ans_ctx.append([[], []])
                cand_labels.append('true' if nbr else 'false')
                continue
            elif isinstance(nbr, float):
                cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                cand_ans_bows.append([vocab2id[str(nbr)] if str(nbr) in vocab2id else config.RESERVED_TOKENS['UNK']])
                cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                cand_ans_types.append([entityType2id['num']])
                cand_ans_type_bows.append([vocab2id['num']])
                cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK']])
                cand_ans_path_bows.append(k_bow)
                cand_ans_ctx.append([[], []])
                cand_labels.append(str(nbr))
                continue
            elif isinstance(nbr, dict):
                nbr_k = list(nbr.keys())[0]
                nbr_v = nbr[nbr_k]
                selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                is_dummy = True
                if not IGNORE_DUMMY or len(selected_names) > 0: # Otherwise, it is an intermediate (dummpy) node
                    cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                    nbr_k_bow = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for x in selected_names for y in tokenize(x.lower())]
                    cand_ans_bows.append(nbr_k_bow)
                    cand_ans_entities.append(entity2id[nbr_k] if nbr_k in entity2id else config.RESERVED_ENTS['UNK'])
                    selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:ENT_TYPE_HOP]
                    cand_ans_types.append([entityType2id[x] if x in entityType2id else config.RESERVED_ENT_TYPES['UNK'] for x in selected_types])
                    cand_ans_type_bows.append([vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for y in selected_types for x in y.lower().split('/')[-1].split('_')])
                    cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK']])
                    cand_ans_path_bows.append(k_bow)
                    cand_labels.append(selected_names[0] if len(selected_names) > 0 else 'UNK')
                    is_dummy = False

                if not 'neighbors' in nbr_v:
                    if not is_dummy:
                        cand_ans_ctx.append([[], []])
                    continue

                rels = []
                labels = []
                all_ctx = [set(), set()]
                for kk, vv in nbr_v['neighbors'].items(): # 2nd hop
                    if if_filterout(kk):
                        continue
                    kk_bow = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in kk.lower().split('/')[-1].split('_')]
                    all_ctx[1].add(kk)
                    for nbr_nbr in vv:
                        if isinstance(nbr_nbr, str):
                            cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                            ent_bow = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for y in tokenize(nbr_nbr.lower())]
                            cand_ans_bows.append(ent_bow)
                            cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                            cand_ans_types.append([])
                            cand_ans_type_bows.append([])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else config.RESERVED_RELS['UNK']])
                            cand_ans_path_bows.append(kk_bow + k_bow)
                            labels.append(nbr_nbr)
                            all_ctx[0].add(nbr_nbr)
                            rels.append(kk)
                            continue
                        elif isinstance(nbr_nbr, bool):
                            cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                            cand_ans_bows.append([vocab2id['true' if nbr_nbr else 'false']])
                            cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                            cand_ans_types.append([entityType2id['bool']])
                            cand_ans_type_bows.append([vocab2id['bool']])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else config.RESERVED_RELS['UNK']])
                            cand_ans_path_bows.append(kk_bow + k_bow)
                            labels.append('true' if nbr_nbr else 'false')
                            all_ctx[0].add('true' if nbr_nbr else 'false')
                            rels.append(kk)
                            continue
                        elif isinstance(nbr_nbr, float):
                            cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                            cand_ans_bows.append([vocab2id[str(nbr_nbr)] if str(nbr_nbr) in vocab2id else config.RESERVED_TOKENS['UNK']])
                            cand_ans_entities.append(config.RESERVED_ENTS['PAD'])
                            cand_ans_types.append([entityType2id['num']])
                            cand_ans_type_bows.append([vocab2id['num']])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else config.RESERVED_RELS['UNK']])
                            cand_ans_path_bows.append(kk_bow + k_bow)
                            labels.append(str(nbr_nbr))
                            all_ctx[0].add(str(nbr_nbr))
                            rels.append(kk)
                            continue
                        elif isinstance(nbr_nbr, dict):
                            nbr_nbr_k = list(nbr_nbr.keys())[0]
                            nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                            selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                            if not IGNORE_DUMMY or len(selected_names) > 0:
                                cand_ans_topic_key_type.append([topic_key_ent_type_bows, topic_key_ent_type])
                                ent_bow = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for x in selected_names for y in tokenize(x.lower())]
                                cand_ans_bows.append(ent_bow)
                                cand_ans_entities.append(entity2id[nbr_nbr_k] if nbr_nbr_k in entity2id else config.RESERVED_ENTS['UNK'])
                                selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:ENT_TYPE_HOP]
                                cand_ans_types.append([entityType2id[x] if x in entityType2id else config.RESERVED_ENT_TYPES['UNK'] for x in selected_types])
                                cand_ans_type_bows.append([vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for y in selected_types for x in y.lower().split('/')[-1].split('_')])
                                cand_ans_paths.append([relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else config.RESERVED_RELS['UNK']])
                                cand_ans_path_bows.append(kk_bow + k_bow)
                                labels.append(selected_names[0] if len(selected_names) > 0 else 'UNK')
                                if len(selected_names) > 0:
                                    all_ctx[0].add(selected_names[0])
                                rels.append(kk)
                        else:
                            raise RuntimeError('Unknown type: %s' % type(nbr_nbr))

                assert len(labels) == len(rels)
                if not is_dummy:
                    ctx_ent_bow = [tokenize(x.lower()) for x in all_ctx[0]]
                    # ctx_rel_bow = list(set([vocab2id[y] for x in all_ctx[1] for y in x.lower().split('/')[-1].split('_') if y in vocab2id]))
                    ctx_rel_bow = []
                    cand_ans_ctx.append([ctx_ent_bow, ctx_rel_bow])
                for i in range(len(labels)):
                    tmp_ent_names = all_ctx[0] - set([labels[i]])
                    # tmp_rel_names = all_ctx[1] - set([rels[i]])
                    ctx_ent_bow = [tokenize(x.lower()) for x in tmp_ent_names]
                    # ctx_rel_bow = list(set([vocab2id[y] for x in tmp_rel_names for y in x.lower().split('/')[-1].split('_') if y in vocab2id]))
                    ctx_rel_bow = []
                    cand_ans_ctx.append([ctx_ent_bow, ctx_rel_bow])
                cand_labels.extend(labels)
            else:
                raise RuntimeError('Unknown type: %s' % type(nbr))

    assert len(cand_ans_bows) == len(cand_ans_entities) == len(cand_ans_types) == len(cand_ans_type_bows) == len(cand_ans_paths) \
            == len(cand_ans_ctx) == len(cand_labels) == len(cand_ans_topic_key_type) == len(cand_ans_path_bows)
    return (cand_ans_bows, cand_ans_entities, cand_ans_type_bows, cand_ans_types, cand_ans_path_bows, cand_ans_paths, cand_ans_ctx, cand_ans_topic_key_type, cand_labels)


# Build seed entity candidates for topic entity classification
def build_seed_ent_data(qa, kb, entity2id, entityType2id, relation2id, vocab2id, topn, dtype):
    queries = []
    seed_ent_features = []
    seed_ent_labels = []
    seed_ent_inds = []
    for each in qa:
        query = tokenize(each['qText'].lower())
        q = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in query]
        queries.append(q)
        tmp_features = []
        tmp_labels = []
        tmp_inds = []
        for i, freebase_key in enumerate(each['freebaseKeyCands'][:topn]):
            tmp_labels.append(freebase_key)
            if freebase_key == each['freebaseKey']:
                tmp_inds.append(i)

            if freebase_key in kb:
                features = build_seed_entity_feature(freebase_key, kb[freebase_key], entity2id, entityType2id, relation2id, vocab2id)
                tmp_features.append(features)
            else:
                tmp_features.append([[]] * 5)

        if dtype == 'test':
            if len(tmp_inds) == 0: # No answer
                tmp_inds.append(-1)
        else:
            assert len(tmp_labels) == topn

        assert len(tmp_inds) == 1
        seed_ent_features.append(list(zip(*tmp_features)))
        seed_ent_labels.append(tmp_labels)
        seed_ent_inds.append(tmp_inds)
    return (queries, seed_ent_features, seed_ent_labels, seed_ent_inds)

def build_seed_entity_feature(seed_ent, graph, entity2id, entityType2id, relation2id, vocab2id):
    # candidate seed entity features:
    # entity name
    # entity type
    # entity neighboring relations
    selected_names = (graph['name'] + graph['alias'])[:1]
    seed_ent_name = [vocab2id[y] if y in vocab2id else config.RESERVED_TOKENS['UNK'] for x in selected_names for y in tokenize(x.lower())]
    selected_types = (graph['notable_types'] + graph['type'])[:ENT_TYPE_HOP]
    seed_ent_type_name = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for y in selected_types for x in y.lower().split('/')[-1].split('_')]
    seed_ent_type = [entityType2id[x] if x in entityType2id else config.RESERVED_ENT_TYPES['UNK'] for x in selected_types]
    seed_rel_names = []
    seed_rels = []

    for k in graph['neighbors']:
        if if_filterout(k):
            continue
        k_bow = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in k.lower().split('/')[-1].split('_')]
        seed_rel_names.append(k_bow)
        seed_rels.append(relation2id[k] if k in relation2id else config.RESERVED_RELS['UNK'])
    return (seed_ent_name, seed_ent_type_name, seed_ent_type, seed_rel_names, seed_rels)
