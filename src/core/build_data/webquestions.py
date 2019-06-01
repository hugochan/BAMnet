'''
Created on Sep, 2017

@author: hugo

'''
import os
# import re
import argparse
from nltk.parse.stanford import StanfordDependencyParser

from ..utils.utils import *
from ..utils.freebase_utils import if_filterout
from ..utils.generic_utils import *


def get_used_fbkeys(data_dir, out_dir):
    # Fetch freebase keys used in training and validation sets.
    fbkeys = set()
    split = ['factoid_webqa/train.json', 'factoid_webqa/valid.json']
    files = [os.path.join(data_dir, x) for x in split]
    for f in files:
        data = load_json(f)
        for qa in data:
            fbkeys.add(qa['freebaseKey'])
    dump_json(list(fbkeys), os.path.join(out_dir, 'fbkeys_train_valid.json'), indent=1)

def get_all_fbkeys(data_dir, out_dir):
    # Fetch all freebase keys possibily useful to answer questions.
    fbkeys = set()
    split = ['factoid_webqa/train.json', 'factoid_webqa/valid.json', 'factoid_webqa/test.json']
    files = [os.path.join(data_dir, x) for x in split]
    for f in files:
        data = load_json(f)
        for qa in data:
            fbkeys.add(qa['freebaseKey'])

    retrieved_test_path = os.path.join(data_dir, 'factoid_webqa/webquestions.examples.test.retrieved.json')
    if os.path.exists(retrieved_test_path):
        data = load_json(retrieved_test_path)
        for qa in data:
            if not 'retrievedList' in qa:
                continue
            for x in qa['retrievedList'].split():
                fbkeys.add(x.split(':')[0])
    dump_json(list(fbkeys), os.path.join(out_dir, 'fbkeys_train_valid_test_retrieved.json'), indent=1)

def main(fb_path, mid2key_path, data_dir, out_dir):
    HAS_DEP = False
    if HAS_DEP:
        dep_parser = StanfordDependencyParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz") # Set CLASSPATH and STANFORD_MODELS environment variables beforehand
    kb = load_ndjson(fb_path, return_type='dict')
    mid2key = load_json(mid2key_path)
    all_split_questions = []
    split = ['factoid_webqa/train.json', 'factoid_webqa/valid.json', 'factoid_webqa/test.json']
    files = [os.path.join(data_dir, x) for x in split]
    missing_mid2key = []

    for f in files:
        data_type = os.path.basename(f).split('.')[0]
        num_unanswerable = 0
        all_questions = []
        data = load_json(f)
        for q in data:
            questions = {}
            questions['answers'] = q['answers']
            questions['entities'] = q['entities']
            questions['qText'] = q['qText']
            questions['qId'] = q['qId']
            questions['freebaseKey'] = q['freebaseKey']
            questions['freebaseKeyCands'] = [q['freebaseKey']]
            for x in q['freebaseMids']:
                if x['mid'] in mid2key:
                    fbkey = mid2key[x['mid']]
                    if fbkey != q['freebaseKey']:
                        questions['freebaseKeyCands'].append(fbkey)
                else:
                    missing_mid2key.append(x['mid'])

            qtext = tokenize(q['qText'])
            if HAS_DEP:
                qw = list(set(qtext).intersection(question_word_list))
                question_word = qw[0] if len(qw) > 0 else ''
                topic_ent = q['freebaseKey']
                dep_path = extract_dep_feature(dep_parser, ' '.join(qtext), topic_ent, question_word)
            else:
                dep_path = []
            questions['dep_path'] = dep_path
            all_questions.append(questions)

            if not q['freebaseKey'] in kb:
                num_unanswerable += 1
                continue
            cand_ans = fetch_ans_cands(kb[q['freebaseKey']])
            norm_cand_ans = set([normalize_answer(x) for x in cand_ans])
            norm_gold_ans = [normalize_answer(x) for x in q['answers']]
            # Check if we can find the gold answer from the candidiate answers.
            if len(norm_cand_ans.intersection(norm_gold_ans)) == 0:
                num_unanswerable += 1
                continue
        all_split_questions.append(all_questions)
        print('{} set: Num of unanswerable questions: {}'.format(data_type, num_unanswerable))

    for i, each in enumerate(all_split_questions):
        dump_ndjson(each, os.path.join(out_dir, split[i].split('/')[-1]))

def fetch_ans_cands(graph):
    cand_ans = set() # candidiate answers
    # We only consider the alias relations of topic entityies
    cand_ans.update(graph['alias'])
    for k, v in graph['neighbors'].items():
        if if_filterout(k):
            continue
        for nbr in v:
            if isinstance(nbr, str):
                cand_ans.add(nbr)
                continue
            elif isinstance(nbr, bool):
                cand_ans.add('true' if nbr else 'false')
                continue
            elif isinstance(nbr, float):
                cand_ans.add(str(nbr))
                continue
            elif isinstance(nbr, dict):
                nbr_k = list(nbr.keys())[0]
                nbr_v = nbr[nbr_k]
                selected_names = nbr_v['name'] if 'name' in nbr_v and len(nbr_v['name']) > 0 else (nbr_v['alias'][:1] if 'alias' in nbr_v else [])
                cand_ans.add(selected_names[0] if len(selected_names) > 0 else 'UNK')
                if not 'neighbors' in nbr_v:
                    continue
                for kk, vv in nbr_v['neighbors'].items(): # 2nd hop
                    if if_filterout(kk):
                        continue
                    for nbr_nbr in vv:
                        if isinstance(nbr_nbr, str):
                            cand_ans.add(nbr_nbr)
                            continue
                        elif isinstance(nbr_nbr, bool):
                            cand_ans.add('true' if nbr_nbr else 'false')
                            continue
                        elif isinstance(nbr_nbr, float):
                            cand_ans.add(str(nbr_nbr))
                            continue
                        elif isinstance(nbr_nbr, dict):
                            nbr_nbr_k = list(nbr_nbr.keys())[0]
                            nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                            selected_names = nbr_nbr_v['name'] if 'name' in nbr_nbr_v and len(nbr_nbr_v['name']) > 0 else (nbr_nbr_v['alias'][:1] if 'alias' in nbr_nbr_v else [])
                            cand_ans.add(selected_names[0] if len(selected_names) > 0 else 'UNK')
                        else:
                            raise RuntimeError('Unknown type: %s' % type(nbr_nbr))
            else:
                raise RuntimeError('Unknown type: %s' % type(nbr))
    return list(cand_ans)
