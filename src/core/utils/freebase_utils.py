'''
Created on Oct, 2017

@author: hugo

'''
from fuzzywuzzy import fuzz, process


def if_filterout(s):
    if s.endswith('has_sentences') or \
        s.endswith('exceptions') or s.endswith('sww_base/source') or \
        s.endswith('kwtopic/assessment'):
        return True
    else:
        return False

def query_kb(kb, ent_name, fuzz_threshold=90):
    results = []
    for k, v in kb.items():
        ret = process.extractOne(ent_name, v['name'] + v['alias'], scorer=fuzz.token_sort_ratio)
        if ret[1] > fuzz_threshold:
            results.append((k, ret[0], ret[1]))
    results = sorted(results, key=lambda d:d[-1], reverse=True)
    return list(zip(*results))[0] if len(results) > 0 else []
