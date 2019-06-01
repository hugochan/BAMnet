'''
Created on Oct, 2017

@author: hugo

'''
import argparse
import os
import json

from core.build_data.freebase import *
from core.utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
parser.add_argument('-fbkeys', '--freebase_keys', required=True, type=str, help='path to the freebase key file')
parser.add_argument('-out_dir', '--out_dir', type=str, required=True, help='path to the output dir')
args = parser.parse_args()

ids = load_json(args.freebase_keys)
total = len(ids)
print('Fetching {} entities and their 2-hop neighbors.'.format(total))
print_bar_len = 50
cnt = 0
missing_ids = set()
with open(os.path.join(args.out_dir, 'freebase.json'), 'a') as out_f:
    for id_ in ids:
        try:
            data = load_gzip_json(os.path.join(args.data_dir, '{}.json.gz'.format(id_)))
        except:
            missing_ids.add(id_)
            continue
        graph = fetch(data, args.data_dir)
        graph2 = {id_: list(graph.values())[0]}
        graph2[id_]['id'] = list(graph.keys())[0]
        line = json.dumps(graph2) + '\n'
        out_f.write(line)
        cnt += 1
        if cnt % int(total / print_bar_len) == 0:
            printProgressBar(cnt, total, prefix='Progress:', suffix='Complete', length=print_bar_len)
    printProgressBar(cnt, total, prefix='Progress:', suffix='Complete', length=print_bar_len)

print('Missed %s mids' % len(missing_ids))
dump_json(list(missing_ids), os.path.join(args.out_dir, 'missing_fbids.json'))
