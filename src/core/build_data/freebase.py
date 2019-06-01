'''
Created on Sep, 2017

@author: hugo

'''
import os

from ..utils.utils import *


def fetch_meta(path):
    try:
        data = load_gzip_json(path)
    except:
        return {}
    content = {}
    properties = data['property']
    if '/type/object/name' in properties:
        content['name'] = [x['value'] for x in properties['/type/object/name']['values']]
    else:
        content['name'] = []
    if '/common/topic/alias' in properties:
        content['alias'] = [x['value'] for x in properties['/common/topic/alias']['values']]
    else:
        content['alias'] = []
    if '/common/topic/notable_types' in properties:
        content['notable_types'] = [x['id'] for x in properties['/common/topic/notable_types']['values']]
    else:
        content['notable_types'] = []
    if '/type/object/type' in properties:
        content['type'] = [x['id'] for x in properties['/type/object/type']['values']]
    else:
        content['type'] = []
    return content

def fetch(data, data_dir):
    if not 'id' in data:
        return data['value']
    mid = data['id']
    # meta data might not be in the subgraph, get it from target files
    meta = fetch_meta(os.path.join(data_dir, '{}.json.gz'.format(mid.strip('/').replace('/', '.'))))
    if meta == {}:
        if not 'property' in data:
            if 'text' in data:
                return data['text']
            else:
                import pdb;pdb.set_trace()
        properties = data['property']
        if '/type/object/name' in properties:
            meta['name'] = [x['value'] for x in properties['/type/object/name']['values']]
        else:
            meta['name'] = []
        if '/common/topic/alias' in properties:
            meta['alias'] = [x['value'] for x in properties['/common/topic/alias']['values']]
        else:
            meta['alias'] = []
        if '/common/topic/notable_types' in properties:
            meta['notable_types'] = [x['id'] for x in properties['/common/topic/notable_types']['values']]
        else:
            meta['notable_types'] = []
        if '/type/object/type' in properties:
            meta['type'] = [x['id'] for x in properties['/type/object/type']['values']]
        else:
            meta['type'] = []
    graph = {mid: meta}
    if not 'property' in data: # we stop at the 2nd hop
        return graph
    properties = data['property']
    neighbors = {}
    for k, v in properties.items():
        if k.startswith('/common') or k.startswith('/type') \
            or k.startswith('/freebase') or k.startswith('/user') \
            or k.startswith('/imdb'):
            continue
        if len(v['values']) > 0:
            neighbors[k] = []
            for nbr in v['values']:
                nbr_graph = fetch(nbr, data_dir)
                neighbors[k].append(nbr_graph)
    graph[mid]['neighbors'] = neighbors
    return graph
