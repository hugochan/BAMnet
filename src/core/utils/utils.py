'''
Created on Sep, 2017

@author: hugo

'''
import os
import re
import yaml
import gzip
import json
import string
import numpy as np
from nltk.tokenize import wordpunct_tokenize#, word_tokenize


# tokenize = lambda s: word_tokenize(re.sub(r'[%s]' % punc_wo_dot, ' ', re.sub(r'(?<!\d)[%s](?!\d)' % string.punctuation, ' ', s)))
tokenize = lambda s: wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', s))

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

def read_lines(path_to_file):
    data = []
    try:
        with open(path_to_file, 'r') as f:
            for line in f:
                tmp = [float(x) for x in line.strip().split()]
                data.append(tmp)
    except Exception as e:
        raise e

    return data

def dump_ndarray(data, path_to_file):
    try:
        with open(path_to_file, 'wb') as f:
            np.save(f, data)
    except Exception as e:
        raise e

def load_ndarray(path_to_file):
    try:
        with open(path_to_file, 'rb') as f:
            data = np.load(f)
    except Exception as e:
        raise e

    return data

def dump_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for each in data:
                f.write(json.dumps(each) + '\n')
    except Exception as e:
        raise e

def load_ndjson(file, return_type='array'):
    if return_type == 'array':
        return load_ndjson_to_array(file)
    elif return_type == 'dict':
        return load_ndjson_to_dict(file)
    else:
        raise RuntimeError('Unknown return_type: %s' % return_type)

def load_ndjson_to_array(file):
    data = []
    try:
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def load_ndjson_to_dict(file):
    data = {}
    try:
        with open(file, 'r') as f:
            for line in f:
                data.update(json.loads(line.strip()))
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_dict_ndjson(data, file):
    try:
        with open(file, 'w') as f:
            for k, v in data.items():
                line = json.dumps([k, v]) + '\n'
                f.write(line)
    except Exception as e:
        raise e

def load_gzip_json(file):
    try:
        with gzip.open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def get_all_files(dir, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(dir) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(dir, filename) for filename in os.listdir(dir) if os.path.isfile(os.path.join(dir, filename)) and not filename.startswith('.')]

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
