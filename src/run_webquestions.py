'''
Created on Oct, 2017

@author: hugo

'''
import argparse
from core.build_data.webquestions import *

parser = argparse.ArgumentParser()
parser.add_argument('-fb', '--freebase_path', required=True, type=str, help='path to the freebase data')
parser.add_argument('-mid2key', '--mid2key_path', required=True, type=str, help='path to the freebase data')
parser.add_argument('-data_dir', '--data_dir', required=True, type=str, help='path to the data dir')
parser.add_argument('-out_dir', '--out_dir', type=str, required=True, help='path to the output dir')
args = parser.parse_args()

main(args.freebase_path, args.mid2key_path, args.data_dir, args.out_dir)
# get_used_fbkeys(args.data_dir, args.out_dir)
# get_all_fbkeys(args.data_dir, args.out_dir)
