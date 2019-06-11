'''
Created on Oct, 2017

@author: hugo

'''
import os

from . import utils as build_utils
from ..utils.utils import *
from .build_data import build_vocab, build_data


def build(dpath, version=None, out_dir=None):
    if not build_utils.built(dpath, version_string=version):
        raise RuntimeError("Please build/preprocess the data by running the build_all_data.py script!")
