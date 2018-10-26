from __future__ import division
import pickle
import numpy as np
import itertools
#from matplotlib import pyplot as plt

import sys
#path_to_codebase = '/mnt/Study/umass/sem3/maxEnt/src/codebase/'
path_to_codebase = './codebase/'
sys.path.insert(0, path_to_codebase)
from codebase.utils import load_disease_data
from codebase.extract_features import ExtractFeatures
from codebase.optimizer import Optimizer

filePath = '../data/Age50_DataExtract_fy.csv'
# filePath = '../data/2010-2014-fy.csv'
# filePath = '../data/test1-fy.csv'

entropy_estimator = 'JAMES-STEIN'
k_val = 5

data_array = load_disease_data(filePath)
feats = ExtractFeatures(data_array, entropy_estimator, k_val)
feats.compute_topK_feats()
feats.partition_features()

# featFile = '../out/feats_obj_red.pk'
# with open(featFile, "rb") as rfile:
#     feats = pickle.load(rfile)

opt = Optimizer(feats) 
soln_opt = opt.solver_optimize()


