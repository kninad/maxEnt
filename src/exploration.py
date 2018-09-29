from __future__ import division
import pickle
import numpy as np
import itertools
from matplotlib import pyplot as plt

from utils import load_data_1yr, load_data_merge
from extractFeatures import ExtractFeatures
# from optimizer import Solver
from partitioned_optimizer import Solver

filePath = '../data/Age50_DataExtract.csv'
entropy_est = 'JAMES-STEIN'
k_val = 10

'''
Use load_data_small since we only need to consider the first year data.
load_data_small loads only the first 9 feature column corresponding to the
first year prevalence of the disease in the individual.
'''
# data_array = load_data_merge(filePath)
data_array = load_data_1yr(filePath)

feats = ExtractFeatures(data_array, entropy_est, k_val)
feats.compute_topK_feats()
feats.partition_features()

opt = Solver(feats)
soln_opt = opt.solver_optimize()

m1, m2 = opt.compare_marginals()
c1, c2 = opt.compare_constraints()
print m1, m2
print c1, c2




## ADD CODE from the ipython terminal