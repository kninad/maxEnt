from __future__ import division
import numpy as np

from utils import load_data
from extractFeatures import ExtractFeatures
from optimizer import Optimizer

filePath = '../data/Age50_DataExtract.csv'
entropy_est = 'JAMES-STEIN'
k_val = 5

data_array = load_data(filePath)

feats = ExtractFeatures(data_array, entropy_est, k_val)
feats.compute_topK_feats()

opt = Optimizer(feats)
soln_opt = opt.solver_optimize()
# soln_opt = opt.solver_optimize()

