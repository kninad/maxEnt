from __future__ import division
import pickle

from utils import load_data, load_data_small
from extractFeatures import ExtractFeatures
# from optimizer import Solver
from partitioned_optimizer import Solver

filePath = '../data/Age50_DataExtract.csv'
entropy_est = 'JAMES-STEIN'
k_val = 5

# data_array = load_data(filePath)
data_array = load_data_small(filePath)

feats = ExtractFeatures(data_array, entropy_est, k_val)
feats.compute_topK_feats()
feats.partition_features()




opt = Solver(feats)
soln_opt = opt.solver_optimize()

# saveFilePath = './feat20_solun.pk'

# with open (saveFilePath, 'wb') as wfile:
#     pickle.dump(soln_opt, wfile, -1)


for i in range(data_array.shape[1]):
    print(opt.compare_marginals(i))