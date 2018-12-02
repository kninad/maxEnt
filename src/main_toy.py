from __future__ import division
import pickle
import numpy as np
import itertools
from matplotlib import pyplot as plt

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
k_val = 4

data_array = load_disease_data(filePath)
feats = ExtractFeatures(data_array, entropy_estimator, k_val)
feats.compute_topK_feats()
feats.partition_features()

# featFile = '../out/feats_obj_red.pk'
# with open(featFile, "rb") as rfile:
#     feats = pickle.load(rfile)

opt = Optimizer(feats) 
soln_opt = opt.solver_optimize()


#### PLOTS #### 

num_feats = data_array.shape[1]
all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))
mxt_prob = np.zeros(num_feats)
for vec in all_perms:
    for j in range(num_feats):
        if sum(vec) == j:
            mxt_prob[j] += opt.prob_dist(vec)
            break


emp_prob = np.zeros(num_feats)
for vec in data_array:
    for j in range(num_feats):
        if sum(vec) == j:
            emp_prob[j] += 1
            break

emp_prob = emp_prob/data_array.shape[0]


xvec = [i+1 for i in range(num_feats)]
x_ticks = np.arange(0, num_feats+2, 1.0)
plot_lims = [0,  num_feats+2, -0.1, 1.0]

# Both on same plot
plt.figure()
plt.plot(xvec, emp_prob, 'ro')  # empirical
plt.plot(xvec, mxt_prob, 'bo')  # maxent
plt.xticks(x_ticks)
plt.axis(plot_lims)

# plt.savefig('../out/brute_p3_1yr_' + str(k_val) + '.png')
plt.savefig('../out/montecarlo_p3_1yr_' + str(k_val) + '.png')



# Difference Plot
xvec = [i+1 for i in range(num_feats)]
x_ticks = np.arange(0, num_feats+2, 1.0)
plot_lims = [0,  num_feats+2, -0.5, 0.5]

diff_vec = emp_prob - mxt_prob
plt.figure()
plt.plot(xvec, diff_vec, 'go')
plt.xticks(x_ticks)
plt.axis(plot_lims)

# plt.savefig('../out/brute_p4_diff_1yr_' + str(k_val) + '.png')
plt.savefig('../out/montecarlo_p4_diff_1yr_' + str(k_val) + '.png')

