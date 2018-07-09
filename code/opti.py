from __future__ import division
import operator
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as spmin_LBFGSB
from pyitlib import discrete_random_variable as drv

from data_loading import load_data
from feature_extract import discrete_normalized_Lmeasure, topK_feats

filePath = '../data/Age50_DataExtract.csv'
EST = 'JAMES-STEIN'
K = 5
data_arr = load_data(filePath)
N = data_arr.shape[0]

# theta vector len = num_features (for marginals) + K

# topK_pairs == vd
def compute_hr(thetas, rvec, topK_pairs_dict):    
    constraint_sum = 0.0
    
    # Can do this before too
    topK_list = [(k,v) for k,v in topK_pairs_dict.items()]
    
    # marginals    
    num_feats = len(rvec)
    assert len(rvec) == (len(thetas) - len(topK_list))
    # CHECKING WITH 1 since BINARY FEATURES
    for i in range(num_feats):
        indicator = 1 if rvec[i] == 1 else 0
        constraint_sum += thetas[i] * indicator
    
    # top K constraints
    for j, tup in enumerate(topK_list):
        key = tup[0]
        val = tup[1]
        condition = rvec[key[0]] == val[0] and rvec[key[1]] == val[1]
        indicator = 1 if condition else 0
        constraint_sum += thetas[j + num_feats] * indicator
        # SINCE THETAS IS CONTIGUOUS VECTOR
        # CAN ALSO BREAK UP THETAS into MARGINAL AND topK CONSTRATINTS    

    return constraint_sum

# Objective for the optimization problem
# It is a function of thetas
def f_objective(thetas, data_arr, topK_pairs_dict):
    obj_sum = 0.0
    N = data_arr.shape[0]
    
    # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
    for i in range(N):
        rvec = data_arr[i,:]
        inner_constraint_sum = compute_hr(thetas, rvec, topK_pairs_dict)
        obj_sum += inner_constraint_sum

    return -1 * obj_sum # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION


def optimizer(func_obj, data_arr, topK_pairs):

    total_len = data_arr.shape[1] + len(topK_pairs)
    init_thetas = np.random.rand(total_len,)

    optimThetas = spmin_LBFGSB(func_obj,                                
                                    x0=init_thetas, fprime=None,
                                    args=(data_arr, topK_pairs),
                                    approx_grad=True,                                    
                                    disp=True)

    return optimThetas



lms, _, _ = discrete_normalized_Lmeasure(data_arr, EST)
vd = topK_feats(lms, data_arr, K)

opt = optimizer(f_objective, data_arr, vd)


# ALL THIS ALSO SHOULD BE IN A CLASS