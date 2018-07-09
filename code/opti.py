from __future__ import division
import operator
import numpy as np
from pyitlib import discrete_random_variable as drv

from data_loading import load_data
from feature_extract import discrete_normalized_Lmeasure, topK_feats

filePath = '../data/Age50_DataExtract.csv'
EST = 'JAMES-STEIN'
K = 5
data_arr = load_data(filePath)
N = data_arr.shape[0]

lms, _, _ = discrete_normalized_Lmeasure(data_arr, EST)
vd = topK_feats(lms, data_arr, K)

# theta vector len = num_features (for marginals) + K

# topK_pairs == vd
def compute_hr(thetas, rvec, topK_pairs):    
    constraint_sum = 0.0
    
    # Can do this before too
    topK_list = [(k,v) for k,v in topK_pairs.items()]
    
    # marginals    
    num_feats = len(rvec)
    assert len(rvec) == (len(thetas) - len(topK_list))
    # CHECKING WITH 1 since BINARY FEATURES
    for i in range(num_feats):
        indicator = 1 if rvec[i] == 1 else 0
        constraint_sum += thetas[i] * indicator
    
    # top K constraints

    

    return