import operator
import numpy as np

from pyitlib import discrete_random_variable as drv
from data_loading import load_data

filePath = './data/Age50_DataExtract.csv'
est = 'JAMES-STEIN'
K = 5
data_arr = load_data(filePath)
num_rand = data_arr.shape[1]

def discrete_Wxy(hx, hy):
    return min(hx,hy)

def discrete_Lmeasure(data_arr):
    
    indi_entropies = drv.entropy(data_arr.T, estimator=EST)

    L_measures = {}     # Dictionary storing the pairwise L-measures
    I_mutinfo = {}      # Dictionary storing the pairwise mutual information
    num_rand = data_arr.shape[1]
    assert num_rand == len(indi_entropies)

    for i in range(num_rand):
        for j in range(i+1, num_rand):
            key = (i+1, j+1)    # since 0-indexed
            h_i = indi_entropies[i]
            h_j = indi_entropies[j]
            # Potential error: I_ij may come out negative depending on the estiamtor
            I_ij = drv.information_mutual(data_arr.T[i], data_arr.T[j], estimator=EST)
            I_ij = np.abs(I_ij)     # TEMP FIX
            W_ij = min(h_i, h_j)
            inner_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
            L_measures[key] = np.sqrt(1 - np.exp(inner_term))
            I_mutinfo[key] = I_ij

    return L_measures, I_mutinfo




def top_k(Lmeasure_dict, k):
    sorted_list = sorted(Lmeasure_dict.items(), key=operator.itemgetter(1), reverse=True)
    # each entry is a tuple of (key,value)

    topK_keys = [item[0] for item in sorted_list[:k]]

    return topK_keys

lms, mutinfos = discrete_Lmeasure(data_arr)


