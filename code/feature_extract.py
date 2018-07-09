from __future__ import division
import operator
import numpy as np
from collections import defaultdict

from pyitlib import discrete_random_variable as drv
from data_loading import load_data

filePath = '../data/Age50_DataExtract.csv'
EST = 'JAMES-STEIN'
K = 5
data_arr = load_data(filePath)
num_rand = data_arr.shape[1]



def discrete_Lmeasure(data_arr):
    
    indi_entropies = drv.entropy(data_arr.T, estimator=EST)
    num_rand = data_arr.shape[1]    
    assert num_rand == len(indi_entropies)
    L_measures = {}     # Dictionary storing the pairwise L-measures
    I_mutinfo = {}      # Dictionary storing the pairwise mutual information

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



# i and j are the respective feature number i.e ith feature and jth feature
# Specifically they are ith and jth columns in the data 2-d array
# assuming 0-indexing
def discrete_mu(data_arr, i , j):
    N = data_arr.shape[0]
    set_xi = set(data_arr[:,i])
    set_yj = set(data_arr[:,j])
    mu_sum = 0.0

    for xi in set_xi:
        for yj in set_yj:
            n_i = sum(data_arr[:,i] == xi)
            n_j = sum(data_arr[:,j] == yj)

            low = max(0, n_i + n_j - N)
            high = min(n_i, n_j)

            for n_ij in range(low, high+1):
                P_nij = ((N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (N / n_i)
                add_term = (n_ij/N) * np.log(n_ij * N / n_i * n_j) * P_nij
                mu_sum += add_term

    return mu_sum


def discrete_normalized_Lmeasure(data_arr, k):

    indi_entropies = drv.entropy(data_arr.T, estimator=EST)
    num_rand = data_arr.shape[1]
    assert num_rand == len(indi_entropies)

    L_measures = {}     # Dictionary storing the pairwise L-measures
    I_mutinfo = {}      # Dictionary storing the pairwise mutual information
    mu_vals = {}        # Dictionary storing the pairwise MU values

    for i in range(num_rand):
        for j in range(i+1, num_rand):
            key = (i+1, j+1)    # since 0-indexed
            h_i = indi_entropies[i]
            h_j = indi_entropies[j]
   
            mu_ij = discrete_mu(data_arr, i, j)
            mu_vals[key] = mu_ij    # Storing for possible future use

            # Potential error: I_ij may come out negative depending on the estiamtor   
            I_ij = drv.information_mutual(data_arr.T[i], data_arr.T[j], estimator=EST)
            #I_ij = np.abs(I_ij)     # TEMP FIX
            W_ij = min(h_i, h_j)
            
            I_ij_hat = I_ij - mu_ij
            W_ij_hat = W_ij - mu_ij

            #inner_exp_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
            inner_exp_term = (-1.0 * 2 * I_ij_hat) / (1 - float(I_ij_hat) / W_ij_hat)
            L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
            I_mutinfo[key] = I_ij

    sorted_list = sorted(L_measures.items(), key=operator.itemgetter(1), reverse=True)
    topK_keys = [item[0] for item in sorted_list[:k]]
    
    
    
    return L_measures, I_mutinfo, mu_vals



def top_k(Lmeasure_dict, k):
    sorted_list = sorted(Lmeasure_dict.items(), key=operator.itemgetter(1), reverse=True)
    # each entry is a tuple of (key,value)

    topK_keys = [item[0] for item in sorted_list[:k]]

    N = data_arr.shape[0]

    for k_tuple in topK_keys:
        i = k_tuple[0]
        j = k_tuple[1]    
        set_xi = set(data_arr[:,i])
        set_yj = set(data_arr[:,j])
    
        maxima = 0.0    # CHOOSING JUST A SINGLE PAIR
        for xi in set_xi:
            for yj in set_yj:
                b_i = data_arr[:,i] == xi
                b_j = data_arr[:,j] == yj
                n_i = sum(b_i)
                n_j = sum(b_j)
                n_ij = sum(b_i & b_j)
                
                
                
                low = max(0, n_i + n_j - N)
                high = min(n_i, n_j)

                for n_ij in range(low, high+1):
                    P_nij = ((N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (N / n_i)
                    add_term = (n_ij/N) * np.log(n_ij * N / n_i * n_j) * P_nij
                    mu_sum += add_term



    return topK_keys

lms, mutinfos = discrete_Lmeasure(data_arr)