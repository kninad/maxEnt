from __future__ import division
import operator
import numpy as np

from pyitlib import discrete_random_variable as drv
from data_loading import load_data

# EST = 'JAMES-STEIN'

def discrete_Lmeasure(data_arr, EST):    
    indi_entropies = drv.entropy(data_arr.T, estimator=EST)
    num_rand = data_arr.shape[1]    
    assert num_rand == len(indi_entropies)
    L_measures = {}     # Dictionary storing the pairwise L-measures
    I_mutinfo = {}      # Dictionary storing the pairwise mutual information

    for i in range(num_rand):
        for j in range(i+1, num_rand):
            key = (i, j)    # Remember!! 0-indexed
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

            for n_ij in range(low + 1, high):
                P_nij = ((N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (N / n_i)
                add_term = (n_ij/N) * np.log(n_ij * N / n_i * n_j) * P_nij
                mu_sum += add_term

    return mu_sum


def discrete_normalized_Lmeasure(data_arr, EST):
    # TAKE note: the function expects the array to be in a transpose form
    indi_entropies = drv.entropy(data_arr.T, estimator=EST)
    num_rand = data_arr.shape[1]
    assert num_rand == len(indi_entropies)

    L_measures = {}     # Dictionary storing the pairwise L-measures
    I_mutinfo = {}      # Dictionary storing the pairwise mutual information
    mu_vals = {}        # Dictionary storing the pairwise MU values

    for i in range(num_rand):
        for j in range(i+1, num_rand):
            key = (i, j)    # since 0-indexed
            h_i = indi_entropies[i]
            h_j = indi_entropies[j]
   
            mu_ij = discrete_mu(data_arr, i, j)            

            # Potential error: I_ij may come out negative depending on the estiamtor   
            I_ij = drv.information_mutual(data_arr.T[i], data_arr.T[j], estimator=EST)
            #I_ij = np.abs(I_ij)     # TEMP FIX
            W_ij = min(h_i, h_j)
            
            # Potential error: I_ij may come out negative depending on the estiamtor   
            I_ij_hat = I_ij - mu_ij
            I_ij_hat = np.abs(I_ij_hat)
            W_ij_hat = W_ij - mu_ij

            #inner_exp_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
            inner_exp_term = (-1.0 * 2 * I_ij_hat) / (1 - float(I_ij_hat) / W_ij_hat)
            L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
            I_mutinfo[key] = I_ij
            mu_vals[key] = mu_ij    # Storing for possible future use

    return L_measures, I_mutinfo, mu_vals



def topK_feats(L_measure_dict, data_arr, K):    
    sorted_list = sorted(L_measure_dict.items(), key=operator.itemgetter(1), reverse=True)
    # each entry is a tuple of (key, value). We just want the keys
    topK_keys = [item[0] for item in sorted_list[:K]]        
    val_dict = {}
    N = data_arr.shape[0]

    for k_tuple in topK_keys:
        i = k_tuple[0]
        j = k_tuple[1]    
        set_xi = set(data_arr[:,i])
        set_yj = set(data_arr[:,j])

        # CHOOSING JUST A SINGLE maxima PAIR of values
        # Can update to include multiple later on
        maxima = 0.0    
        for xi in set_xi:
            for yj in set_yj:
                b_i = data_arr[:,i] == xi
                b_j = data_arr[:,j] == yj
                n_i = sum(b_i)
                n_j = sum(b_j)
                n_ij = sum(b_i & b_j)
                
                # print(i,j, xi, yj, n_i, n_j, n_ij)
                delta_ij = np.abs( (n_ij / N) * np.log((n_ij * N) / (n_i * n_j)) )

                if delta_ij > maxima :
                    maxima = delta_ij
                    val_dict[k_tuple] = (xi, yj)

    return val_dict



# Make it into a class so that data array is shared amongst all the methods
# Things like EST can also be set like that

class Features(object):

    def __init__(self, dataArray, entropy_estimator='JAMES-STEIN', topK=5):
        self.data_arr = dataArray
        self.ent_estimator = entropy_estimator
        self.K = topK        
        self.N = self.data_arr.shape[0] # Number of training examples
        
        self.L_measure_dict = {}    # Will be assinged once the appropriate method is run
        self.top_feats_dict = {}    # Will be assinged once the appropriate method is run

    def discrete_Lmeasure(self):    
        indi_entropies = drv.entropy(self.data_arr.T, self.ent_estimator)   # indvidual entropies
        num_rand = self.data_arr.shape[1]    # number of features / random variables
        
        assert num_rand == len(indi_entropies)

        L_measures = {}     # Dictionary storing the pairwise L-measures
        I_mutinfo = {}      # Dictionary storing the pairwise mutual information

        for i in range(num_rand):
            for j in range(i+1, num_rand):
                key = (i, j)    # Remember!! 0-indexed
                h_i = indi_entropies[i]
                h_j = indi_entropies[j]
    
                # Potential error: I_ij may come out negative depending on the estiamtor
    
                I_ij = drv.information_mutual(self.data_arr.T[i], 
                                                self.data_arr.T[j],
                                                estimator=self.ent_estimator)
                
                # TEMP FIX for I(x,y) going negative
                I_ij = np.abs(I_ij)    
                
                W_ij = min(h_i, h_j)
                inner_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
                L_measures[key] = np.sqrt(1 - np.exp(inner_term))
                I_mutinfo[key] = I_ij

        self.L_measure_dict = L_measures
        # return L_measures



    # i and j are the respective feature number i.e ith feature and jth feature
    # Specifically they are ith and jth columns in the data 2-d array
    # assuming 0-indexing
    def discrete_mu(self, i , j):        
        set_xi = set(self.data_arr[:,i])
        set_yj = set(self.data_arr[:,j])
        mu_sum = 0.0

        for xi in set_xi:
            for yj in set_yj:
                n_i = sum(self.data_arr[:,i] == xi)
                n_j = sum(self.data_arr[:,j] == yj)

                low = max(0, n_i + n_j - N)
                high = min(n_i, n_j)

                for n_ij in range(low + 1, high):
                    P_nij = ((N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (N / n_i)
                    add_term = (n_ij/N) * np.log(n_ij * N / n_i * n_j) * P_nij
                    mu_sum += add_term

        return mu_sum


    def discrete_normalized_Lmeasure(self):
        # TAKE note: the function expects the array to be in a transpose form
        indi_entropies = drv.entropy(self.data_arr.T, estimator=self.ent_estimator)
        num_rand = self.data_arr.shape[1]
        assert num_rand == len(indi_entropies)

        L_measures = {}     # Dictionary storing the pairwise L-measures
        I_mutinfo = {}      # Dictionary storing the pairwise mutual information
        mu_vals = {}        # Dictionary storing the pairwise MU values

        for i in range(num_rand):
            for j in range(i+1, num_rand):
                key = (i, j)    # since 0-indexed
                h_i = indi_entropies[i]
                h_j = indi_entropies[j]
    
                mu_ij = discrete_mu(self, i, j)            

                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij = drv.information_mutual(self.data_arr.T[i], self.data_arr.T[j], estimator=self.ent_estimator)
                #I_ij = np.abs(I_ij)     # TEMP FIX
                W_ij = min(h_i, h_j)
                
                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij_hat = I_ij - mu_ij
                I_ij_hat = np.abs(I_ij_hat)
                
                W_ij_hat = W_ij - mu_ij

                #inner_exp_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
                inner_exp_term = (-1.0 * 2 * I_ij_hat) / (1 - float(I_ij_hat) / W_ij_hat)
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij
                mu_vals[key] = mu_ij    # Storing for possible future use
        
        self.L_measure_dict = L_measures
        # return L_measures



    def topK_feats(self):   

        # First, run the method for setting the Lmeasures dictionary
        self.discrete_normalized_Lmeasure()
        # Now the dict is accessible to the current method

        sorted_list = sorted(self.L_measure_dict.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)

        # each entry is a tuple of (key, value). We just want the keys
        topK_keys = [item[0] for item in sorted_list[:self.K]]        
        val_dict = {}        

        for k_tuple in topK_keys:
            i = k_tuple[0]
            j = k_tuple[1]    
            set_xi = set(self.data_arr[:,i])
            set_yj = set(self.data_arr[:,j])

            # CHOOSING JUST A SINGLE maxima PAIR of values
            # Can update to include multiple later on
            maxima = 0.0    
            for xi in set_xi:
                for yj in set_yj:
                    b_i = self.data_arr[:,i] == xi
                    b_j = self.data_arr[:,j] == yj
                    n_i = sum(b_i)
                    n_j = sum(b_j)
                    n_ij = sum(b_i & b_j)
                    
                    # print(i,j, xi, yj, n_i, n_j, n_ij)
                    delta_ij = np.abs( (n_ij / self.N) * np.log((n_ij * self.N) / (n_i * n_j)) )

                    if delta_ij > maxima :
                        maxima = delta_ij
                        val_dict[k_tuple] = (xi, yj)


        self.top_feats_dict = val_dict
        return val_dict     # can comment it out