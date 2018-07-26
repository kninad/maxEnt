from __future__ import division
from collections import defaultdict
import operator
import numpy as np

from pyitlib import discrete_random_variable as drv


class ExtractFeatures(object):

    def __init__(self, dataArray, entropy_estimator='JAMES-STEIN', topK=5):
        self.data_arr = dataArray
        self.ent_estimator = entropy_estimator
        self.K = topK        
        self.N = self.data_arr.shape[0] # Number of training data examples
        
        self.L_measure_dict = {}    # Will be assinged once the appropriate method is run
        self.feats_pairs_dict = {} # For the exact feature value pairs in the Constraints
        self.feat_graph = {}     # Feature realtion graph, ADJ-LIST rep
        self.feat_partitions = []   # Feature partitions

    # un-normalized L-measure
    def compute_discrete_Lmeasure(self):    
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
                # I_ij = np.abs(I_ij)
                I_ij = np.max(I_ij, 0)    
                
                W_ij = min(h_i, h_j)
                inner_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
                L_measures[key] = np.sqrt(1 - np.exp(inner_term))
                I_mutinfo[key] = I_ij

        self.L_measure_dict = L_measures
        # return L_measures


    # i and j are the respective feature number i.e ith feature and jth feature
    # Specifically they are ith and jth columns in the data 2-d array
    # assuming 0-indexing
    def get_discrete_mu(self, i , j):        
        set_xi = set(self.data_arr[:,i])
        set_yj = set(self.data_arr[:,j])
        mu_sum = 0.0

        for xi in set_xi:
            for yj in set_yj:
                n_i = sum(self.data_arr[:,i] == xi)
                n_j = sum(self.data_arr[:,j] == yj)

                low = max(0, n_i + n_j - self.N)
                high = min(n_i, n_j)

                for n_ij in range(low + 1, high):
                    P_nij = ((self.N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (self.N / n_i)
                    add_term = (n_ij/self.N) * np.log(n_ij * self.N / n_i * n_j) * P_nij
                    mu_sum += add_term

        return mu_sum


    def compute_discrete_norm_Lmeasure(self):
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
    
                mu_ij = self.get_discrete_mu(i, j)            

                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij = drv.information_mutual(self.data_arr.T[i], self.data_arr.T[j], estimator=self.ent_estimator)
                #I_ij = np.abs(I_ij)     # TEMP FIX
                W_ij = min(h_i, h_j)
                
                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij_hat = I_ij - mu_ij
                # I_ij_hat = np.abs(I_ij_hat)
                I_ij_hat = np.max(I_ij_hat, 0)
                
                W_ij_hat = W_ij - mu_ij

                #inner_exp_term = (-1.0 * 2 * I_ij) / (1 - float(I_ij)/W_ij)
                inner_exp_term = (-1.0 * 2 * I_ij_hat) / (1 - float(I_ij_hat) / W_ij_hat)
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij
                mu_vals[key] = mu_ij    # Storing for possible future use
        
        self.L_measure_dict = L_measures
        # return L_measures



    def compute_topK_feats(self):   

        # First, run the method for setting the Lmeasures dictionary
        self.compute_discrete_norm_Lmeasure()
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

        self.feats_pairs_dict = val_dict
        # return val_dict     # can comment it out


    def create_partition_graph(self):
        graph = {}  # undirected graph
        num_feats = self.data_arr.shape[1]

        # init for each node an empty set of neighbors
        for i in range(num_feats):
            graph[i] = set()

        # create adj-list representation of the graph
        for tup in self.feats_pairs_dict:
            graph[tup[0]].add(tup[1])
            graph[tup[1]].add(tup[0])

        self.feat_graph = graph
        # return graph

    def partition_features(self):        
        self.create_partition_graph()

        def connected_components(neighbors):
            seen = set()
            def component(node):
                nodes = set([node])
                while nodes:
                    node = nodes.pop()
                    seen.add(node)
                    nodes |= neighbors[node] - seen
                    yield node
            for node in neighbors:
                if node not in seen:
                    yield component(node)

        partitions = []
        for comp in connected_components(self.feat_graph):
            partitions.append(list(comp))
        
        self.feat_partitions = partitions
        # return partitions
        


