from __future__ import division
from collections import defaultdict
import operator
import numpy as np

from pyitlib import discrete_random_variable as drv


"""TODO
- improve top-K constraint calculation code?
- efficient calculation of norm_const?
- 
"""

class ExtractFeatures(object):
    """ Class summary
    Extract the relevant feature pairs from a given numpy data array to form
    the constraints for the maximum-entropy optimization algorithm. Currently it
    has methods to deal with discrete binary data arrays.

    Give extended description of extraction procedure here (see math classes
    implementation in python for reference on documentation)

    Attributes:
        data_arr: A numpy array (binary) for the disease prevalence data
        ent_estimator: String indicating which entropy estimator to use from the
            the `pyitlib` library. Default is 'JAMES-STEN'
        K: Total number of constraints to find for maxent optimization
        N: Total number of training examples in the dataset
        L_measure_dict: Dict to store the value of normalized L-measures 
            between different feature pairs.
        feats_pairs_dict: Dict to store the the top K feature pairs along with 
            their values to be used for the constraints.
        feat_graph: Dict to store the transitive graph induced by the feature
            pairs in feats_pairs_dict. Adjacency list representation is used.
        feat_partitions: List to store the partitions (connected components)
            found in the feature graph. Made up of lists containing indices for 
            each partition which have the feature(column) numbers.
    """

    def __init__(self, dataArray, entropy_estimator='JAMES-STEIN', topK=5):
        self.data_arr = dataArray
        self.ent_estimator = entropy_estimator
        self.K = topK   # number of feature pairs to extract
        self.N = self.data_arr.shape[0] # Number of training data examples
        
        self.L_measure_dict = {}  
        self.feats_pairs_dict = {}        
        self.feat_graph = {}             
        self.feat_partitions = []   


    """
    Function to get the mu values between two discrete features
    # i and j are the respective feature number i.e ith feature and jth feature
    # Specifically they are ith and jth columns in the data 2-d array
    # assuming 0-indexing
    """
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
                    P_nij = ((self.N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (self.N / n_i)  # TYPO IN FORMULA?
                    add_term = (n_ij/self.N) * np.log(n_ij * self.N / n_i * n_j) * P_nij
                    mu_sum += add_term

        return mu_sum

    
    """
    Function to compute the normalized L-measure between the all the discrete
    feature pairs
    """
    def compute_discrete_norm_Lmeasure(self):
        # TAKE note: the function expects the array to be in a transpose form
        indi_entropies = drv.entropy(self.data_arr.T, estimator=self.ent_estimator)
        num_rand = self.data_arr.shape[1]  # Number of random variables (feature columns)
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
                W_ij = min(h_i, h_j)
                W_ij_hat = W_ij - mu_ij

                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij_hat = I_ij - mu_ij
                I_ij_hat = np.max(I_ij_hat, 0) * 1.0  # Clamp it at zero, convert to float
                
                inner_exp_term = (-1.0 * 2 * I_ij_hat) / (1 - float(I_ij_hat) / W_ij_hat)
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij
                mu_vals[key] = mu_ij    # Storing for possible future use
        
        self.L_measure_dict = L_measures
        return


    """
    Function to compute the top-K feature pairs and their corresponding values
    from amongst all the pairs. Approximate computation
    """
    def compute_topK_feats_approx(self):   

        # First, run the method for setting the Lmeasures dictionary with 
        # appropriate values.
        self.compute_discrete_norm_Lmeasure()
        
        # This sorted list will also be useful in approximate partitioning 
        # by dropping the lowest L(x,y) pairs of edges in the feat-graph.
        sorted_list = sorted(self.L_measure_dict.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)


        # Just consider the top-K pairs of features first. This will ensure that
        # you will get at least K exact feature pairs (x_i, y_j) from the list.
        # each entry is a tuple of (key, value). We just want the keys
        topK_keys = [item[0] for item in sorted_list[:self.K]]        
        val_dict = {}        


        # tuple_list = []
        for k_tuple in topK_keys:
            i = k_tuple[0]
            j = k_tuple[1]    
            
            # Do this for computing when multi-valued discrete features 
            # involved. Not needed for binary.
            # set_xi = set(self.data_arr[:,i])
            # set_yj = set(self.data_arr[:,j])
            set_xi = [0,1]
            set_yj = [0,1]

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


    """
    Function to compute the top-K feature pairs and their corresponding values
    from amongst all the pairs. Exact computation
    """
    def compute_topK_feats_exact(self):   

        # First, run the method for setting the Lmeasures dictionary with 
        # appropriate values.
        self.compute_discrete_norm_Lmeasure()
        
        # This sorted list will also be useful in approximate partitioning 
        # by dropping the lowest L(x,y) pairs of edges in the feat-graph.
        sorted_list = sorted(self.L_measure_dict.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)


        # Just consider the top-K pairs of features first. This will ensure that
        # you will get at least K exact feature pairs (x_i, y_j) from the list.
        # each entry is a tuple of (key, value). We just want the keys
        topK_keys = [item[0] for item in sorted_list[:self.K]]        
        val_dict = {}        


        # tuple_list = []
        for k_tuple in topK_keys:
            i = k_tuple[0]
            j = k_tuple[1]    
            
            # Do this for computing when multi-valued discrete features 
            # involved. Not needed for binary.
            # set_xi = set(self.data_arr[:,i])
            # set_yj = set(self.data_arr[:,j])
            set_xi = [0,1]
            set_yj = [0,1]

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




    """
    # Function to create a graph out of the feature pairs (constraints)
    # Two nodes (feature indices) have an edge if they appear in a 
    # constraint together 
    """
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
    

    """
    Function to partition the set of features for easier computation
    Partitoning is equivalent to finding all the connected components in 
    the undirected graph
    """
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
        return    
