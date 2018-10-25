from __future__ import division
from collections import defaultdict
import operator
import numpy as np

from pyitlib import discrete_random_variable as drv


"""TODO
- better documentation
    - give math formulas for the variables defined in the functions
    - where applicable, explain the code woriking with one line comments
- improve top-K constraint calculation code?
    - through some caching of often used computations?
    - since this is a one-time computation, it will be helpful
- approximate the algo for connected components?
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
        self.count_map = {}

    def compute_binary_counts(self):
        """
        Redundant function, used in the normalized L-measure calculation.
        Ignore it for now.
        """
        N, num_rand = self.data_arr.shape
        counts = defaultdict(int)
        # set_xi = set([0, 1])

        for i in range(num_rand):
            n_i_0 = sum(self.data_arr[:,i] == 0)
            n_i_1 = N - n_i_0
            counts[(i, 0)] = n_i_0
            counts[(i, 1)] = n_i_1
        
        self.count_map = counts
        return


    def get_discrete_mu(self, i , j):        
        """Function to get the mu values between two discrete BINARY features
        i and j are the respective feature number i.e ith feature and jth feature
        Specifically they are ith and jth columns in the data 2-d array
        assuming 0-indexing

        Args:
            i: Feature column number i i.e ith random variable
            j: Feature column number j i.e jth random variable
        
        Returns:
            mu_sum: The value of the function mu(i,j). Define it here!
        """
        # set_xi = set(self.data_arr[:,i])
        # set_yj = set(self.data_arr[:,j])
        
        # Compute the counts and set the class attribute counts
        self.compute_binary_counts()
        # counts = self.count_map

        # Since considering binary features for now.
        set_xi = set([0, 1])
        set_yj = set([0, 1])
        mu_sum = 0.0

        for xi in set_xi:
            for yj in set_yj:
                # This computation can be stored instead of computing it again 
                # and again
                n_i = sum(self.data_arr[:,i] == xi)
                n_j = sum(self.data_arr[:,j] == yj)
                # n_i = counts[(i, xi)]
                # n_j = counts[(j, yj)]

                low = max(0, n_i + n_j - self.N)
                high = min(n_i, n_j)

                for n_ij in range(low + 1, high):
                    P_nij = ((self.N - n_j) / (n_i - n_ij)) * (n_j / n_ij) * (self.N / n_i)  # TYPO IN FORMULA?
                    add_term = (n_ij/self.N) * np.log(n_ij * self.N / n_i * n_j) * P_nij
                    mu_sum += add_term

        return mu_sum
    

    

    def compute_discrete_Lmeasure(self):
        """Function to compute the un-normalized L-measure between the all the 
        discrete feature pairs. The value for all the possible pairs is stored
        in the L_measures dict. Auxiliary values like the mutual information
        (I_mutinfo) are also in their respective dicts for all the possible pairs.        
        This method sets the `feats_pairs_dict` class attribute.

        Args:
            None
        
        Returns:
            None
        """
        # TAKE note: the function expects the array to be in a transpose form
        indi_entropies = drv.entropy(self.data_arr.T, estimator=self.ent_estimator)
        # indi_entropies = drv.entropy(self.data_arr.T)
        num_rand = self.data_arr.shape[1]  # Number of random variables (feature columns)
        assert num_rand == len(indi_entropies)

        L_measures = {}     # Dictionary storing the pairwise L-measures
        I_mutinfo = {}      # Dictionary storing the pairwise mutual information
        # mu_vals = {}        # Dictionary storing the pairwise MU values

        for i in range(num_rand):
            for j in range(i+1, num_rand):
                key = (i, j)    # since 0-indexed
                h_i = indi_entropies[i]
                h_j = indi_entropies[j]
    
                # mu_ij = self.get_discrete_mu(i, j)            

                # Potential error: I_ij may come out negative depending on the estiamtor   
                I_ij = drv.information_mutual(self.data_arr.T[i], self.data_arr.T[j], estimator=self.ent_estimator)                
                W_ij = min(h_i, h_j)
                
                num = (-2.0 * I_ij * W_ij)
                den = (W_ij - I_ij)
                eps = 1e-9   # epsilon value for denominator
                inner_exp_term = num/(den + eps)                              
                # removing numerical errors by upper bounding exponent by 0
                inner_exp_term = min(0, inner_exp_term)
                
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij                

                # print(I_ij, W_ij, num, den)
                # print(key, L_measures[key], inner_exp_term)
                # print('\n')

        
        self.L_measure_dict = L_measures
        return


    def compute_discrete_norm_Lmeasure(self):
        """Function to compute the normalized L-measure between the all the 
        discrete feature pairs. The value for all the possible pairs is stored
        in the L_measures dict. Auxiliary values like the mutual information
        (I_mutinfo) and mu-values (mu_vals) are also in their respective 
        dicts for all the possible pairs.
        
        This method sets the `feats_pairs_dict` class attribute.

        Args:
            None
        
        Returns:
            None
        """
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

                # Potential error: I_ij_hat may come out negative
                I_ij_hat = I_ij - mu_ij
                                
                num = -2.0 * I_ij_hat * W_ij_hat
                den = W_ij_hat - I_ij_hat
                inner_exp_term = num/den                
                # removing numerical errors by bounding exponent by 0
                inner_exp_term = min(0, inner_exp_term)
                
                L_measures[key] = np.sqrt(1 - np.exp(inner_exp_term))
                I_mutinfo[key] = I_ij
                mu_vals[key] = mu_ij    # Storing for possible future use

                print(I_ij, I_ij_hat, W_ij, W_ij_hat, mu_ij)
                print(key, L_measures[key], inner_exp_term)
                print('\n')
        
        # self.L_measure_dict = L_measures
        self.L_measure_dict = L_measures
        return


    def compute_topK_feats(self):   
        """Function to compute the top-K feature pairs and their corresponding 
        feature assignment from amongst all the pairs. Approximate computation: 
        Select the top K pairs based on their L_measures values. For each pair 
        just select the highest scoring feature assignment. Score is calculated
        by $\delta(x_i, y_j)$. 
        
        This method sets the `feats_pairs_dict` class attribute.

        Args:
            None

        Returns:
            None 
        """

        # First, run the method for setting the Lmeasures dictionary with 
        # appropriate values.
        print("Computing the L_measures between the feature pairs")
        # self.compute_discrete_norm_Lmeasure() # Only use it for multi-discrete
        self.compute_discrete_Lmeasure() # Use it for binary case
        
        print("Sorting the L_measures")
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

        print("Computing the topK pairs")
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
                    n_i = sum(b_i) # CAN BE pre-fetched
                    n_j = sum(b_j) # CAN be pre-fetched
                    # n_i = counts[(i, xi)]
                    # n_j = counts[(j, yj)]
                    n_ij = sum(b_i & b_j)
                    
                    # print(i,j, xi, yj, n_i, n_j, n_ij)
                    delta_ij = np.abs( (n_ij / self.N) * np.log((n_ij * self.N) / (n_i * n_j)) )

                    if delta_ij > maxima :
                        maxima = delta_ij
                        val_dict[k_tuple] = (xi, yj)

        print(k_tuple, (xi, yj), maxima)
        self.feats_pairs_dict = val_dict
        # return val_dict     # can comment it out

   
    def create_partition_graph(self):
        """Function to create a graph out of the feature pairs (constraints)
        Two nodes (feature indices) have an edge between them if they appear in 
        a constraint together. The graph is an adjacency list representation
        stored in the graph dictionary.
        
        This method sets the class attribute `feat_graph` to the graph dict

        Args: 
            None

        Returns:
            None
        """
        graph = {}  # undirected graph
        num_feats = self.data_arr.shape[1]

        # init for each node an empty set of neighbors
        for i in range(num_feats):
            graph[i] = set()

        print("Creating the feature graph")
        # create adj-list representation of the graph
        # the key for the dict are the (X,Y) pairs
        for tup in self.feats_pairs_dict:
            print("Added edge for:", tup)
            graph[tup[0]].add(tup[1])
            graph[tup[1]].add(tup[0])

        self.feat_graph = graph
        # return graph
    


    """
    TODO: How to pick a relevant value for the treshold? 0.5 seems like a high 
            value for tresholding the L_measure value?
            An efficient way would be to drop the low-scoring ones. Maybe 
            select the median OR select a given perecentile (from the 
            sorted top-K list of top L-measures )
    """
    def create_partition_graph_approx(self, threshold=0.5):
        """ Function to create a graph out of the feature pairs (constraints)
        Two nodes (feature indices) have an edge if they appear in a 
        constraint together. This is an approximate method where certain 
        edges between two nodes (i,i) are dropped if their edge-weight falls 
        below a certain threshold.         
        Edge-weight is equal to the value L(i,j) i.e how dependent the two 
        random variables are according to their L measure.
        This approximate method may be useful when the value for K is very large.

        This method sets the class attribute `feat_graph` to the graph dict

        Args:
            threshold: Value between 0 and 1 used for dropping the edges. 
                Default is 0.5
        
        Returns: 
            None
        """
        graph = {}  # undirected graph
        num_feats = self.data_arr.shape[1]

        lms_dict = self.L_measure_dict

        # init for each node an empty set of neighbors
        for i in range(num_feats):
            graph[i] = set()

        # create adj-list representation of the graph
        # the key for the dict are the (X,Y) pairs
        for tup in self.feats_pairs_dict:
            val = lms_dict[tup]

            # only add an edge if the value is above the threshold
            if val >= threshold:
                graph[tup[0]].add(tup[1])
                graph[tup[1]].add(tup[0])

        self.feat_graph = graph
        # return graph    


    def partition_features(self):        
        """Function to partition the set of features (for easier computation).
        Partitoning is equivalent to finding all the connected components in 
        the undirected graph of the features indices as their nodes.         
        This method find the partitions as sets of feature indices and stores 
        them in a list of lists with each inner list storing the indices 
        corresponding to a particular partition.

        This method sets the class attribute `feats_partitions` which is list of
        lists containing the partition assignments.

        Args:
            None
        
        Returns:
            None
        """
        self.create_partition_graph()
        print("Partioning the feature graph")

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
        print("Finding the connected components")
        for comp in connected_components(self.feat_graph):
            partitions.append(list(comp))
        
        self.feat_partitions = partitions
        return    
