from __future__ import division
import itertools

import numpy as np
from scipy.optimize import fmin_l_bfgs_b as spmin_LBFGSB

'''
# Todo

- opt inits improvement
- passing other optimizn params for the scipy function 
- dealing with unconstrained/problematic optimizations
- explicitly pass the function gradient as instead of approx_grad = True
    which calcs it numerically.
'''

class Solver(object):
    # features_object is an object of ExtractFeatures class
    def __init__(self, features_object, initial_theta_vec=None):

        self.featsObj = features_object
        self.init_thetas = initial_theta_vec
        
         # theta vector len = num_features (for marginals) + K
        if initial_theta_vec:           
            assert len(initial_theta_vec) == (len(self.featsObj.data_arr.shape[1] + self.featsObj.K))                
        
        # Variable to store the output of the optimization process/solver/function
        self.opt_sol = None     
        self.norm_z = None
        

    # This function computes the inner sum of the 
    # optimization function objective
    
    # could split thetas into marginal and specials
    def compute_constraint_sum(self, thetas, rvec):    
        constraint_sum = 0.0
        topK_pairs_dict = self.featsObj.feats_pairs_dict

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

    # normalization constant Z(theta)
    # assuming binary features for now.
    def binary_norm_Z(self, thetas):
        norm_sum = 0.0
        # N = self.featsObj.N        
        data_arr = self.featsObj.data_arr
        num_feats = data_arr.shape[1]
        # lst = map(list, itertools.product([0, 1], repeat=n))
        all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))
        
        for vec in all_perms:
            tmp = self.compute_constraint_sum(thetas, vec)
            norm_sum += np.exp(tmp)

        return norm_sum


    # Objective for the optimization problem
    # It is a function of thetas
    def func_objective(self, thetas):
        objective_sum = 0.0
        N = self.featsObj.N        
        data_arr = self.featsObj.data_arr

        # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
        for i in range(N):
            rvec = data_arr[i,:]
            inner_constraint_sum = self.compute_constraint_sum(thetas, rvec)
            objective_sum += inner_constraint_sum

        subtraction_term = N * np.log(self.binary_norm_Z(thetas))
        objective_sum -= subtraction_term

        return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION


    def solver_optimize(self):

        data_arr = self.featsObj.data_arr
        topK_pairs_dict = self.featsObj.feats_pairs_dict

        total_len = data_arr.shape[1] + len(topK_pairs_dict)
        
        # If inital values not specified, use a random vector
        # Initialization could be improved upon
        if not self.init_thetas:
            self.init_thetas = np.random.rand(total_len,)

        optimThetas = spmin_LBFGSB(self.func_objective, x0=self.init_thetas, 
                                    fprime=None, approx_grad=True, 
                                    disp=True)

        self.opt_sol = optimThetas
        self.norm_z = self.binary_norm_Z(self.opt_sol[0])
        return optimThetas

    def prob_dist(self, rvec):
        opt_thetas = self.opt_sol[0]
        # norm_z = self.binary_norm_Z(opt_thetas)
        term_exp = self.compute_constraint_sum(opt_thetas, rvec)

        return (1.0/self.norm_z) * np.exp(term_exp)


    def compare_marginals(self, col):        
        prob = 0.0
        
        N = self.featsObj.N        
        data_arr = self.featsObj.data_arr
        num_feats = data_arr.shape[1]
        # lst = map(list, itertools.product([0, 1], repeat=n))
        all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))
        
        for vec in all_perms:
            if vec[col] == 1:
                prob += self.prob_dist(vec)
        
        mle = 0
        for vec in data_arr:
            if vec[col] == 1:
                mle += 1
        mle = mle * 1.0 / N


        return prob, mle







