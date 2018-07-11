from __future__ import division
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

class Optimizer(object):
    # features_object is an object of ExtractFeatures class
    def __init__(self, features_object, initial_theta_vec=None):

        self.featsObj = features_object
        self.init_thetas = initial_theta_vec
        
         # theta vector len = num_features (for marginals) + K
        if initial_theta_vec:           
            assert len(initial_theta_vec) == (len(self.featsObj.data_arr.shape[1] + self.featsObj.K))                
        
        # Variable to store the output of the optimization process/solver/function
        self.opt_sol = None     
        

    # This function computes the inner sum of the 
    # optimization function objective
    def compute_hr(self, thetas, rvec):    
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


    # Objective for the optimization problem
    # It is a function of thetas
    def func_objective(self, thetas):
        objective_sum = 0.0
        N = self.featsObj.N        
        data_arr = self.featsObj.data_arr

        # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
        for i in range(N):
            rvec = data_arr[i,:]
            inner_constraint_sum = self.compute_hr(thetas, rvec)
            objective_sum += inner_constraint_sum

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
        return optimThetas

