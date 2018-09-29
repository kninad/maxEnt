from __future__ import division
import itertools
from collections import defaultdict

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
    def __init__(self, features_object):
        self.featsObj = features_object
        
        # Variables to store the output of the optimization process/solver/function
        self.opt_sol = None     
        self.norm_z = None
        

    # This function computes the inner sum of the 
    # optimization function objective
    
    # could split thetas into marginal and specials
    def compute_constraint_sum(self, thetas, rvec, partition):    
        # print '\n'
        # print partition
        
        constraint_sum = 0.0
        topK_pairs_dict = self.featsObj.feats_pairs_dict

        # k is a tuple == feature-pair.
        # topK_list = [(k,v) for k,v in topK_pairs_dict.items() if (k[0] in partition and k[1] in partition) ]
        topK_list = []

        for k,v in topK_pairs_dict.items():
            # print partition, k,v   
            condition = k[0] in partition and k[1] in partition
            if condition:
                topK_list.append((k,v))
        
        # marginals    
        # num_feats = len(rvec)
        num_feats = len(partition)
        assert len(rvec) == num_feats
        assert len(rvec) == (len(thetas) - len(topK_list))
        
        # CHECKING WITH 1 since BINARY FEATURES
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            constraint_sum += thetas[i] * indicator
        
        # this dict is needed since we are given a 'cropped'
        # vector and hence the indices may not match up
        findpos = {elem:i for i,elem in enumerate(partition)}
        
        # top K constraints
        # will not execute if a single feature list
        for j, tup in enumerate(topK_list):
            key = tup[0]
            val = tup[1]
            condition = rvec[findpos[key[0]]] == val[0] and rvec[findpos[key[1]]] == val[1]
            indicator = 1 if condition else 0
            constraint_sum += thetas[j + num_feats] * indicator
            # SINCE THETAS IS CONTIGUOUS VECTOR
            # CAN ALSO BREAK UP THETAS into MARGINAL AND topK CONSTRATINTS    

        return constraint_sum


    # normalization constant Z(theta)
    # assuming binary features for now.
    def binary_norm_Z(self, thetas, partition):
        norm_sum = 0.0
        # N = self.featsObj.N        
        # data_arr = self.featsObj.data_arr
        
        # num_feats = data_arr.shape[1]
        num_feats = len(partition)

        # lst = map(list, itertools.product([0, 1], repeat=n))
        all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))
        
        for vec in all_perms:
            tmp = self.compute_constraint_sum(thetas, vec, partition)
            norm_sum += np.exp(tmp)

        return norm_sum


    # Objective for the optimization problem
    # It is a function of thetas
    # def func_objective(self, thetas, partition):
    #     objective_sum = 0.0
    #     N = self.featsObj.N        
    #     data_arr = self.featsObj.data_arr

    #     # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
    #     for i in range(N):
    #         rvec = data_arr[i, partition]
    #         inner_constraint_sum = self.compute_constraint_sum(thetas, rvec, partition)
    #         objective_sum += inner_constraint_sum

    #     subtraction_term = N * np.log(self.binary_norm_Z(thetas, partition))
    #     objective_sum -= subtraction_term

    #     return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION


    def solver_optimize(self):
        parts = self.featsObj.feat_partitions
        solution = [None for i in parts]
        norm_sol = [None for i in parts]

        topK_pairs_dict = self.featsObj.feats_pairs_dict

        for i,partition in enumerate(parts):
            length1 = len(partition)
        
            # number of 'extra' constraints for that partition
            length2 = len([(k,v) for k,v in topK_pairs_dict.items() 
                        if (k[0] in partition and k[1] in partition) ])
        
            initial_val = np.random.rand(length1+length2)

            def func_objective(thetas):
                objective_sum = 0.0
                N = self.featsObj.N        
                data_arr = self.featsObj.data_arr

                # THIS CAN SPED UP BY EFFICIENT NUMPY OPERATIONS
                for i in range(N):
                    rvec = data_arr[i, partition]
                    inner_constraint_sum = self.compute_constraint_sum(thetas, rvec, partition)
                    objective_sum += inner_constraint_sum

                subtraction_term = N * np.log(self.binary_norm_Z(thetas, partition))
                objective_sum -= subtraction_term

                return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION


            optimThetas = spmin_LBFGSB(func_objective, x0=initial_val,
                                    fprime=None, approx_grad=True, 
                                    disp=True)

            solution[i] = optimThetas
            norm_sol[i] = self.binary_norm_Z(optimThetas[0], partition)

        self.opt_sol = solution
        self.norm_z = norm_sol

        return (solution, norm_sol)


    def prob_dist(self, rvec):
        
        prob_product = 1.0
        parts = self.featsObj.feat_partitions
        solution = self.opt_sol
        norm_sol = self.norm_z

        for i,partition in enumerate(parts):
            tmpvec = rvec[partition]
            term_exp = self.compute_constraint_sum(solution[i][0], tmpvec, partition)
            prob_part = (1.0/norm_sol[i]) * np.exp(term_exp)
            prob_product *= prob_part

        return prob_product


    def compare_marginals(self):        
                
        N = self.featsObj.N        
        data_arr = self.featsObj.data_arr
        num_feats = data_arr.shape[1]
        # lst = map(list, itertools.product([0, 1], repeat=n))
        all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))
        
        mxt_dict = defaultdict(float)
        emp_dict = defaultdict(float)

        for vec in all_perms:
            for j in range(num_feats):
                if vec[j] == 1:
                    mxt_dict[j] += self.prob_dist(vec)
        
        
        for vec in data_arr:
            for j in range(num_feats):
                if vec[j] == 1:
                    emp_dict[j] += 1


        for k in emp_dict:
            emp_dict[k] /= N

        return mxt_dict, emp_dict


    def compare_constraints(self):        

        N = self.featsObj.N        
        data_arr = self.featsObj.data_arr
        num_feats = data_arr.shape[1]
        # lst = map(list, itertools.product([0, 1], repeat=n))
        all_perms = map(np.array, itertools.product([0, 1], repeat=num_feats))

        pair_dict = self.featsObj.feats_pairs_dict
        mxt_dict = defaultdict(float)
        emp_dict = defaultdict(float)

        for vec in all_perms:
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1]:
                    mxt_dict[(key,val)] += self.prob_dist(vec)
        
        
        for vec in data_arr:
            for key,val in pair_dict.items():
                if vec[key[0]] == val[0] and vec[key[1]] == val[1]:
                    emp_dict[(key,val)] += 1.0

        for k in emp_dict:
            emp_dict[k] /= N

        return mxt_dict, emp_dict



    def transition_prob(self, rv1, rv2):
        # rv1 and rv2 are the first and second year's disease 
        # prevalence respectively
        given_rvec = np.append(rv1, rv2)       
        norm_prob = 0   # g_a(r)
        num_feats2 = len(rv2)
        rv2_perms = map(np.array, itertools.product([0, 1], repeat=num_feats2))

        for tmp_v2 in rv2_perms:
            tmp = np.append(rv1, tmp_v2)
            norm_prob += self.prob_dist(tmp)
        
        trans_prob = self.prob_dist(given_rvec)/norm_prob

        return trans_prob
