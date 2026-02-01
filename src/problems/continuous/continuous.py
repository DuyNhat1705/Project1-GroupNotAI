import os
import numpy as np
import random
from src.problems.base_problem import BaseProblem

class Rosenbrock_function(BaseProblem):
    # minimum = 0 at x = [1 ; ... ; 1]
    def __init__(self, name, dimension = 3, bounds = [-1.0, 1.0], seed=None):
        super().__init__(name, dimension=dimension, bounds=bounds)

    def evaluate(self, x):
        """"
        x[i] i from 0 to dimension-1 & x[i+1] from 1 to highest dimension
        """
        return np.sum((100*(x[1:]-x[:-1]**2)**2 + (1 - x[:-1])**2)**2)



