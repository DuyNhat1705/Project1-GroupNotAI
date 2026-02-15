from src.algorithms.base_algorithm import BaseAlgorithm
import numpy as np
class TLBO(BaseAlgorithm):
    def __init__(self, params = None):
        super().__init__("TLBO",params)
        """params (dictionary) contains: population_size, num_iterations, dimensions of problem (normally 2) """
    
    def solve(self, problem, seed=None):
        print("TLBO is not implemented yet.")