import os
import numpy as np
import random
from src.problems.base_problem import BaseProblem

class TSPProblem(BaseProblem):
    def __init__(self, dist_mat):
        """
        Args:
            dist_mat (np.ndarray): NxN distance matrix
        """
        self.dist_mat = dist_mat
        dim = dist_mat.shape[0] # dimension of matrix = number of nodes

        # Bound 0 to N-1
        bounds = [(0, dim - 1)] * dim

        super().__init__(name="TSP", dimension=dim, bounds=bounds, cont_flag=False)

    def evaluate(self, solution):
        """
        Calculates total distance of the path.
        """
        # Ensure solution is int idx
        path = np.array(solution, dtype=int)

        total_cost = 0
        dim = self.dimension

        for i in range(dim - 1):
            start_node = path[i]
            end_node = path[i + 1]
            total_cost += self.dist_mat[start_node][end_node]

        # cost of last edge of the circuit
        total_cost += self.dist_mat[path[-1]][path[0]] # from end back to start

        return total_cost