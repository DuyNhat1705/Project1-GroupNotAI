import numpy as np
from src.problems.base_problem import BaseProblem

class KnapsackProblem(BaseProblem):
    def __init__(self, name, weights, values, capacity):
        """
        Args:
            weights (list): Weight of each item
            values (list): Value of each item
            capacity (int): Max weight knapsack can hold
        """

        if len(weights) != len(values):
            raise ValueError("Weights and Values must be the same length") #ensure each item has both weight and value

        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity

        # dimension = num of items
        # bounds = [0, 2] (0 or 1)
        # cont_flag = False (Discrete)
        super().__init__(name, dimension=len(weights), bounds=[0, 2], cont_flag=False)

    def evaluate(self, solution):
        """
        Calculate fitness.
        solution: arr of 0/1 (flag of selection)
        """
        # round to int if any algo pass float
        selection = np.round(solution).astype(int)

        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)

        # Constraint Handling
        if total_weight > self.capacity:
            return 0

        # Return Negative Value for Minimization Algorithms
        return float(total_value)


