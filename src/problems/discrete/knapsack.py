import os

import numpy as np
from src.problems.base_problem import BaseProblem

class KnapsackProblem(BaseProblem):
    def __init__(self, name="Knapsack Problem"):
        self.weights =None
        self.values = None
        self.capacity = None
        self.seed = None
        self.filepath = os.path.join(BaseProblem.project_root, 'data', 'knapsack.txt')
        # Load from filepath
        self.load_from_file(self.filepath)

        # Init BaseProblem params
        super().__init__(name, dimension=len(self.weights), bounds=[0, 2], cont_flag=False)

    def load_from_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')] # skip comment lines

            if not lines:
                raise ValueError(f"The file {filename} is empty.")

            self.capacity = float(lines[0]) # assign capacity (max weigth allowed)
            weights = []
            values = []
            for line in lines[1:]: # append weight and value
                w, v = map(float, line.split())
                weights.append(w)
                values.append(v)

            self.weights = np.array(weights)
            self.values = np.array(values)

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
            penalty = total_weight - self.capacity
            return 1e-5 / penalty

        # Return Negative Value for Minimization Algorithms
        return float(total_value)


