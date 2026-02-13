import os
import numpy as np
import random
from src.problems.base_problem import BaseProblem

class Rosenbrock_function(BaseProblem):
    # minimum = 0 at x = [1 ; ... ; 1]
    def __init__(self, name = "Rosenbrock", dimension = 3, bounds = [-1.0, 1.0], seed=None):
        super().__init__(name, dimension=dimension, bounds=bounds)

    def evaluate(self, x):
        """"
        x[i] i from 0 to dimension-1 & x[i+1] from 1 to highest dimension
        """
        return np.sum((100*(x[1:]-x[:-1]**2)**2 + (1 - x[:-1])**2)**2)

class Griewank_function(BaseProblem):
    def __init__(self, name = "Griewank", dimension=3, bounds=[-600.0, 600.0], seed=None):
        super().__init__(name, dimension=dimension, bounds=bounds)

    def evaluate(self, x):
        """
        f(x) = 1 + (1/4000) * sum(x_i^2) - prod(cos(x_i / sqrt(i)))
        """
        x = np.array(x)

        # Term 1: Sum of squares
        sum_sq = np.sum(x ** 2) / 4000.0

        # Term 2: Product of cosines
        # Note: i ranges from 1 to d (indices in formula are 1-based)
        indices = np.arange(1, len(x) + 1)
        prod_cos = np.prod(np.cos(x / np.sqrt(indices)))

        return 1.0 + sum_sq - prod_cos


class Ackley_function(BaseProblem):
    def __init__(self, name="Ackley", dimension=3, bounds=[-32.0, 32.0], seed=None):
        super().__init__(name, dimension=dimension, bounds=bounds)

    def evaluate(self, x):
        """
        f(x) = -20 * exp(-0.2 * sqrt(1/d * sum(x^2)))
               - exp(1/d * sum(cos(2*pi*x)))
               + 20 + e
        """
        x = np.array(x)
        d = float(len(x))

        # Term 1: Exponential of sum squares
        sum_sq = np.sum(x ** 2)
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))

        # Term 2: Exponential of cosine sum
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        term2 = -np.exp(sum_cos / d)

        return term1 + term2 + 20.0 + np.e


class Sphere_function(BaseProblem):
    def __init__(self, name="Ackley", dimension=3, bounds=[-32.0, 32.0], seed=None):
        super().__init__(name, dimension=dimension, bounds=bounds)

    def evaluate(self, x):
      """
      Sphere benchmark function: global minimum at f(0, ..., 0) = 0.
      """
      x = np.array(x)
      z = 0.0
      for i in range(self.dimension):
        z += x[i] ** 2
      return z







