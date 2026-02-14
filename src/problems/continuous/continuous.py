import numpy as np
from src.problems.base_problem import BaseProblem

class ContinuousProblem(BaseProblem):
    def __init__(self, name, dimension, min_val, max_val, global_x=None, global_min=0.0):
        bounds = [(min_val, max_val)] * dimension
        super().__init__(name, dimension, bounds=bounds, cont_flag=True)
        self.min_range = min_val
        self.max_range = max_val
        self.global_x = global_x if global_x is not None else np.zeros(dimension)
        self.global_min = global_min

    def random_population(self, pop_size):
        """Khởi tạo quần thể ngẫu nhiên trong bounds"""
        return np.random.uniform(self.min_range, self.max_range, (pop_size, self.dimension))

    def evaluate_population(self, population):
        """Đánh giá cả quần thể (Vectorization)"""
        return np.array([self.evaluate(ind) for ind in population])

class Sphere(ContinuousProblem):
    def __init__(self, dimension=2):
        super().__init__("Sphere", dimension, -5.12, 5.12, np.zeros(dimension), 0.0)
    def evaluate(self, x):
        return np.sum(x ** 2)

class Rosenbrock(ContinuousProblem):
    def __init__(self, dimension=2):
        super().__init__("Rosenbrock", dimension, -10.0, 10.0, np.ones(dimension), 0.0)
    def evaluate(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

class Griewank(ContinuousProblem):
    def __init__(self, dimension=2):
        super().__init__("Griewank", dimension, -50.0, 50.0, np.zeros(dimension), 0.0)
    def evaluate(self, x):
        sum_term = np.sum(x**2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dimension + 1))))
        return 1 + sum_term - prod_term

class Ackley(ContinuousProblem):
    def __init__(self, dimension=2):
        super().__init__("Ackley", dimension, -32.768, 32.768, np.zeros(dimension), 0.0)
        self.a = 20; self.b = 0.2; self.c = 2 * np.pi
    def evaluate(self, x):
        d = self.dimension
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        return -self.a * np.exp(-self.b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + self.a + np.e

class Rastrigin(ContinuousProblem):
    def __init__(self, dimension=2):
        super().__init__("Rastrigin", dimension, -5.12, 5.12, np.zeros(dimension), 0.0)
    def evaluate(self, x):
        return 10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

class Michalewicz(ContinuousProblem):
    def __init__(self, dimension=2, m=10):
        g_x = np.array([2.20, 1.57]) if dimension == 2 else None
        g_min = -1.8013 if dimension == 2 else None
        super().__init__("Michalewicz", dimension, 0.0, np.pi, g_x, g_min)
        self.m = m
    def evaluate(self, x):
        i = np.arange(1, self.dimension + 1)
        return -np.sum(np.sin(x) * (np.sin(i * x**2 / np.pi)) ** (2 * self.m))