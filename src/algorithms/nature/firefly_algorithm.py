from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
import random
import math


class FireflyAlgorithm(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'population_size': 25, 'iterations': 100, 'alpha': 0.5,
            'beta_0': 1.0, 'gamma': 1.0
        }
        if params:
            default_params.update(params)
        super().__init__("Firefly Algorithm", default_params)
        for key, val in default_params.items():
            setattr(self, key, val)

    def calc_fitness(self, flag, cost):
        if flag:
            return 1 / (cost + 1) if cost >= 0 else 1 + np.abs(cost)
        return cost

    def solve(self, problem, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        dims = problem.dimension
        flag = problem.cont_flag
        lb, ub = (problem.bounds[:, 0], problem.bounds[:, 1]) if flag else (0, 2)
        
        # Initialize fireflies
        fireflies = np.random.uniform(lb, ub, (self.population_size, dims)) if flag else \
                    np.random.randint(0, 2, (self.population_size, dims)).astype(float)
        
        fitness = np.array([problem.evaluate(f) for f in fireflies])
        best_idx = np.argmin(fitness) if flag else np.argmax(fitness)
        best_position, best_cost = fireflies[best_idx].copy(), fitness[best_idx]
        
        for iteration in range(self.iterations):
            alpha = self.alpha * (0.97 ** iteration)
            new_fireflies = fireflies.copy()
            
            for i in range(self.population_size):
                moved = False
                for j in range(self.population_size):
                    is_brighter = (i != j and ((fitness[j] < fitness[i]) if flag else (fitness[j] > fitness[i])))
                    
                    if is_brighter:
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta = self.beta_0 * math.exp(-self.gamma * r**2)
                        new_fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(dims) - 0.5)
                        moved = True
                        
                        if flag:
                            new_fireflies[i] = np.clip(new_fireflies[i], lb, ub)
                        else:
                            new_fireflies[i] = (np.random.rand(dims) < 1 / (1 + np.exp(-new_fireflies[i]))).astype(float)
                
                if not moved:
                    new_fireflies[i] += alpha * (np.random.rand(dims) - 0.5) * 2.0
                    if flag:
                        new_fireflies[i] = np.clip(new_fireflies[i], lb, ub)
                    else:
                        new_fireflies[i] = (np.random.rand(dims) < 1 / (1 + np.exp(-new_fireflies[i]))).astype(float)
            
            fireflies = new_fireflies
            fitness = np.array([problem.evaluate(f) for f in fireflies])
            
            iter_best_idx = np.argmin(fitness) if flag else np.argmax(fitness)
            iter_best = fitness[iter_best_idx]
            
            if (flag and iter_best < best_cost) or (not flag and iter_best > best_cost):
                best_cost, best_position = iter_best, fireflies[iter_best_idx].copy()
            
            logger.history["iteration_best"].append(float(iter_best))
        
        logger.finish(best_solution=best_position.tolist(), best_fitness=self.calc_fitness(flag, best_cost))
        return best_position, best_cost, logger
