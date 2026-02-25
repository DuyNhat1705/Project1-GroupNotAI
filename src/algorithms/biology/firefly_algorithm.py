from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
import random
import math


class FireflyAlgorithm(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'pop_size': 50, 'num_iters': 100, 'alpha': 0.5,
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
        
        if hasattr(problem, 'cont_flag') and problem.cont_flag:
            return self._solve_continuous(problem, seed)
        elif hasattr(problem, 'dist_mat'):
            return self._solve_tsp(problem, seed)
        else:
            return self._solve_continuous(problem, seed)
    
    def _solve_continuous(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []  # For visualization
        logger.history["best_fitness"] = []
        logger.history["avg_fitness"] = []
        
        dims = problem.dimension
        flag = problem.cont_flag
        if flag:
            bounds = np.array(problem.bounds)
            lb, ub = bounds[:, 0], bounds[:, 1]
            diameter = np.linalg.norm(ub - lb)
            effective_gamma = self.gamma / (diameter ** 2) if diameter > 0 else self.gamma
        else:
            lb, ub = 0, 2
            effective_gamma = self.gamma
        
        fireflies = np.random.uniform(lb, ub, (self.pop_size, dims)) if flag else \
                    np.random.randint(0, 2, (self.pop_size, dims)).astype(float)
        
        fitness = np.array([problem.evaluate(f) for f in fireflies])
        best_idx = np.argmin(fitness) if flag else np.argmax(fitness)
        best_position, best_cost = fireflies[best_idx].copy(), fitness[best_idx]
        
        for iteration in range(self.num_iters):
            alpha = self.alpha * (0.97 ** iteration)
            new_fireflies = fireflies.copy()
            
            for i in range(self.pop_size):
                moved = False
                origin = fireflies[i].copy()
                # Find single brightest attractor
                best_j, best_beta = -1, -1.0
                for j in range(self.pop_size):
                    is_brighter = (i != j and ((fitness[j] < fitness[i]) if flag else (fitness[j] > fitness[i])))
                    if is_brighter:
                        r = np.linalg.norm(origin - fireflies[j])
                        beta = self.beta_0 * math.exp(-effective_gamma * r**2)
                        if beta > best_beta:
                            best_beta, best_j = beta, j
                
                # Move toward single best attractor
                if best_j >= 0:
                    r = np.linalg.norm(origin - fireflies[best_j])
                    beta = self.beta_0 * math.exp(-effective_gamma * r**2)
                    new_fireflies[i] = origin + beta * (fireflies[best_j] - origin) + alpha * (np.random.rand(dims) - 0.5)
                    if flag:
                        new_fireflies[i] = np.clip(new_fireflies[i], lb, ub)
                    else:
                        new_fireflies[i] = (np.random.rand(dims) < 1 / (1 + np.exp(-new_fireflies[i]))).astype(float)
                    moved = True
                
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
            
            # Log metrics for convergence visualization (every iteration)
            avg_fitness = np.mean(fitness)
            logger.history["best_fitness"].append(best_cost)
            logger.history["avg_fitness"].append(avg_fitness)
            # Log population for visualization
            logger.history["population"].append(fireflies.copy())
        
        logger.finish(best_solution=best_position.tolist(), best_fitness=self.calc_fitness(flag, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_position.tolist(), "best_fitness": self.calc_fitness(flag, best_cost), "logger": logger}}
    
    def _solve_tsp(self, problem, seed):
        """FA for TSP - order-based encoding"""
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        n = problem.dimension
        tsp_diameter = math.sqrt(n)  # TSP positions in [0,1]^n
        effective_gamma = self.gamma / (tsp_diameter ** 2) if tsp_diameter > 0 else self.gamma
        
        fireflies = np.random.uniform(0, 1, (self.pop_size, n))
        
        def pos_to_tour(pos):
            return np.argsort(pos)
        
        fitness = np.array([problem.evaluate(pos_to_tour(f)) for f in fireflies])
        best_idx = np.argmin(fitness)
        best_firefly = fireflies[best_idx].copy()
        best_cost = fitness[best_idx]
        
        for iteration in range(self.num_iters):
            alpha = self.alpha * (0.97 ** iteration)
            new_fireflies = fireflies.copy()
            
            for i in range(self.pop_size):
                moved = False
                origin = fireflies[i].copy()
                # Find single brightest TSP attractor
                best_j, best_beta = -1, -1.0
                for j in range(self.pop_size):
                    if i != j and fitness[j] < fitness[i]:
                        r = np.linalg.norm(origin - fireflies[j])
                        beta = self.beta_0 * math.exp(-effective_gamma * r**2) 
                        if beta > best_beta:
                            best_beta, best_j = beta, j
                
                if best_j >= 0:
                    r = np.linalg.norm(origin - fireflies[best_j])
                    beta = self.beta_0 * math.exp(-effective_gamma * r**2) 
                    new_fireflies[i] = origin + beta * (fireflies[best_j] - origin) + alpha * (np.random.rand(n) - 0.5)
                    new_fireflies[i] = np.clip(new_fireflies[i], 0, 1)
                    moved = True
                
                if not moved:
                    new_fireflies[i] = origin + alpha * (np.random.rand(n) - 0.5) * 2.0
                    new_fireflies[i] = np.clip(new_fireflies[i], 0, 1)
            
            fireflies = new_fireflies
            fitness = np.array([problem.evaluate(pos_to_tour(f)) for f in fireflies])
            
            iter_best_idx = np.argmin(fitness)
            iter_best = fitness[iter_best_idx]
            
            if iter_best < best_cost:
                best_cost = iter_best
                best_firefly = fireflies[iter_best_idx].copy()
            
            # Log every iteration for TSP convergence tracking
            best_tour_iter = pos_to_tour(best_firefly)
            logger.history["iteration_best"].append((best_tour_iter.copy(), best_cost))
        
        logger.history["explored"] = logger.history["iteration_best"]
        
        best_tour = pos_to_tour(best_firefly)
        logger.finish(best_solution=best_tour, best_fitness=best_cost)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_tour.tolist(), "cost": best_cost, "logger": logger}}