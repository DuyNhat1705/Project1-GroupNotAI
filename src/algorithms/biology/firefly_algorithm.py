from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
import random
import math


class FireflyAlgorithm(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'swarm_size': 50, 'iterations': 100, 'alpha': 0.5,
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
        else:
            # Discrete (TSP or other)
            return self._solve_discrete(problem, seed)
    
    def _solve_continuous(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []  # For visualization
        logger.history["best_fitness"] = []
        logger.history["avg_fitness"] = []
        
        dims = problem.dimension
        bounds = np.array(problem.bounds)
        lb, ub = bounds[:, 0], bounds[:, 1]
        diameter = np.linalg.norm(ub - lb)
        effective_gamma = self.gamma / (diameter ** 2) if diameter > 0 else self.gamma
        
        fireflies = np.random.uniform(lb, ub, (self.swarm_size, dims))
        
        fitness = np.array([problem.evaluate(f) for f in fireflies])
        best_idx = np.argmin(fitness)
        best_position, best_cost = fireflies[best_idx].copy(), fitness[best_idx]
        
        for iteration in range(self.iterations):
            alpha = self.alpha * (0.97 ** iteration)
            new_fireflies = fireflies.copy()
            
            for i in range(self.swarm_size):
                moved = False
                origin = fireflies[i].copy()
                # Find single brightest attractor
                best_j, best_beta = -1, -1.0
                for j in range(self.swarm_size):
                    if i != j and fitness[j] < fitness[i]:
                        r = np.linalg.norm(origin - fireflies[j])
                        beta = self.beta_0 * math.exp(-effective_gamma * r**2)
                        if beta > best_beta:
                            best_beta, best_j = beta, j
                
                # Move toward single best attractor
                if best_j >= 0:
                    r = np.linalg.norm(origin - fireflies[best_j])
                    beta = self.beta_0 * math.exp(-effective_gamma * r**2)
                    new_fireflies[i] = origin + beta * (fireflies[best_j] - origin) + alpha * (np.random.rand(dims) - 0.5)
                    new_fireflies[i] = np.clip(new_fireflies[i], lb, ub)
                    moved = True
                
                if not moved:
                    new_fireflies[i] += alpha * (np.random.rand(dims) - 0.5) * 2.0
                    new_fireflies[i] = np.clip(new_fireflies[i], lb, ub)
            
            fireflies = new_fireflies
            fitness = np.array([problem.evaluate(f) for f in fireflies])
            
            iter_best_idx = np.argmin(fitness)
            iter_best = fitness[iter_best_idx]
            
            if iter_best < best_cost:
                best_cost, best_position = iter_best, fireflies[iter_best_idx].copy()
            
            # Log metrics for convergence visualization (every iteration)
            avg_fitness = np.mean(fitness)
            logger.history["best_fitness"].append(best_cost)
            logger.history["avg_fitness"].append(avg_fitness)
            # Log population for visualization
            logger.history["population"].append(fireflies.copy())
        
        logger.finish(best_solution=best_position.tolist(), best_fitness=self.calc_fitness(True, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_position.tolist(), "best_fitness": self.calc_fitness(True, best_cost), "logger": logger}}
    
    def _solve_discrete(self, problem, seed):
        """Discrete solver entry point. Currently supports: TSP (requires dist_mat)."""
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []

        if not hasattr(problem, 'dist_mat'):
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"best_solution": [], "cost": float('inf'), "logger": logger}}
        
        n = problem.dimension
        tsp_diameter = math.sqrt(n)  # TSP positions in [0,1]^n
        effective_gamma = self.gamma / (tsp_diameter ** 2) if tsp_diameter > 0 else self.gamma
        
        fireflies = np.random.uniform(0, 1, (self.swarm_size, n))
        
        def pos_to_tour(pos):
            return np.argsort(pos)
        
        fitness = np.array([problem.evaluate(pos_to_tour(f)) for f in fireflies])
        best_idx = np.argmin(fitness)
        best_firefly = fireflies[best_idx].copy()
        best_cost = fitness[best_idx]
        
        for iteration in range(self.iterations):
            alpha = self.alpha * (0.97 ** iteration)
            new_fireflies = fireflies.copy()
            
            for i in range(self.swarm_size):
                moved = False
                origin = fireflies[i].copy()
                # Find single brightest TSP attractor
                best_j, best_beta = -1, -1.0
                for j in range(self.swarm_size):
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