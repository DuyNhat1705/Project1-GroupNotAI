from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import random
import math
import numpy as np


class ACO(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'pop_size': 10, 'num_iters': 100, 'alpha': 1.0, 'beta': 2.0,
            'evaporation': 0.5, 'Q': 100, 'archive_size': 50, 'xi': 0.85
        }
        if params:
            default_params.update(params)
        super().__init__("Ant Colony Optimization", default_params)
        for key, val in default_params.items():
            setattr(self, key, val)

    def calc_fitness(self, flag, cost):
        if flag:
            return 1 / (cost + 1) if cost >= 0 else 1 + np.abs(cost)
        return 1 / cost if cost > 0 else float('inf')

    def solve(self, problem, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Check problem type
        if hasattr(problem, 'cont_flag') and problem.cont_flag:
            return self._solve_continuous(problem, seed)
        else:
            # Discrete (TSP or other)
            return self._solve_discrete(problem, seed)
    
    def _roulette_select(self, probabilities):
        """Roulette wheel selection using numpy vectorized approach for accuracy and speed"""
        cities = np.array(list(probabilities.keys()))
        probs = np.array(list(probabilities.values()))
        total = probs.sum()
        if total <= 0:
            return int(np.random.choice(cities))
        probs = probs / total  # Normalize to sum to 1.0 (eliminates floating-point bias)
        return int(np.random.choice(cities, p=probs))
    
    def _solve_continuous(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["best_fitness"] = []
        logger.history["avg_fitness"] = []
        logger.history["population"] = []  # For visualization
        
        dims = problem.dimension
        bounds = np.array(problem.bounds)
        lb, ub = bounds[:, 0], bounds[:, 1]
        
        sols = [np.random.uniform(lb, ub, dims) for _ in range(self.archive_size)]
        archive = [(s, problem.evaluate(s)) for s in sols]
        archive.sort(key=lambda x: x[1])
        best_solution, best_cost = archive[0][0].copy(), archive[0][1]
        
        for num_iters in range(self.num_iters):
            # Calculate weights
            weights = [math.exp(-(r**2) / (2 * self.xi**2 * self.archive_size**2)) / 
                      (self.archive_size * self.xi * math.sqrt(2 * math.pi)) for r in range(len(archive))]
            total_w = sum(weights)
            weights = [w / total_w for w in weights] if total_w > 0 else [1.0 / len(archive)] * len(archive)
            
            new_solutions = []
            for _ in range(self.pop_size):
                new_sol = np.zeros(dims)
                for d in range(dims):
                    idx = np.random.choice(len(archive), p=weights)
                    std = self.xi * sum(weights[i] * abs(archive[i][0][d] - archive[idx][0][d]) for i in range(len(archive)))
                    std = max(std, 0.01 * (ub[d] - lb[d]))  
                    new_sol[d] = np.clip(np.random.normal(archive[idx][0][d], std), lb[d], ub[d])
                
                cost = problem.evaluate(new_sol)
                new_solutions.append((new_sol.copy(), cost))
                if cost < best_cost:
                    best_cost, best_solution = cost, new_sol.copy()
            
            archive = sorted(archive + new_solutions, key=lambda x: x[1])[:self.archive_size]
            current_best = archive[0][1]
            current_avg = np.mean([cost for _, cost in archive])
            
            logger.history["best_fitness"].append(current_best)
            logger.history["avg_fitness"].append(current_avg)
            # Log archive for visualization
            logger.history["population"].append([sol for sol, _ in archive])
        
        logger.finish(best_solution=best_solution.tolist(), best_fitness=self.calc_fitness(True, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution.tolist(), "best_fitness": self.calc_fitness(True, best_cost), "logger": logger}}
    
    def _solve_discrete(self, problem, seed):
        """Discrete solver entry point. Currently supports: TSP (requires dist_mat)."""
        logger = Logger(self.name, run_id=seed)
        logger.history["num_iters_best"] = []
        
        if not hasattr(problem, 'dist_mat'):
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"best_solution": [], "cost": float('inf'), "logger": logger}}
        
        n = problem.dimension
        dist_mat = problem.dist_mat
        pheromones = np.ones((n, n))
        best_tour, best_cost = None, float('inf')
        
        for iteration in range(self.num_iters):
            tours, costs = [], []
            
            for _ in range(self.pop_size):
                # Construct tour
                current = random.randint(0, n - 1)
                tour, unvisited = [current], set(range(n)) - {current}
                
                while unvisited:
                    probs = {}
                    for city in unvisited:
                        dist = dist_mat[current][city] if dist_mat[current][city] > 0 else 0.001
                        probs[city] = (pheromones[current][city] ** self.alpha) * ((1.0 / dist) ** self.beta)
                    next_city = self._roulette_select(probs)
                    tour.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city
                
                cost = problem.evaluate(tour)
                tours.append(tour)
                costs.append(cost)
                if cost < best_cost:
                    best_cost, best_tour = cost, tour[:]
            
            # Update pheromones
            pheromones *= (1 - self.evaporation)
            for ant_tour, ant_cost in zip(tours, costs):
                if ant_cost > 0 and ant_cost != float('inf'):
                    deposit = self.Q / ant_cost
                    for i in range(len(ant_tour) - 1):
                        pheromones[ant_tour[i]][ant_tour[i+1]] += deposit
                        pheromones[ant_tour[i+1]][ant_tour[i]] += deposit
                    pheromones[ant_tour[-1]][ant_tour[0]] += deposit
                    pheromones[ant_tour[0]][ant_tour[-1]] += deposit
            
            # Log every iteration for TSP convergence tracking
            logger.history["iteration_best"].append((best_tour[:] if best_tour else [], best_cost))
        
        logger.history["explored"] = logger.history["iteration_best"]
        
        logger.finish(best_solution=best_tour, best_fitness=self.calc_fitness(False, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_tour, "cost": best_cost, "logger": logger}}

