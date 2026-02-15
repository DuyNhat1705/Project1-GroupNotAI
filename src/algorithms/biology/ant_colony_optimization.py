from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import random
import math
import numpy as np


class ACO(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'num_ants': 10, 'iterations': 100, 'alpha': 1.0, 'beta': 2.0,
            'evaporation': 0.5, 'Q': 100, 'initial_pheromone': 1.0,
            'elitist_weight': 2.0, 'archive_size': 10, 'xi': 0.85
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
        return self._solve_continuous(problem, seed) if problem.cont_flag else self._solve_discrete(problem, seed)
    
    def _roulette_select(self, probabilities):
        """Roulette wheel selection"""
        total = sum(probabilities.values())
        if total == 0:
            return random.choice(list(probabilities.keys()))
        rand, cumulative = random.random(), 0.0
        for key, prob in probabilities.items():
            cumulative += prob / total
            if rand <= cumulative:
                return key
        return random.choice(list(probabilities.keys()))
    
    def _solve_continuous(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        dims = problem.dimension
        lb, ub = problem.bounds[:, 0], problem.bounds[:, 1]
        
        # Initialize archive
        archive = [(np.random.uniform(lb, ub, dims), problem.evaluate(np.random.uniform(lb, ub, dims))) 
                   for _ in range(self.archive_size)]
        archive.sort(key=lambda x: x[1])
        best_solution, best_cost = archive[0][0].copy(), archive[0][1]
        
        for iteration in range(self.iterations):
            # Calculate weights
            weights = [math.exp(-(r**2) / (2 * self.xi**2 * self.archive_size**2)) / 
                      (self.archive_size * self.xi * math.sqrt(2 * math.pi)) for r in range(len(archive))]
            total_w = sum(weights)
            weights = [w / total_w for w in weights] if total_w > 0 else [1.0 / len(archive)] * len(archive)
            
            new_solutions = []
            for _ in range(self.num_ants):
                new_sol = np.zeros(dims)
                for d in range(dims):
                    idx = np.random.choice(len(archive), p=weights)
                    std = self.xi * sum(weights[i] * abs(archive[i][0][d] - archive[idx][0][d]) for i in range(len(archive)))
                    std = std if std > 0 else 0.01 * (ub[d] - lb[d])
                    new_sol[d] = np.clip(np.random.normal(archive[idx][0][d], std), lb[d], ub[d])
                
                cost = problem.evaluate(new_sol)
                new_solutions.append((new_sol.copy(), cost))
                if cost < best_cost:
                    best_cost, best_solution = cost, new_sol.copy()
            
            archive = sorted(archive + new_solutions, key=lambda x: x[1])[:self.archive_size]
            logger.history["iteration_best"].append(archive[0][1])
        
        logger.finish(best_solution=best_solution.tolist(), best_fitness=self.calc_fitness(True, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution.tolist(), "best_fitness": self.calc_fitness(True, best_cost), "logger": logger}}
    
    def _solve_discrete(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        if not hasattr(problem, 'dist_mat'):
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"best_solution": [], "cost": float('inf'), "logger": logger}}
        
        n = problem.dimension
        dist_mat = problem.dist_mat
        pheromones = np.ones((n, n)) * self.initial_pheromone
        best_tour, best_cost = None, float('inf')
        
        for iteration in range(self.iterations):
            tours, costs = [], []
            
            for _ in range(self.num_ants):
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
            for tour, cost in zip(tours, costs):
                if cost > 0 and cost != float('inf'):
                    deposit = self.Q / cost
                    for i in range(len(tour) - 1):
                        pheromones[tour[i]][tour[i+1]] += deposit
                        pheromones[tour[i+1]][tour[i]] += deposit
            
            # Elitist
            if best_tour and best_cost > 0 and best_cost != float('inf'):
                elite = (self.elitist_weight * self.Q) / best_cost
                for i in range(len(best_tour) - 1):
                    pheromones[best_tour[i]][best_tour[i+1]] += elite
                    pheromones[best_tour[i+1]][best_tour[i]] += elite
                pheromones[best_tour[-1]][best_tour[0]] += elite
                pheromones[best_tour[0]][best_tour[-1]] += elite
            
            logger.history["iteration_best"].append(min(costs))
        
        logger.finish(best_solution=best_tour, best_fitness=self.calc_fitness(False, best_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_tour, "cost": best_cost, "logger": logger}}
