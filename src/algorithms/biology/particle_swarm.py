from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
import random


class PSO(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'swarm_size': 30, 'iterations': 100,
            'w_max': 0.9, 'w_min': 0.4,
            'c1': 2.05, 'c2': 2.05
        }
        if params:
            default_params.update(params)
        super().__init__("Particle Swarm Optimization", default_params)
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
        
        positions = np.random.uniform(lb, ub, (self.swarm_size, dims))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, dims))
        
        pbest_pos = positions.copy()
        pbest_cost = np.array([problem.evaluate(p) for p in positions])
        gbest_idx = np.argmin(pbest_cost)
        gbest_pos, gbest_cost = pbest_pos[gbest_idx].copy(), pbest_cost[gbest_idx]
        
        for iteration in range(self.iterations):
            w = self.w_max - (self.w_max - self.w_min) * iteration / (self.iterations - 1)
            iter_costs = []
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.random(dims), np.random.random(dims)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - positions[i]) + \
                                self.c2 * r2 * (gbest_pos - positions[i])
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lb, ub)
                
                cost = problem.evaluate(positions[i])
                iter_costs.append(cost)
                
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = positions[i].copy()
                    
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = positions[i].copy()
            
            # Log metrics for convergence visualization (every iteration for better tracking)
            avg_fitness = np.mean(iter_costs)
            logger.history["best_fitness"].append(gbest_cost)
            logger.history["avg_fitness"].append(avg_fitness)
            # Log population for visualization
            logger.history["population"].append(positions.copy())
        
        logger.finish(best_solution=gbest_pos.tolist(), best_fitness=self.calc_fitness(True, gbest_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": gbest_pos.tolist(), "best_fitness": self.calc_fitness(True, gbest_cost), "logger": logger}}
    
    def _solve_discrete(self, problem, seed):
        """Discrete solver entry point. Currently supports: TSP (requires dist_mat)."""
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        if not hasattr(problem, 'dist_mat'):
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"best_solution": [], "cost": float('inf'), "logger": logger}}
        
        n = problem.dimension
        positions = np.random.uniform(0, 1, (self.swarm_size, n))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, n))
        
        def pos_to_tour(pos):
            return np.argsort(pos)
        
        tours = [pos_to_tour(pos) for pos in positions]
        costs = [problem.evaluate(tour) for tour in tours]
        
        pbest_pos = positions.copy()
        pbest_cost = np.array(costs, dtype=float) 
        gbest_idx = np.argmin(pbest_cost)
        gbest_pos = positions[gbest_idx].copy()
        gbest_cost = costs[gbest_idx]
        
        for iteration in range(self.iterations):
            w = self.w_max - (self.w_max - self.w_min) * iteration / (self.iterations - 1)
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.random(n), np.random.random(n)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - positions[i]) + \
                                self.c2 * r2 * (gbest_pos - positions[i])
                velocities[i] = np.clip(velocities[i], -0.4, 0.4)  # TSP stability
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0, 1)
                
                tour = pos_to_tour(positions[i])
                cost = problem.evaluate(tour)
                
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = positions[i].copy()
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = positions[i].copy()
            
            # Log every iteration for TSP convergence tracking
            best_tour_iter = pos_to_tour(gbest_pos)
            logger.history["iteration_best"].append((best_tour_iter.copy(), gbest_cost))
        
        logger.history["population"] = logger.history["iteration_best"]
        
        best_tour = pos_to_tour(gbest_pos)
        logger.finish(best_solution=best_tour, best_fitness=gbest_cost)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_tour.tolist(), "cost": gbest_cost, "logger": logger}}
