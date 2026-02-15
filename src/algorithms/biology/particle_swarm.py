from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
import random


class PSO(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'swarm_size': 30, 'iterations': 100, 'w_max': 0.9, 'w_min': 0.4,
            'c1': 2.0, 'c2': 2.0, 'v_clamp': True
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
        
        logger = Logger(self.name, run_id=seed)
        logger.history["iteration_best"] = []
        
        dims = problem.dimension
        flag = problem.cont_flag
        lb, ub = (problem.bounds[:, 0], problem.bounds[:, 1]) if flag else (0, 2)
        v_max = 0.2 * (ub - lb) if (self.v_clamp and flag) else (0.4 if self.v_clamp else None)
        
        # Initialize swarm
        positions = np.random.uniform(lb, ub, (self.swarm_size, dims)) if flag else \
                    np.random.randint(0, 2, (self.swarm_size, dims)).astype(float)
        velocities = np.random.uniform(-1, 1, (self.swarm_size, dims))
        
        # Initialize personal and global best
        pbest_pos = positions.copy()
        pbest_cost = np.array([problem.evaluate(p) for p in positions])
        gbest_idx = np.argmin(pbest_cost) if flag else np.argmax(pbest_cost)
        gbest_pos, gbest_cost = pbest_pos[gbest_idx].copy(), pbest_cost[gbest_idx]
        
        for iteration in range(self.iterations):
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.iterations
            iter_costs = []
            
            for i in range(self.swarm_size):
                # Update velocity and position
                r1, r2 = np.random.random(dims), np.random.random(dims)
                velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - positions[i]) + \
                                self.c2 * r2 * (gbest_pos - positions[i])
                
                if v_max is not None:
                    velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                positions[i] += velocities[i]
                
                # Apply bounds
                if flag:
                    positions[i] = np.clip(positions[i], lb, ub)
                else:
                    positions[i] = (np.random.rand(dims) < 1 / (1 + np.exp(-positions[i]))).astype(float)
                
                # Evaluate and update best
                cost = problem.evaluate(positions[i])
                iter_costs.append(cost)
                
                if (flag and cost < pbest_cost[i]) or (not flag and cost > pbest_cost[i]):
                    pbest_cost[i] = cost
                    pbest_pos[i] = positions[i].copy()
                    
                    if (flag and cost < gbest_cost) or (not flag and cost > gbest_cost):
                        gbest_cost = cost
                        gbest_pos = positions[i].copy()
            
            logger.history["iteration_best"].append(min(iter_costs) if flag else max(iter_costs))
        
        logger.finish(best_solution=gbest_pos.tolist(), best_fitness=self.calc_fitness(flag, gbest_cost))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": gbest_pos.tolist(), "best_fitness": self.calc_fitness(flag, gbest_cost), "logger": logger}}
