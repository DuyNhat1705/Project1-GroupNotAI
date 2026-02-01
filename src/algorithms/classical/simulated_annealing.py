import numpy as np
import math
import random
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, params={"temperature": 10, "decay": 0.95, "step": 0.1, "iteration": 600}):
        """
        temperature: the initial temperature
        decay: decay rate, decide how the temperature decrease
        step: step size from current position
        iteration: number of iterations
        """
        super().__init__("Simulated Annealing", params)

    def get_neighbor(self, cur_pos, lower, upper):
        step = self.params["step"]

        # noise: exploring by take small step from current position
        noise = np.random.normal(0, step, size=cur_pos.shape)
        neigh = cur_pos + noise

        # ensure the explored neighbors lay within bounds by clipping outrange value
        neigh = np.clip(neigh, lower, upper)
        return neigh

    def solve(self, problem, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # assign bounds and start with a random position
        bounds = problem.bounds
        cur = np.random.uniform(bounds[0], bounds[1], size=problem.dimension)
        cur_fit = problem.evaluate(cur)

        # record the best solution
        best = cur.copy()
        best_fit = cur_fit

        # assign initial temperature
        temp = self.params["temperature"]

        # log best evalutaion as list
        logger = Logger(self.name, run_id=seed)
        logger.history["explored"] = []
        logger.history["explored"].append((cur, cur_fit))

        for ite in range(self.params["iteration"]):

            # exploring at each iteration
            next_pos = self.get_neighbor(cur, bounds[0], bounds[1])
            next_fit = problem.evaluate(next_pos)

            # Accept if better evaluation OR by chance
            # (probability depends on temperature)
            delta = next_fit - cur_fit
            if delta < 0 or random.random() < math.exp(-delta / temp):
                cur = next_pos
                cur_fit = next_fit
                # Log the current position
                logger.history["explored"].append((cur, cur_fit))

            # Update Best
            if cur_fit < best_fit:
                best = cur.copy()
                best_fit = cur_fit

            temp *= self.params["decay"] # decrease temperature by decay factor

        logger.finish(best_solution=best, best_fitness=best_fit) # log the best and terminate
        return best, best_fit, logger