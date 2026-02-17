import numpy as np
import math
import random
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, params=None):
        """
        temperature: the initial temperature
        decay: decay rate, decide how the temperature decrease
        step: step size from current position
        iteration: number of iterations
        """
        default_params = {"temperature": 10, "decay": 0.95, "step": 0.1, "iteration": 600}
        if params:
            default_params.update(params)
        # Pass the merged dict to BaseAlgorithm
        super().__init__("Simulated Annealing", default_params)

    def get_neighbor(self, cur_pos, lower, upper, cont_flag):

        if cont_flag:
            step = self.params["step"]

            # noise: exploring by take small step from current position
            noise = np.random.normal(0, step, size=cur_pos.shape)
            neigh = cur_pos + noise

            # ensure the explored neighbors lay within bounds by clipping outrange value
            neigh = np.clip(neigh, lower, upper)

            return neigh

        else:
            # Create a copy, do not change the current postion
            bounds = [None, None]

            neigh = cur_pos.copy()
            n = len(neigh)

            # Pick two random indices and swap them
            i, j = random.sample(range(n), 2)
            neigh[i], neigh[j] = neigh[j], neigh[i]

            return neigh

    def solve(self, problem, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if problem.cont_flag:
            # assign bounds and start with a random position
            bounds = np.array(problem.bounds)
            lower_bound = bounds[:, 0]
            upper_bound = bounds[:, 1]
            cur = np.random.uniform(lower_bound, upper_bound, size=problem.dimension)

        else:
            # TSP initialization (random path 0 to N-1)
            cur = np.random.permutation(problem.dimension)
            # Unused bounds are set to None
            lower_bound, upper_bound = None, None

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
            next_pos = self.get_neighbor(cur, lower_bound, upper_bound, problem.cont_flag)
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
        #return best, best_fit, logger
        logger.finish(best_solution=best, best_fitness=best_fit)

        # Return standard dictionary format
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best,
                "best_fitness": best_fit,
                "logger": logger
            }
        }