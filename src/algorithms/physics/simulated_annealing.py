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
        num_iters: number of num_iters
        """
        default_params = {"temperature": 50, "decay": 0.98, "step": 1.5, "num_iters": 300}
        if params:
            default_params.update(params)
        # Pass the merged dict to BaseAlgorithm
        super().__init__("Simulated Annealing", default_params)
        for key, val in default_params.items():
            setattr(self, key, val) #pass arguments from terminal

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
            bounds = [None, None] #initialize to prevent later error

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
        logger.history["explored"] = [] # explored point

        #logger.history["explored"].append(([cur.copy(), best.copy()], best_fit))

        logger.history["best_fitness"] = [] # best evaluation


        for ite in range(self.params["num_iters"]):

            # exploring at each num_iters
            next_pos = self.get_neighbor(cur, lower_bound, upper_bound, problem.cont_flag)
            next_fit = problem.evaluate(next_pos)

            # Accept if better evaluation OR by chance
            # (probability depends on temperature)
            delta = next_fit - cur_fit
            if delta < 0 or random.random() < math.exp(-delta / temp):
                cur = next_pos
                cur_fit = next_fit

            # Update Best
            if cur_fit < best_fit:
                best = cur.copy()
                best_fit = cur_fit

            logger.log("best_fitness", best_fit)
            logger.history["explored"].append(([cur.copy(), best.copy()], best_fit))

            temp *= self.params["decay"] # decrease temperature by decay factor

        logger.finish(best_solution=best, best_fitness=best_fit) # log the best and terminate

        # Return standard dictionary format
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best,
                "best_fitness": best_fit,
                "logger": logger
            }
        }