import numpy as np
import random
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class HillClimbing(BaseAlgorithm):
    def __init__(self, params={"step": 0.99, "iteration": 600}):
        """"
        step: max step size from current position
        iteration: number of iterations
        """
        super().__init__("Hill Climbing", params)

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
        # Ensure repeatable runs if seed is provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if problem.cont_flag:
            # assign bounds and start with a random position
            bounds = problem.bounds
            lower_bound = bounds[:, 0]
            upper_bound = bounds[:, 1]
            cur = np.random.uniform(lower_bound, upper_bound, size=problem.dimension)

        else:
            # TSP initialization (random path 0 to N-1)
            cur = np.random.permutation(problem.dimension)
            # Unused bounds are set to None
            lower_bound, upper_bound = None, None

        cur_fit = problem.evaluate(cur)

        logger = Logger(self.name, run_id=seed)
        # log the position if move occurs
        logger.history["explored"] = []
        # Log starting point
        logger.history["explored"].append((cur, cur_fit))

        for ite in range(self.params["iteration"]):
            next_pos = self.get_neighbor(cur, lower_bound, upper_bound, problem.cont_flag)
            next_fit = problem.evaluate(next_pos)

            # Hill Climbing Logic: Only move if better
            if next_fit < cur_fit:
                cur_fit = next_fit  # update position and evaluation
                cur = next_pos
                # log new postion
                logger.history["explored"].append((cur, cur_fit))

        # log the solution and best score
        logger.finish(best_solution=cur, best_fitness=cur_fit)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": cur, "best_fitness": cur_fit, "logger": logger}}