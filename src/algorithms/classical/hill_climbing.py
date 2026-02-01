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

    def get_neighbor(self, cur_pos, lower, upper):
        step = self.params["step"]

        # random small step from current postion
        noise = np.random.normal(0, step, size=cur_pos.shape)
        neigh = cur_pos + noise

        # ensure the explored neighbors lay within bounds by clipping outrange value
        neigh = np.clip(neigh, lower, upper)

        return neigh

    def solve(self, problem, seed=None):
        # Ensure repeatable runs if seed is provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        bounds = problem.bounds

        # Initialize random start
        cur = np.random.uniform(bounds[0], bounds[1], size=problem.dimension)
        cur_fit = problem.evaluate(cur)

        logger = Logger(self.name, run_id=seed)

        # log the position if move occurs
        logger.history["explored"] = []

        # Log starting point
        logger.history["explored"].append((cur, cur_fit))

        for ite in range(self.params["iteration"]):
            next_pos = self.get_neighbor(cur, bounds[0], bounds[1])
            next_fit = problem.evaluate(next_pos)

            # Hill Climbing Logic: Only move if better
            if next_fit < cur_fit:
                cur_fit = next_fit  # update position and evaluation
                cur = next_pos
                # log new postion
                logger.history["explored"].append((cur, cur_fit))

        # log the solution and best score
        logger.finish(best_solution=cur, best_fitness=cur_fit)
        return cur, cur_fit, logger