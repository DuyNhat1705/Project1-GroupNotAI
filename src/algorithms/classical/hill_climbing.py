import numpy as np
import random
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class HillClimbing(BaseAlgorithm):
    def __init__(self, params=None):
        """"
        step: max step size from current position
        num_iters: number of num_iters
        """
        # parameter initialization
        default_params = {"step": 1.0, "num_iters": 500}
        if params:
            if "step" in params: params["step"] = float(params["step"])
            default_params.update(params)
        super().__init__("Hill Climbing", default_params)

        for key, val in default_params.items():
            setattr(self, key, val)


    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # The Master Router: Direct the problem to the correct SA behavior
        prob_name = getattr(problem, 'name', problem.getName() if hasattr(problem, 'getName') else "").lower()

        if "coloring" in prob_name:
            return self._solve_color(problem, seed)
        elif "tsp" in prob_name:
            return self._solve_tsp(problem, seed)
        elif getattr(problem, 'cont_flag', False):
            return self._solve_cont(problem, seed)
        else:
            return self._solve_tsp(problem, seed)  # Fallback

    def _solve_cont(self, problem, seed=None):
        # Ensure repeatable runs if seed is provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # assign bounds and start with a random position
        bounds = np.array(problem.bounds)
        lower_bound = bounds[:, 0]
        upper_bound = bounds[:, 1]
        cur = np.random.uniform(lower_bound, upper_bound, size=problem.dimension)
        cur_fit = problem.evaluate(cur)

        logger = Logger(self.name, run_id=seed)
        # log the position if move occurs
        logger.history["explored"] = []
        # Log starting point
        logger.history["explored"].append((cur.copy(), cur_fit))
        logger.history["best_fitness"] = []

        def get_neighbor_cont(cur_pos, lower, upper):
            step = self.params["step"]

            # noise: exploring by take small step from current position
            noise = np.random.normal(0, step, size=cur_pos.shape)
            neigh = cur_pos + noise

            # ensure the explored neighbors lay within bounds by clipping outrange value
            neigh = np.clip(neigh, lower, upper)

            return neigh


        for ite in range(self.params["num_iters"]):
            next_pos = get_neighbor_cont(cur, lower_bound, upper_bound)
            next_fit = problem.evaluate(next_pos)

            # Hill Climbing Logic: Only move if better
            if next_fit < cur_fit:
                cur_fit = next_fit  # update position and evaluation
                cur = next_pos

            # log new postion
            logger.history["explored"].append((cur, cur_fit))
            logger.history["best_fitness"].append(cur_fit)

        # log the solution and best score
        logger.finish(best_solution=cur, best_fitness=cur_fit)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": cur, "best_fitness": cur_fit, "logger": logger}}


    def _solve_tsp(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        cur = np.random.permutation(problem.dimension)
        cur_fit = problem.evaluate(cur)
        logger = Logger(self.name, run_id=seed)
        # log the position if move occurs
        logger.history["explored"] = []
        # Log starting point
        logger.history["explored"].append((cur.copy(), cur_fit))
        logger.history["best_fitness"] = []

        def get_neighbor_tsp(cur_pos):
            step = self.params["step"]
            # Create a copy, do not change the current postion
            bounds = [None, None]

            neigh = cur_pos.copy()
            n = len(neigh)

            # Pick two random indices and swap them
            i, j = random.sample(range(n), 2)
            neigh[i], neigh[j] = neigh[j], neigh[i]

            return neigh

        for ite in range(self.params["num_iters"]):
            next_pos = get_neighbor_tsp(cur)
            next_fit = problem.evaluate(next_pos)

            # Hill Climbing Logic: Only move if better
            if next_fit < cur_fit:
                cur_fit = next_fit  # update position and evaluation
                cur = next_pos

            # log new postion
            logger.history["explored"].append((cur, cur_fit))
            logger.history["best_fitness"].append(cur_fit)

        # log the solution and best score
        logger.finish(best_solution=cur, best_fitness=cur_fit)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": cur, "best_fitness": cur_fit, "logger": logger}}

    def _solve_color(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        dim = problem.dimension

        # random color initialization
        cur = np.random.randint(0, dim, size=dim)
        cur_fit = problem.evaluate(cur)

        # record the best solution
        best = cur.copy()
        best_fit = cur_fit


        # log best evalutaion as list
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []  # explored point

        logger.history["best_fitness"] = []  # best evaluation


        for ite in range(self.params["num_iters"]):
            neigh = cur.copy()

            # Pick a random node
            idx = np.random.randint(dim)

            # Pick a random color (Bound to max used + 1 to reduce)
            max_color = min(dim - 1, np.max(neigh) + 1)
            available_colors = [c for c in range(max_color + 1) if c != neigh[idx]]

            if available_colors:
                neigh[idx] = np.random.choice(available_colors)
            else:
                neigh[idx] = 0

            next_fit = problem.evaluate(neigh)

            # Minimize acceptance
            if next_fit < cur_fit:
                cur = neigh
                cur_fit = next_fit

            logger.log("best_fitness", best_fit)
            # Log 'cur' so the visualizer shows the chaotic bouncing of SA
            logger.history["current_best"].append(cur.copy())


        logger.finish(best_solution=best, best_fitness=best_fit)
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {"path": best.tolist(), "best_solution": best.tolist(), "best_fitness": best_fit,
                       "logger": logger}
        }
