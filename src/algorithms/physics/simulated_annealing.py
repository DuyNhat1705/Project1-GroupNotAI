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
        default_params = {"temperature": 100, "decay": 0.995, "step": 1.0, "num_iters": 100}
        if params:
            if "temperature" in params: params["temperature"] = float(params["temperature"])
            if "decay" in params: params["decay"] = float(params["decay"])
            if "step" in params: params["step"] = float(params["step"])
            default_params.update(params)
        # Pass the merged dict to BaseAlgorithm
        super().__init__("Simulated Annealing", default_params)
        for key, val in default_params.items():
            setattr(self, key, val) #pass arguments from terminal


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

        # assign initial temperature
        temp = self.params["temperature"]

        # log best evaluation as list
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []  # explored point

        # logger.history["explored"].append(([cur.copy(), best.copy()], best_fit))

        logger.history["best_fitness"] = []  # best evaluation

        for ite in range(self.params["num_iters"]):
            neigh = cur.copy() # extract used colors

            # Pick a random node
            idx = np.random.randint(dim)

            # Pick a random color (Bound to max used + 1)
            max_color = min(dim - 1, np.max(neigh) + 1)
            available_colors = [c for c in range(max_color + 1) if c != neigh[idx]] # differ from current color of chosen node

            if available_colors:
                neigh[idx] = np.random.choice(available_colors) # random the new color
            else:
                neigh[idx] = 0

            next_fit = problem.evaluate(neigh)

            # Minimize acceptance
            delta = next_fit - cur_fit
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                cur = neigh
                cur_fit = next_fit

            if cur_fit < best_fit:
                best = cur.copy()
                best_fit = cur_fit

            logger.log("best_fitness", best_fit)
            # Log 'cur'
            logger.history["current_best"].append(cur.copy())

            temp *= self.params["decay"]

        logger.finish(best_solution=best, best_fitness=best_fit)
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {"path": best.tolist(), "best_solution": best.tolist(), "best_fitness": best_fit,
                       "logger": logger}
        }

    def _solve_tsp(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # TSP initialization (random path 0 to N-1)
        cur = np.random.permutation(problem.dimension)

        cur_fit = problem.evaluate(cur)

        # record the best solution
        best = cur.copy()
        best_fit = cur_fit

        # assign initial temperature
        temp = self.params["temperature"]

        # log best evalutaion as list
        logger = Logger(self.name, run_id=seed)
        logger.history["explored"] = []  # explored point

        # logger.history["explored"].append(([cur.copy(), best.copy()], best_fit))

        logger.history["best_fitness"] = []  # best evaluation

        def get_neighbor_tsp(cur):
            neigh = cur.copy()
            n = len(neigh)

            # Pick two random indices and swap them
            i, j = random.sample(range(n), 2)
            neigh[i], neigh[j] = neigh[j], neigh[i]

            return neigh

        for ite in range(self.params["num_iters"]):

            # exploring at each num_iters
            next_pos = get_neighbor_tsp(cur)
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

            temp *= self.params["decay"]  # decrease temperature by decay factor

        logger.finish(best_solution=best, best_fitness=best_fit)  # log the best and terminate

        # Return standard dictionary format
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best,
                "best_fitness": best_fit,
                "logger": logger
            }
        }

    def _solve_cont(self, problem, seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if problem.cont_flag:
            # assign bounds and start with a random position
            bounds = np.array(problem.bounds)
            lower_bound = bounds[:, 0]
            upper_bound = bounds[:, 1]
            cur = np.random.uniform(lower_bound, upper_bound, size=problem.dimension)

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

        def get_neighbor_cont(cur_pos, lower, upper):

            step = self.params["step"]

            # noise: exploring by take small step from current position
            noise = np.random.normal(0, step, size=cur_pos.shape)
            neigh = cur_pos + noise

            # ensure the explored neighbors lay within bounds by clipping outrange value
            neigh = np.clip(neigh, lower, upper)

            return neigh

        for ite in range(self.params["num_iters"]):

            # exploring at each num_iters
            next_pos = get_neighbor_cont(cur, lower_bound, upper_bound)
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