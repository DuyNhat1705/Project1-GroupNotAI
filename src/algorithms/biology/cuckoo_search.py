from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
from math import gamma
import numpy as np
class CS(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'pop_size': 25, 'pa': 0.25, 'alpha': 0.01, 'beta': 1.5,
            'num_iters': 100, 'seed': None
        }
        if params:
            default_params.update(params)
        super().__init__("Cuckoo Search", default_params)
        for key, val in default_params.items():
            setattr(self, key, val)

    def simple_bounds(self, s, Lb, Ub):
        s = np.maximum(s, Lb)
        s = np.minimum(s, Ub)
        return s

    # Get best nest
    def get_best_nest(self,nest, new_nest, fitness, problem):
        n = nest.shape[0]

        for j in range(n):
            fnew = problem.evaluate(new_nest[j])
            if fnew <= fitness[j]:
                fitness[j] = fnew
                nest[j] = new_nest[j].copy()

        best_idx = np.argmin(fitness)
        fmin = fitness[best_idx]
        best = nest[best_idx].copy()

        return fmin, best, nest, fitness
    
    # Levy flight phase
    def get_cuckoos(self,nest, best, Lb, Ub):
        n, dim = nest.shape
        beta = self.params.get("beta", 1.5)

        sigma = (
            gamma(1 + beta) * np.sin(np.pi * beta / 2) /
            (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        new_nest = np.zeros_like(nest)
        alpha = self.params.get("alpha", 0.01)

        for j in range(n):
            s = nest[j].copy()

            # Levy flight (Mantegna)
            u = np.random.randn(dim) * sigma
            v = np.random.randn(dim)
            step = u / (np.abs(v) ** (1 / beta))

            stepsize = alpha * step * (s - best)

            s = s + stepsize * np.random.randn(dim)

            new_nest[j] = self.simple_bounds(s, Lb, Ub)

        return new_nest

    # Abandon nests
    def empty_nests(self,nest, Lb, Ub, pa):
        n, dim = nest.shape

        K = np.random.rand(n, dim) > pa

        perm1 = np.random.permutation(n)
        perm2 = np.random.permutation(n)

        stepsize = np.random.rand() * (nest[perm1] - nest[perm2])

        new_nest = nest + stepsize * K

        for j in range(n):
            new_nest[j] = self.simple_bounds(new_nest[j], Lb, Ub)

        return new_nest

    # Main Cuckoo Search
    def solve(self, problem, seed=None):
        dim = problem.dimension
        pa = self.params.get("pa", 0.25)
        num_iters = self.params.get("num_iters", 100)
        n = self.params.get("pop_size", 25)


        Lb = self.params.get("lb", -5) * np.ones(dim)
        Ub = self.params.get("ub", 5) * np.ones(dim)

        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []
        logger.history["best_fitness"] = []

        nest = Lb + (Ub - Lb) * np.random.rand(n, dim)
        fitness = np.ones(n) * 1e10

        fmin, bestnest, nest, fitness = self.get_best_nest(nest, nest, fitness,problem)
        logger.log("best_fitness", fmin)
        logger.log("population", nest.copy())

        for i in range(num_iters):

            # Levy flight
            new_nest = self.get_cuckoos(nest, bestnest, Lb, Ub)
            fnew, best, nest, fitness = self.get_best_nest(nest, new_nest, fitness,problem)

            # Abandon phase
            new_nest = self.empty_nests(nest, Lb, Ub, pa)
            fnew, best, nest, fitness = self.get_best_nest(nest, new_nest, fitness,problem)

            if fnew < fmin:
                fmin = fnew
                bestnest = best.copy()
            
            logger.log("best_fitness", fmin)
            logger.log("population", nest.copy())

        logger.finish(bestnest, fmin)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": bestnest, "best_fitness": fmin, "logger": logger}}