from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
from math import gamma
import numpy as np
class CS(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            'pop_size': 50, 'pa': 0.25, 'alpha': 0.01, 'beta': 1.5,
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
    def solveContinuous(self, problem, seed=None):
        dim = problem.dimension
        pa = self.params.get("pa", 0.25)
        num_iters = self.params.get("num_iters", 100)
        n = self.params.get("pop_size", 25)


        Lb = problem.min_range * np.ones(dim)
        Ub = problem.max_range * np.ones(dim)

        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []
        logger.history["best_fitness"] = []

        nest = Lb + (Ub - Lb) * np.random.rand(n, dim)
        fitness = np.ones(n) * 1e10

        fmin, bestnest, nest, fitness = self.get_best_nest(nest, nest, fitness,problem)

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
    
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def repair(self,selection,weights, values, capacity):
        total_weight = np.sum(selection * weights)

        if total_weight <= capacity:
            return selection

        ratio = values / weights
        idx_sorted = np.argsort(ratio)

        for idx in idx_sorted:
            if selection[idx] == 1:
                selection[idx] = 0
                total_weight -= weights[idx]
                if total_weight <= capacity:
                    break
        return selection
    def solveKnapsack(self, problem, seed = None):
        
        n = self.params.get("pop_size", 25)
        dim = problem.dimension
        pa = self.params.get("pa", 0.25)
        num_iters = self.params.get("num_iters", 100)

        weights = problem.weights
        values = problem.values
        capacity = problem.capacity
        
        nest = np.random.rand(n, dim)
        fitness = np.zeros(n)
        binary_pop = np.zeros((n,dim))
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []

        # Initialize population
        for i in range (n):
            prob = self.sigmoid(nest[i])
            bin_sol = (np.random.rand(dim) < prob).astype(int)
            bin_sol = self.repair(bin_sol,weights, values, capacity)
            fitness[i] = problem.evaluate(bin_sol)
            binary_pop[i] = bin_sol.copy()
        best_idx = np.argmax(fitness)
        best = nest[best_idx].copy()
        fmax = fitness[best_idx]
        best_binary = binary_pop[best_idx].copy()


        for i in range(num_iters):
            # Levy flight
            new_nest = self.get_cuckoos(nest, best, np.zeros(dim),np.ones(dim))

            for j in range(n):
                prob = self.sigmoid(new_nest[j])
                bin_sol = (np.random.rand(dim) < prob).astype(int)
                bin_sol = self.repair(bin_sol, weights, values, capacity)
                fnew = problem.evaluate(bin_sol)

                if fnew >= fitness[j]:
                    nest[j] = new_nest[j].copy()
                    fitness[j] = fnew
                    binary_pop[j] = bin_sol.copy()

            # Abandon phase
            new_nest = self.empty_nests(nest, np.zeros(dim), np.ones(dim), pa)

            for j in range (n):
                prob = self.sigmoid(new_nest[j])
                bin_sol = (np.random.rand(dim) < prob).astype(int)
                bin_sol = self.repair(bin_sol, weights, values, capacity)
                fnew = problem.evaluate(bin_sol)

                if fnew >= fitness[j]:
                    nest[j] = new_nest[j].copy()
                    fitness[j] = fnew
                    binary_pop[j] = bin_sol.copy()
            
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > fmax:
                fmax = fitness[best_idx]
                best = nest[best_idx].copy()
                best_binary = binary_pop[best_idx].copy()
            logger.log("current_best", best_binary)
            logger.log("best_fitness",fmax)

        logger.finish(best_binary, fmax)
        
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_binary, "best_fitness": fmax, "logger": logger}}

    def solve(self, problem, seed=None):
        if problem.cont_flag:
            return self.solveContinuous(problem, seed)
        else:
            return self.solveKnapsack(problem, seed)