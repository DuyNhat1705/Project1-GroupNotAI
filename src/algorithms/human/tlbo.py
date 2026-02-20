from venv import logger
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
class TLBO(BaseAlgorithm):
    def __init__(self, params = None):
        super().__init__("TLBO",params)
        """params (dictionary) contains: population_size, num_iterations, num_variables, lb, ub """
    
    def solve(self, problem, seed=None):
        # Maximize fitness
        population_size = self.params.get("pop_size", 50)
        num_iterations = self.params.get("num_iters", 100)
        lb = self.params.get("lb", -10)
        ub = self.params.get("ub", 10)
        num_variables = self.params.get("num_variables", 2)

        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []
        logger.history["best_fitness"] = []   
        # Initialize randomly for each student in poplation with
        population = np.random.uniform(lb, ub, size=(population_size, num_variables))
        fitness = np.array([problem.evaluate(ind) for ind in population])
        logger.log("population", population.copy())
        logger.log("best_fitness", np.min(fitness))

        # Find the best solution for the 1st teacher phase
        for iteration in range(num_iterations):
            for student in range(population_size):
                #Teacher phase
                best_idx = np.argmin(fitness)
                X_best = population[best_idx]
                best_fitness = fitness[best_idx]

                X_mean = np.mean(population, axis=0)
                r = np.random.rand(2)
                Tf = np.random.randint(1,3)

                X_new = population[student] + r*(X_best - Tf*X_mean)
                X_new = np.clip(X_new, lb, ub)
                new_fitness = problem.evaluate(X_new)
                if new_fitness < fitness[student]:
                    population[student] = X_new
                    fitness[student] = new_fitness
                
                #Learner phase
                partner_idx = np.random.randint(population_size)
                while partner_idx == student:
                    partner_idx = np.random.randint(population_size)

                X_partner = population[partner_idx]
                partner_fitness = fitness[partner_idx]
                r = np.random.rand(2)

                if fitness[student] < partner_fitness:
                    X_new = population[student] + r*(population[student]-X_partner)
                else:
                    X_new = population[student] - r*(population[student]-X_partner)

                X_new = np.clip(X_new, lb, ub)
                new_fitness = problem.evaluate(X_new)
                if new_fitness < fitness[student]:
                    population[student] = X_new
                    fitness[student] = new_fitness
            logger.log("best_fitness", np.min(fitness))
            logger.log("population", population.copy())

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        logger.finish(best_solution, best_fitness)

        return {"time(ms)": logger.meta["runtime"], "result":{"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}

