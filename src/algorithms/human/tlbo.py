from venv import logger
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np
class TLBO(BaseAlgorithm):
    def __init__(self, params = None):
        super().__init__("TLBO",params)
        """params (dictionary) contains: population_size, num_iterations, num_variables, lb, ub """
    
    def solveContinuous(self, problem, seed=None):
        # Minimize fitness
        population_size = self.params.get("pop_size", 50)
        num_iterations = self.params.get("num_iters", 100)
        lb = problem.min_range
        ub = problem.max_range
        num_variables = problem.dimension

        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []
        logger.history["best_fitness"] = []   
        # Initialize randomly for each student in population
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
                r = np.random.rand(num_variables)
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
                r = np.random.rand(num_variables)

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

        return {"time(ms)": logger.meta["runtime"], 
                "result":{"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def repair(self, solution, weights, values, capacity):
        # Nếu vượt capacity → bỏ item có value/weight nhỏ nhất trước
        while np.dot(solution, weights) > capacity:
            ratios = values / weights
            idx = np.where(solution == 1)[0]
            worst = idx[np.argmin(ratios[idx])]
            solution[worst] = 0
        return solution
    
    def solveKnapsack(self, problem, seed=None):
        n = self.params.get("pop_size", 25)
        dim = problem.dimension
        num_iters = self.params.get("num_iters", 100)

        weights = problem.weights
        values = problem.values
        capacity = problem.capacity

        logger = Logger(self.name, run_id = seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []


        #Initialize randomly for each student in population
        population = np.random.uniform(-4,4, size=(n, dim))
        fitness = np.zeros(n)
        binary_pop = np.zeros_like(population)

        for i in range(n):
            prob = self.sigmoid(population[i])
            bin_sol = (np.random.rand(dim) < prob).astype(int)
            bin_sol = self.repair(bin_sol, weights, values, capacity)
            binary_pop[i] = bin_sol.copy()
            fitness[i] = problem.evaluate(bin_sol)
        
        best_idx = np.argmax(fitness)
        logger.log("best_fitness",fitness[best_idx])
        logger.log("current_best", binary_pop[best_idx].copy())
        # Main Loop
        for iter in range (num_iters):
            for student in range(n):
                # Teacher Phase
                best_idx = np.argmax(fitness)
                X_best = population[best_idx]
                X_mean = np.mean(population, axis = 0)
                r = np.random.rand(dim)
                Tf = np.random.randint(1,3)

                X_new = population[student] + r*(X_best - Tf*X_mean)
                prob = self.sigmoid(X_new)
                bin_sol = (np.random.rand(dim) < prob).astype(int)
                bin_sol = self.repair(bin_sol, weights, values, capacity)

                fnew = problem.evaluate(bin_sol)
                if fnew > fitness[student]:
                    population[student] = X_new
                    fitness[student] = fnew
                    binary_pop[student] = bin_sol.copy()
                
                # Learner Phase
                partner_idx = np.random.randint(n)
                while partner_idx == student:
                    partner_idx = np.random.randint(n)
                r = np.random.rand(dim)

                if fitness[student] > fitness[partner_idx]:
                    X_new = population[student] + r*(population[student]-population[partner_idx])
                else:
                    X_new = population[student] - r*(population[student]-population[partner_idx])
                
                prob = self.sigmoid(X_new)
                bin_sol = (np.random.rand(dim) < prob).astype(int)
                bin_sol = self.repair(bin_sol, weights, values, capacity)

                fnew = problem.evaluate(bin_sol)
                if fnew > fitness[student]:
                    population[student] = X_new
                    fitness[student] = fnew
                    binary_pop[student] = bin_sol.copy()
            
            best_idx = np.argmax(fitness)
            logger.log("best_fitness", fitness[best_idx])
            logger.log("current_best", binary_pop[best_idx].copy())

        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_solution = binary_pop[best_idx].copy()
        logger.finish(best_solution, best_fitness)

        return {"time(ms)": logger.meta["runtime"], 
                "result": {"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}
    def solve(self, problem, seed = None):
        if problem.cont_flag:
            return self.solveContinuous(problem, seed)
        else:
            return self.solveKnapsack(problem, seed)
