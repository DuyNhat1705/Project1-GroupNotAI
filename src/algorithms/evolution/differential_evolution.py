import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger

class DifferentialEvolution(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            "pop_size": 50,
            "generations": 100,
            "F": 0.8,   # Differential Weight
            "CR": 0.9,  # Crossover Probability
            "elitism": True
        }
        if params:
            default_params.update(params)
        super().__init__("Differential Evolution", default_params)

    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dim = problem.dimension
        bounds = problem.bounds
        pop_size = self.params["pop_size"]
        max_gens = self.params["generations"]
        F = self.params["F"]
        CR = self.params["CR"]

        # Bounds arrays
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        # Init Population
        population = np.random.uniform(lower, upper, (pop_size, dim))
        
        # Evaluate initial
        fitness = np.array([problem.evaluate(ind) for ind in population])

        # Logging
        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []
        logger.history["best_fitness"] = []

        # Track Best
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for gen in range(max_gens):
            # Log đầu chu kỳ
            logger.log("population", population.copy())
            logger.log("best_fitness", best_fitness)

            new_population = np.zeros_like(population)
            trial_fitness = np.zeros_like(fitness)

            for i in range(pop_size):
                # 1. Mutation: select 3 distinct random indices != i
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                
                mutant_vector = population[r1] + F * (population[r2] - population[r3])
                mutant_vector = np.clip(mutant_vector, lower, upper)

                # 2. Crossover
                trial_vector = np.zeros(dim)
                j_rand = np.random.randint(dim)
                
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial_vector[j] = mutant_vector[j]
                    else:
                        trial_vector[j] = population[i, j]

                # 3. Selection (Greedy)
                f_trial = problem.evaluate(trial_vector)
                
                if f_trial < fitness[i]:
                    new_population[i] = trial_vector
                    trial_fitness[i] = f_trial
                    
                    # Update global best immediately
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_solution = trial_vector.copy()
                else:
                    new_population[i] = population[i]
                    trial_fitness[i] = fitness[i]

            population = new_population
            fitness = trial_fitness

        logger.finish(best_solution, best_fitness)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}