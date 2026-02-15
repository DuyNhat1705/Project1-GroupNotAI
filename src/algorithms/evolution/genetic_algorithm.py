import numpy as np
from copy import deepcopy
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger

class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, params=None):
        # Default params
        default_params = {
            "pop_size": 50,
            "generations": 100,
            "mutation_rate": 0.1,
            "mutation_fluctuation": 0.5,
            "elitism": True
        }
        if params:
            default_params.update(params)
        super().__init__("Genetic Algorithm", default_params)

    def _selection(self, population, fitness, individuals_to_keep):
        sorted_indices = np.argsort(fitness)[:individuals_to_keep]
        return population[sorted_indices]

    def _crossover(self, parents, n_offspring, dim):
        offspring = np.zeros((n_offspring, dim))
        for k in range(n_offspring):
            # Chọn ngẫu nhiên 2 cha mẹ
            p_idx = np.random.randint(0, len(parents), 2)
            p1, p2 = parents[p_idx[0]], parents[p_idx[1]]
            
            # Mask bit crossover
            mask = np.random.randint(0, 2, dim).astype(bool)
            child = np.where(mask, p1, p2)
            offspring[k] = child
        return offspring

    def _mutation(self, population, bounds):
        pop_size, dim = population.shape
        prob = self.params["mutation_rate"]
        fluc = self.params["mutation_fluctuation"]
        
        # Mask đột biến
        mutation_mask = np.random.random((pop_size, dim)) < prob
        
        # Tạo nhiễu Gaussian
        noise = np.random.normal(0, fluc, size=population.shape)
        
        mutants = population + noise
        
        # Clip trong bound
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        mutants = np.clip(mutants, lower, upper)
        
        return np.where(mutation_mask, mutants, population)

    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dim = problem.dimension
        bounds = problem.bounds
        pop_size = self.params["pop_size"]
        n_gens = self.params["generations"]
        
        # Khởi tạo quần thể
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        population = np.random.uniform(lower, upper, (pop_size, dim))
        
        logger = Logger(self.name, run_id=seed)
        logger.history["population"] = []     # Để vẽ video
        logger.history["best_fitness"] = []   # Để vẽ biểu đồ hội tụ

        best_solution = None
        best_fitness = float('inf')

        for gen in range(n_gens):
            # 1. Đánh giá
            fitness_values = np.array([problem.evaluate(ind) for ind in population])
            
            # Cập nhật Best Global
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_solution = population[min_idx].copy()

            # Log history
            logger.log("population", population.copy())
            logger.log("best_fitness", best_fitness)

            # 2. Selection (Elitism)
            n_elites = 0
            elites = []
            if self.params["elitism"]:
                n_elites = max(1, int(pop_size * 0.1))
                elites = self._selection(population, fitness_values, n_elites)

            # 3. Selection cho lai ghép (Giữ lại top 50%)
            survivors = self._selection(population, fitness_values, int(pop_size * 0.5))

            # 4. Crossover
            n_needed = pop_size - n_elites
            offspring = self._crossover(survivors, n_needed, dim)

            # 5. Mutation
            offspring = self._mutation(offspring, bounds)

            # 6. Tạo quần thể mới
            if n_elites > 0:
                population = np.vstack((elites, offspring))
            else:
                population = offspring

        logger.finish(best_solution, best_fitness)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}