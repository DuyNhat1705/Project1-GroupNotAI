import numpy as np
from copy import deepcopy
import random
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger

class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, params=None):
        default_params = {
            "pop_size": 50,
            "generations": 100,
            "F": 0.5,
            "CR": 0.1,
        }
        if params:
            default_params.update(params)
        super().__init__("Genetic Algorithm", default_params)

    def solve(self, problem, seed=None):
        problem_name = problem.getName().lower()

        if "tsp" in problem_name:
            return self.tsp_solve(problem, seed)
            
        return self.continous_solve(problem, seed)
    
    def tsp_create_population(self, problem, pop_size):
        dist_mat = problem.dist_mat
        n_cities = len(dist_mat[0])

        base_path = np.arange(n_cities)
        population = np.zeros((pop_size, n_cities), dtype=int)

        for i in range(pop_size):
            p = base_path.copy()
            np.random.shuffle(p)
            population[i] = p

        return population

    def tsp_selection(self, population, fitness, individuals_to_keep):
        sorted_indices = np.argsort(fitness)[:individuals_to_keep]
        return population[sorted_indices]

    def tsp_crossover(self, parents, n_offspring, dim):
        offspring = np.zeros((n_offspring, dim), dtype=int)

        for k in range(n_offspring):
            idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
            p1, p2 = parents[idx1], parents[idx2]

            i, j = sorted(np.random.choice(dim, 2, replace=False))

            child = -np.ones(dim, dtype=int)
            used = np.zeros(dim, dtype=bool)   

            # copy segment from p1
            child[i:j] = p1[i:j]
            used[p1[i:j]] = True

            # fill remaining from p2
            pos = j
            for city in p2:
                if not used[city]:            
                    if pos >= dim:
                        pos = 0
                    child[pos] = city
                    used[city] = True
                    pos += 1

            offspring[k] = child

        return offspring

    def tsp_mutation(self, population):
        dim = population.shape[1]
        prob = self.params["CR"]

        for path in population:
            if np.random.rand() < prob:
                i, j = sorted(np.random.choice(dim, 2, replace=False))
                path[i:j] = path[i:j][::-1]

        return population

    def tsp_solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        pop_size = self.params["pop_size"]
        n_gens = self.params["generations"]
        dim = problem.dimension

        # Logger
        logger = Logger(self.name, run_id=seed)
        logger.history["best_fitness"] = []
        logger.history["population"] = []

        # Init population (numpy array)
        population = self.tsp_create_population(problem, pop_size)

        best_solution = None
        best_fitness = float("inf")

        for gen in range(n_gens):
            # Evaluate
            fitness = np.array([problem.evaluate(ind) for ind in population])

            idx = np.argmin(fitness)
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                best_solution = population[idx].copy()

            # Log (copy để tránh mutation ảnh hưởng history)
            logger.log("best_fitness", best_fitness)
            logger.log("population", population.copy())
            logger.log("best_solution", population[idx].copy())
            logger.log("best_fitness_gen", fitness[idx])

            # Selection
            survivors = self.tsp_selection(
                population, fitness, int(pop_size * 0.5)
            )

            # Crossover
            offspring = self.tsp_crossover(survivors, pop_size, dim)

            # Mutation
            population = self.tsp_mutation(offspring)

        logger.finish(best_solution, best_fitness)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best_solution,
                "best_fitness": best_fitness,
                "logger": logger
            }
        }

    def continous_selection(self, population, fitness, individuals_to_keep):
        sorted_indices = np.argsort(fitness)[:individuals_to_keep]
        return population[sorted_indices]

    def continous_crossover(self, parents, n_offspring, dim):
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

    def continous_mutation(self, population, bounds):
        pop_size, dim = population.shape
        prob = self.params["CR"]
        fluc = self.params["F"]
        
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

    def continous_solve(self, problem, seed=None):
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
        logger.history["population"] = []     
        logger.history["best_fitness"] = []   
        logger.history["avg_fitness"] = []

        best_solution = None
        best_fitness = float('inf')

        for gen in range(n_gens):
            # 1. Đánh giá
            fitness_values = np.array([problem.evaluate(ind) for ind in population])
            avg_fitness = np.mean(fitness_values)

            # Cập nhật Best Global (Chỉ để tracking, không giữ lại trong quần thể)
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_solution = population[min_idx].copy()

            # Log history
            logger.log("avg_fitness", avg_fitness)
            logger.log("population", population.copy())
            logger.log("best_fitness", best_fitness)

            # 2. Selection cho lai ghép (Chọn top 50% làm cha mẹ)
            survivors = self.continous_selection(population, fitness_values, int(pop_size * 0.5))

            # 3. Crossover (Tạo ra số lượng con cái bằng đúng pop_size để thay thế hoàn toàn)
            offspring = self.continous_crossover(survivors, pop_size, dim)

            # 4. Mutation
            offspring = self.continous_mutation(offspring, bounds)

            # 5. Tạo quần thể mới (Thay máu toàn bộ 100%)
            population = offspring

        logger.finish(best_solution, best_fitness)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution, "best_fitness": best_fitness, "logger": logger}}