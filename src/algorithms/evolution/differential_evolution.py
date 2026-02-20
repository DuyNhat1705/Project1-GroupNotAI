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
        }
        if params:
            default_params.update(params)
        super().__init__("Differential Evolution", default_params)

    def _mutation(self, population, lower, upper):
        """Bước 1: Tạo vector đột biến cho toàn bộ quần thể"""
        pop_size, dim = population.shape
        F = self.params["F"]
        mutants = np.zeros_like(population)
        
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            mutants[i] = population[r1] + F * (population[r2] - population[r3])
            
        # Đảm bảo các vector đột biến không vượt qua bounds
        return np.clip(mutants, lower, upper)

    def _crossover(self, population, mutants):
        """Bước 2: Lai ghép nhị thức tạo vector thử nghiệm (trial vectors)"""
        pop_size, dim = population.shape
        CR = self.params["CR"]
        trial_vectors = np.zeros_like(population)
        
        for i in range(pop_size):
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial_vectors[i, j] = mutants[i, j]
                else:
                    trial_vectors[i, j] = population[i, j]
                    
        return trial_vectors

    def _selection(self, population, fitness, trial_vectors, problem):
        """Bước 3: Đánh giá và chọn lọc cá thể tốt hơn"""
        # Đánh giá fitness cho toàn bộ tập trial vectors mới
        trial_fitness = np.array([problem.evaluate(ind) for ind in trial_vectors])
        
        # So sánh trực tiếp: True nếu trial tốt hơn population
        better_mask = trial_fitness < fitness
        
        # Cập nhật quần thể và fitness mới bằng np.where (nhanh hơn if-else)
        new_population = np.where(better_mask[:, None], trial_vectors, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness

    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dim = problem.dimension
        bounds = problem.bounds
        pop_size = self.params["pop_size"]
        max_gens = self.params["generations"]

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

            # 1. Mutation
            mutants = self._mutation(population, lower, upper)
            
            # 2. Crossover
            trial_vectors = self._crossover(population, mutants)
            
            # 3. Selection
            population, fitness = self._selection(population, fitness, trial_vectors, problem)
            
            # Cập nhật Global Best sau khi hoàn tất chọn lọc
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()

        logger.finish(best_solution, best_fitness)
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best_solution, 
                "best_fitness": best_fitness, 
                "logger": logger
            }
        }