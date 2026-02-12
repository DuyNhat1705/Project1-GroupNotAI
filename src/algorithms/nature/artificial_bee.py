import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class ArtificialBee(BaseAlgorithm):

    def __init__(self, params={"total bee": 50, "iteration": 200, "limit": 20}):
        super().__init__("Artificial Bee Colony", params)

    def calc_fitness(self, cost):
        # Fitness conversion for minimization
        if cost >= 0:
            return 1 / (cost + 1)
        else:
            return 1 + np.abs(cost)

    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # --- Initialization ---
        lower_bound = problem.bounds[:, 0]
        upper_bound = problem.bounds[:, 1]
        dim = problem.dimension

        # divide total bees into employed and unemployed
        employed = self.params["total bee"] // 2
        onlooker = self.params["total bee"] - employed
        limit = self.params["limit"] # if trial count excesses this limit, reset the food source

        # Random food sources corresponding to employed bees
        food_source_arr = np.random.uniform(lower_bound, upper_bound, (employed, dim))

        cost_arr = np.array([problem.evaluate(i) for i in food_source_arr]) # evaluate cost for each food source
        fitness_arr = np.array([self.calc_fitness(np.array([c]))[0] for c in cost_arr]) # calculate fitness for each cost

        trial_cnt_arr = np.zeros(employed) # count trial for each employed bee

        # Track Global Best
        best_idx = np.argmin(cost_arr)
        best_solution = food_source_arr[best_idx].copy()
        best_cost = cost_arr[best_idx]

        logger = Logger(self.name, run_id=seed)

        # --- Main Loop ---
        for it in range(self.params["iteration"]):

            # --- Employed Bees ---
            for j in range(employed):
                # Select random partner k != j (force the while loop to run at least 1)
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k!= j

                # create candidate solution
                phi = np.random.uniform(-1, 1, dim)
                # <solution> = <food source> + <phi> * <delta food source>
                new_solution = food_source_arr[j] + phi * (food_source_arr[j] - food_source_arr[k])
                new_solution = np.clip(new_solution, lower_bound, upper_bound) # clipping to ensure boundary condition

                new_cost = problem.evaluate(new_solution)
                new_fitness = self.calc_fitness(np.array([new_cost]))[0]

                # greedy selection
                if new_fitness > fitness_arr[j]:
                    food_source_arr[j] = new_solution
                    cost_arr[j] = new_cost
                    fitness_arr[j] = new_fitness
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Onlooker Bees ---
            sum_fitness = np.sum(fitness_arr)
            if sum_fitness == 0: # handle special case so that (avoid denominator = 0)
                probs = np.ones(employed) / employed # even probability for each food source
            else:
                probs = fitness_arr / sum_fitness # total prob = 1; in which higher fitness attached with high probability

            for _ in range(onlooker):
                # Probabilistic selection
                j = np.random.choice(np.arange(employed), p=probs)

                # Same logic as Employed bees
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k !=j

                phi = np.random.uniform(-1, 1, dim)
                new_solution = food_source_arr[j] + phi * (food_source_arr[j] - food_source_arr[k])
                new_solution = np.clip(new_solution, lower_bound, upper_bound)

                new_cost = problem.evaluate(new_solution)
                new_fitness = self.calc_fitness(np.array([new_cost]))[0]

                if new_fitness > fitness_arr[j]:
                    food_source_arr[j] = new_solution
                    cost_arr[j] = new_cost
                    fitness_arr[j] = new_fitness
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Scout Bees ---
            max_trials_idx = np.argmax(trial_cnt_arr) # take the highest trial count
            if trial_cnt_arr[max_trials_idx] > limit:
                # Reset this food source randomly
                food_source_arr[max_trials_idx] = np.random.uniform(lower_bound, upper_bound, dim)
                cost_arr[max_trials_idx] = problem.evaluate(food_source_arr[max_trials_idx])
                fitness_arr[max_trials_idx] = self.calc_fitness(np.array([cost_arr[max_trials_idx]]))[0]
                trial_cnt_arr[max_trials_idx] = 0

            # --- Update Global Best ---

            cur_best_idx = np.argmin(cost_arr)
            if cost_arr[cur_best_idx] < best_cost:
                best_cost = cost_arr[cur_best_idx]
                best_solution = food_source_arr[cur_best_idx].copy()

            # Logging per iteration
            logger.log("best_cost", best_cost)

        # Finish
        logger.finish(best_solution=best_solution, best_fitness=best_cost)
        return best_solution, best_cost, logger

