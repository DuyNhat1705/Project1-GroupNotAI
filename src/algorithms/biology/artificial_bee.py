import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
from src.problems.base_problem import BaseProblem


class ArtificialBee(BaseAlgorithm):

    def __init__(self, params=None):
        default_params = {"pop_size": 50, "num_iters": 100, "limit": 20}

        # If passed anything, update the defaults with their values
        if params:
            default_params.update(params)

        # Pass the merged dictionary to the BaseAlgorithm
        super().__init__("Artificial Bee Colony", default_params)

        for key, val in default_params.items():
            setattr(self, key, val)

    def calc_score(self, flag, fitness):
        # flag = problem.cont_flag
        # score conversion for minimization
        if flag: # continuous case: as standard
            if fitness >= 0:
                return 1 / (fitness + 1)
            else:
                return 1 + np.abs(fitness)
        else:
            return fitness # return the value of knapsack


    def helper_cand_gen(self, dim, cur, partner, flag, lower, upper):
        if flag:  # Continuous problem
            phi = np.random.uniform(-1, 1, dim)
            # <solution> = <food source> + <phi> * <delta food source>
            new_solution = cur + phi * (cur - partner)
            return np.clip(new_solution, lower, upper) # clip to ensure bound safety

        else:  # Discrete problem (Knapsack)
            new_solution = cur.copy()

            # Crossover: If partner differs => 50% chance to copy the partner's item
            diff = cur != partner
            copy_mask = diff & (np.random.rand(dim) < 0.5)
            new_solution[copy_mask] = partner[copy_mask]

            # 2. Mutation: Very small chance to randomly drop or pick up a new item
            # (1.0 / dim) ensures only ~1 item is flipped per turn, keeping the search stable!
            mutation_mask = np.random.rand(dim) < (1.0 / dim)
            new_solution[mutation_mask] = 1 - new_solution[mutation_mask]

            return new_solution

    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # --- Initialization ---

        # divide pop_sizes into employed and unemployed
        employed = self.params["pop_size"] // 2
        onlooker = self.params["pop_size"] - employed
        limit = self.params["limit"]  # if trial count excesses this limit, reset the food source
        dim = problem.dimension
        flag = problem.cont_flag

        # Random food sources corresponding to employed bees
        # 2 cases: continuous and discrete
        if flag:
            bounds = np.array(problem.bounds)
            lower_bound = bounds[:, 0]
            upper_bound = bounds[:, 1]
            food_source_arr = np.random.uniform(lower_bound, upper_bound, (employed, dim)) # random float
        else:
            lower_bound = 0
            upper_bound = 2
            #food_source_arr = np.random.randint(lower_bound, upper_bound, (employed, problem.dimension)) # rand int 0, 1
            food_source_arr = (np.random.rand(employed, dim) < 0.05).astype(int)

        fitness_arr = np.array([problem.evaluate(i) for i in food_source_arr]) # evaluate fitness for each food source
        score_arr = np.array([self.calc_score(flag, np.array([c])) for c in fitness_arr]) # calculate score for each fitness

        trial_cnt_arr = np.zeros(employed) # count trial for each employed bee

        # Track Global Best
        if flag:
            # Continuous: Minimization (Smallest fitness)
            best_idx = np.argmin(fitness_arr)
        else:
            # Discrete: Maximization (Largest Value)
            best_idx = np.argmax(fitness_arr)

        best_solution = food_source_arr[best_idx].copy()
        best_fitness = fitness_arr[best_idx]

        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []
        logger.history["population"] = []
        logger.history["avg_fitness"] = []

        # --- Main Loop ---
        for it in range(self.params["num_iters"]):

            # --- Employed Bees ---
            for j in range(employed):
                # Select random partner k != j (force the while loop to run at least 1)
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k!= j

                # generate candidate solution
                new_solution = self.helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], flag, lower_bound, upper_bound)
                new_fitness = problem.evaluate(new_solution)
                new_score = self.calc_score(flag, new_fitness)

                # greedy selection
                if new_score > score_arr[j]:
                    food_source_arr[j] = new_solution
                    fitness_arr[j] = new_fitness
                    score_arr[j] = new_score
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Onlooker Bees ---
            sum_score = np.sum(score_arr)
            if sum_score == 0: # handle special case so that (avoid denominator = 0)
                probs = np.ones(employed) / employed # even probability for each food source
            else:
                probs = score_arr / sum_score # total prob = 1; in which higher score attached with high probability

            probs = probs.flatten()
            for _ in range(onlooker):
                # Probabilistic selection
                j = np.random.choice(np.arange(employed), p=probs)

                # Same logic as Employed bees
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k !=j

                new_solution = self.helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], flag, lower_bound, upper_bound)
                new_fitness = problem.evaluate(new_solution)
                new_score = self.calc_score(flag, new_fitness)

                if new_score > score_arr[j]:
                    food_source_arr[j] = new_solution
                    fitness_arr[j] = new_fitness
                    score_arr[j] = new_score
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Scout Bees ---
            max_trials_idx = np.argmax(trial_cnt_arr) # take the highest trial count
            if trial_cnt_arr[max_trials_idx] > limit:
                # Reset this food source randomly
                if flag:
                    # Continuous Reset
                    food_source_arr[max_trials_idx] = np.random.uniform(lower_bound, upper_bound, dim)
                else:
                    # Discrete Reset
                    food_source_arr[max_trials_idx] = np.random.randint(lower_bound, upper_bound, dim)

                fitness_arr[max_trials_idx] = problem.evaluate(food_source_arr[max_trials_idx])
                score_arr[max_trials_idx] = self.calc_score(flag, fitness_arr[max_trials_idx])
                trial_cnt_arr[max_trials_idx] = 0


            # --- Update Global Best ---
            if flag:
                # Minimization (Continuous)
                cur_best_idx = np.argmin(fitness_arr)
                if fitness_arr[cur_best_idx] < best_fitness:
                    best_fitness = fitness_arr[cur_best_idx]
                    best_solution = food_source_arr[cur_best_idx].copy()
            else:
                # Maximization (Knapsack)
                cur_best_idx = np.argmax(fitness_arr)
                if fitness_arr[cur_best_idx] > best_fitness:  # Greater than
                    best_fitness = fitness_arr[cur_best_idx]
                    best_solution = food_source_arr[cur_best_idx].copy()

            # Logging per iteration
            logger.log("best_solution", best_solution.copy())

            if flag:
                current_best_idx = np.argmin(fitness_arr)
            else:
                current_best_idx = np.argmax(fitness_arr)
            current_best_bee = food_source_arr[current_best_idx].copy()
            logger.history["avg_fitness"].append(np.mean(fitness_arr))
            logger.history["current_best"].append(current_best_bee)
            logger.history["population"].append(food_source_arr.copy())
            logger.history["best_fitness"].append(best_fitness)
        # Finish
        logger.finish(best_solution=best_solution, best_fitness=best_fitness)
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "best_solution": best_solution.tolist(),
                "best_fitness": best_fitness,
                "logger": logger
            }
        }

