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


    def solve(self, problem, seed=None):
        if seed is not None:
            np.random.seed(seed)

            # Route the problem based on the continuous flag!
        if getattr(problem, 'cont_flag', False):
            return self._solve_cont(problem, seed)
        else:
            return self._solve_gb(problem, seed)

    def _solve_cont(self, problem, seed=None):

        # divide pop_sizes into employed and unemployed
        employed = self.params["pop_size"] // 2
        onlooker = self.params["pop_size"] - employed
        limit = self.params["limit"]  # if trial count excesses this limit, reset the food source
        dim = problem.dimension
        flag = problem.cont_flag

        bounds = np.array(problem.bounds)
        lower_bound = bounds[:, 0]
        upper_bound = bounds[:, 1]
        food_source_arr = np.random.uniform(lower_bound, upper_bound, (employed, dim)) # random float

        fitness_arr = np.array([problem.evaluate(i) for i in food_source_arr]) # evaluate fitness for each food source
        score_arr = np.array([self.calc_score(flag, np.array([c])) for c in fitness_arr]) # calculate score for each fitness

        trial_cnt_arr = np.zeros(employed) # count trial for each employed bee

        best_idx = np.argmin(fitness_arr)

        best_solution = food_source_arr[best_idx].copy()
        best_fitness = fitness_arr[best_idx]

        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []
        logger.history["population"] = []
        logger.history["avg_fitness"] = []

        def helper_cand_gen(dim, cur, partner, lower, upper):
            # Continuous problem
            phi = np.random.uniform(-1, 1, dim)
            # <solution> = <food source> + <phi> * <delta food source>
            new_solution = cur + phi * (cur - partner)
            return np.clip(new_solution, lower, upper)  # clip to ensure bound safety

        # --- Main Loop ---
        for it in range(self.params["num_iters"]):

            # --- Employed Bees ---
            for j in range(employed):
                # Select random partner k != j (force the while loop to run at least 1)
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k!= j

                # generate candidate solution
                new_solution = helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], lower_bound, upper_bound)
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

                new_solution = helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], lower_bound, upper_bound)
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

                food_source_arr[max_trials_idx] = np.random.uniform(lower_bound, upper_bound, dim)

                fitness_arr[max_trials_idx] = problem.evaluate(food_source_arr[max_trials_idx])
                score_arr[max_trials_idx] = self.calc_score(flag, fitness_arr[max_trials_idx])
                trial_cnt_arr[max_trials_idx] = 0

            # --- Update Global Best ---
            cur_best_idx = np.argmin(fitness_arr)
            if fitness_arr[cur_best_idx] < best_fitness:
                best_fitness = fitness_arr[cur_best_idx]
                best_solution = food_source_arr[cur_best_idx].copy()

            current_best_idx = np.argmin(fitness_arr)
            current_best_bee = food_source_arr[current_best_idx].copy()

            # Logging per iteration
            logger.log("best_solution", best_solution.copy())

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

    def _solve_gb(self, problem, seed=None):
        employed = self.params["pop_size"] // 2
        onlooker = self.params["pop_size"] - employed
        limit = self.params["limit"]
        dim = problem.dimension

        # Sparse binary initialization for Knapsack
        food_source_arr = (np.random.rand(employed, dim) < 0.05).astype(int)
        fitness_arr = np.array([problem.evaluate(i) for i in food_source_arr])

        # Maximize: score = fitness
        score_arr = fitness_arr.copy()
        trial_cnt_arr = np.zeros(employed)

        best_idx = np.argmax(fitness_arr)
        best_solution = food_source_arr[best_idx].copy()
        best_fitness = fitness_arr[best_idx]

        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []
        logger.history["population"] = []
        logger.history["avg_fitness"] = []

        # Genetic-based generator
        def gen_candidate_gb(cur, par1, par2, gbest):
            zero_sol = np.zeros(dim, dtype=int) # use dim from outer
            pool = [par1, par2, gbest, zero_sol]
            mate = pool[np.random.randint(len(pool))]

            # Crossover
            pt1, pt2 = sorted(np.random.choice(range(dim), 2, replace=False))
            child = cur.copy()
            child[pt1:pt2] = mate[pt1:pt2]

            # Mutation
            grandchild = child.copy()
            idx1, idx2 = np.random.choice(range(dim), 2, replace=False)
            grandchild[idx1], grandchild[idx2] = grandchild[idx2], grandchild[idx1]

            return grandchild

        for ite in range(self.params["num_iters"]):

            # --- Employed Bees ---
            for j in range(employed):
                # GB-ABC: Pick 2 partners
                k1 = j
                while k1 == j: k1 = np.random.randint(employed) # ensure k1 != j
                k2 = j
                while k2 == j or k2 == k1: k2 = np.random.randint(employed) # ensure k1 != j

                new_solution = gen_candidate_gb(food_source_arr[j], food_source_arr[k1], food_source_arr[k2],
                                                best_solution)
                new_fitness = problem.evaluate(new_solution)

                if new_fitness > fitness_arr[j]:
                    food_source_arr[j] = new_solution
                    fitness_arr[j] = new_fitness
                    score_arr[j] = new_fitness
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Onlooker Bees ---
            sum_score = np.sum(score_arr)
            probs = np.ones(employed) / employed if sum_score == 0 else score_arr / sum_score

            for _ in range(onlooker):
                j = np.random.choice(np.arange(employed), p=probs)

                # GB-ABC: Pick 2 partners
                k1 = j
                while k1 == j: k1 = np.random.randint(employed)
                k2 = j
                while k2 == j or k2 == k1: k2 = np.random.randint(employed)

                new_solution = gen_candidate_gb(food_source_arr[j], food_source_arr[k1], food_source_arr[k2],
                                                best_solution)
                new_fitness = problem.evaluate(new_solution)

                if new_fitness > fitness_arr[j]:
                    food_source_arr[j] = new_solution
                    fitness_arr[j] = new_fitness
                    score_arr[j] = new_fitness
                    trial_cnt_arr[j] = 0
                else:
                    trial_cnt_arr[j] += 1

            # --- Scout Bees ---
            max_trials_idx = np.argmax(trial_cnt_arr) # pick max
            if trial_cnt_arr[max_trials_idx] > limit:
                food_source_arr[max_trials_idx] = (np.random.rand(dim) < 0.05).astype(int)
                fitness_arr[max_trials_idx] = problem.evaluate(food_source_arr[max_trials_idx])
                score_arr[max_trials_idx] = fitness_arr[max_trials_idx]
                trial_cnt_arr[max_trials_idx] = 0

            # --- Update Global Best ---
            cur_best_idx = np.argmax(fitness_arr)
            if fitness_arr[cur_best_idx] > best_fitness:
                best_fitness = fitness_arr[cur_best_idx]
                best_solution = food_source_arr[cur_best_idx].copy()

            logger.log("best_solution", best_solution.copy())
            logger.history["avg_fitness"].append(np.mean(fitness_arr))
            logger.history["current_best"].append(food_source_arr[np.argmax(fitness_arr)].copy())
            logger.history["population"].append(food_source_arr.copy())
            logger.history["best_fitness"].append(best_fitness)

        logger.finish(best_solution=best_solution, best_fitness=best_fitness)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution.tolist(), "best_fitness": best_fitness, "logger": logger}}