import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
from src.problems.base_problem import BaseProblem


class ArtificialBee(BaseAlgorithm):

    def __init__(self, params=None):
        # 1. Define the safe default parameters using UNDERSCORES consistently
        default_params = {"total_bee": 50, "iteration": 400, "limit": 20}

        # 2. If the user passed anything, update the defaults with their values
        if params:
            default_params.update(params)

        # 3. Pass the merged dictionary to the BaseAlgorithm
        super().__init__("Artificial Bee Colony", default_params)


    def calc_fitness(self, flag, cost):
        # flag = problem.cont_flag
        # Fitness conversion for minimization
        if flag: # continuous case: as standard
            if cost >= 0:
                return 1 / (cost + 1)
            else:
                return 1 + np.abs(cost)
        else:
            return cost # return the value of knapsack

    # def helper_cand_gen(self, dim, cur, partner, flag, lower, upper):
    #     # create candidate solution
    #     phi = np.random.uniform(-1, 1, dim)
    #
    #     # <solution> = <food source> + <phi> * <delta food source>
    #     new_solution = cur + phi * (cur - partner)
    #
    #     if flag: # case: continuous problem
    #         return np.clip(new_solution, lower, upper)
    #     else:
    #         probs = 1 / (1 + np.exp(-new_solution))
    #         rand_mat = np.random.rand(dim)
    #         return (rand_mat < probs).astype(int)

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

        # divide total_bees into employed and unemployed
        employed = self.params["total_bee"] // 2
        onlooker = self.params["total_bee"] - employed
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

        cost_arr = np.array([problem.evaluate(i) for i in food_source_arr]) # evaluate cost for each food source
        fitness_arr = np.array([self.calc_fitness(flag, np.array([c])) for c in cost_arr]) # calculate fitness for each cost

        trial_cnt_arr = np.zeros(employed) # count trial for each employed bee

        # Track Global Best
        if flag:
            # Continuous: Minimization (Find Smallest Cost)
            best_idx = np.argmin(cost_arr)
        else:
            # Discrete: Maximization (Find Largest Value)
            best_idx = np.argmax(cost_arr)

        best_solution = food_source_arr[best_idx].copy()
        best_cost = cost_arr[best_idx]

        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_cost"] = []
        logger.history["population"] = []

        # --- Main Loop ---
        for it in range(self.params["iteration"]):

            # --- Employed Bees ---
            for j in range(employed):
                # Select random partner k != j (force the while loop to run at least 1)
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k!= j

                # generate candidate solution
                new_solution = self.helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], flag, lower_bound, upper_bound)
                new_cost = problem.evaluate(new_solution)
                new_fitness = self.calc_fitness(flag, new_cost)

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

            probs = probs.flatten()
            for _ in range(onlooker):
                # Probabilistic selection
                j = np.random.choice(np.arange(employed), p=probs)

                # Same logic as Employed bees
                k = j
                while k == j:
                    k = np.random.randint(employed) # ensure k !=j

                new_solution = self.helper_cand_gen(dim, food_source_arr[j], food_source_arr[k], flag, lower_bound, upper_bound)
                new_cost = problem.evaluate(new_solution)
                new_fitness = self.calc_fitness(flag, new_cost)

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
                if flag:
                    # Continuous Reset
                    food_source_arr[max_trials_idx] = np.random.uniform(lower_bound, upper_bound, dim)
                else:
                    # Discrete Reset
                    food_source_arr[max_trials_idx] = np.random.randint(lower_bound, upper_bound, dim)

                cost_arr[max_trials_idx] = problem.evaluate(food_source_arr[max_trials_idx])
                fitness_arr[max_trials_idx] = self.calc_fitness(flag, cost_arr[max_trials_idx])
                trial_cnt_arr[max_trials_idx] = 0


            # --- Update Global Best ---
            if flag:
                # Minimization (Continuous)
                cur_best_idx = np.argmin(cost_arr)
                if cost_arr[cur_best_idx] < best_cost:
                    best_cost = cost_arr[cur_best_idx]
                    best_solution = food_source_arr[cur_best_idx].copy()
            else:
                # Maximization (Knapsack)
                cur_best_idx = np.argmax(cost_arr)
                if cost_arr[cur_best_idx] > best_cost:  # Greater than
                    best_cost = cost_arr[cur_best_idx]
                    best_solution = food_source_arr[cur_best_idx].copy()

            # Logging per iteration
            logger.log("best_solution", best_solution.copy())
            logger.log("best_cost", best_cost)
            if flag:
                current_best_idx = np.argmin(cost_arr)
            else:
                current_best_idx = np.argmax(cost_arr)
            current_best_bee = food_source_arr[current_best_idx].copy()
            logger.history["current_best"].append(current_best_bee)
            logger.history["population"].append(food_source_arr.copy())
        # Finish
        logger.finish(best_solution=best_solution, best_fitness=best_cost)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_solution": best_solution.tolist(), "best_fitness": self.calc_fitness(flag, best_cost), "logger": logger}}

