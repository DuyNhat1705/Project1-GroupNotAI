from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np



class BFS(BaseAlgorithm):
    def __init__(self, params=None):

        super().__init__("Breadth-First Search", params if params else {})

    def solve(self, problem, seed=None):
        # Route the problem to the correct solver logic!
        if problem.getName() == "Knapsack Problem":
            return self._solve_knapsack(problem, seed)
        else:
            return self._solve_pathfinding(problem, seed)

    def _solve_knapsack(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []

        # (item_index, current_weight, current_value, selection_tuple)
        start_state = (0, 0.0, 0.0, ())
        queue = [start_state]

        best_value = -1
        best_solution = []
        nodes_expanded = 0

        while len(queue) > 0:
            current = queue.pop(0)  # FIFO for BFS
            idx, current_weight, current_value, path = current
            nodes_expanded += 1

            # If leaf of the decision tree reached
            if idx == problem.dimension:
                if current_value > best_value:
                    best_value = current_value
                    best_solution = list(path)
                    # Log the 1D numpy array for later used by KnapsackViz
                    logger.history["current_best"].append(np.array(best_solution))
                continue

            # Branch 1: Exclude the current item (0)
            queue.append((idx + 1, current_weight, current_value, path + (0,)))

            # Branch 2: Include the current item (1)
            # Only branch if avaible capacity
            if current_weight + problem.weights[idx] <= problem.capacity:
                queue.append(
                    (idx + 1, current_weight + problem.weights[idx], current_value + problem.values[idx], path + (1,)))

        # Evaluate final formulation
        final_solution = np.array(best_solution)
        cost = problem.evaluate(final_solution)

        logger.log("cost", cost)
        logger.finish(best_solution=final_solution, best_fitness=best_value)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {"path": best_solution, "cost": cost, "best_fitness": best_value,
                       "nodes_expanded": nodes_expanded, "logger": logger}
        }

    def _solve_pathfinding(self, problem, seed=None):

        start = problem.start
        goal = problem.goal

        queue = [start] # queue of frontier
        visited = {start} # visited nodes
        predecessor = {start: None} # reconstruct path from root to current node

        logger = Logger(self.name, run_id=seed)
        logger.history["visited_edges"] = []
        logger.history["visited_edges"].append((start, start))
        ite = 0  # iteration count

        while len(queue) > 0:
            current = queue.pop(0) # pop front
            ite += 1

            if current == goal:
                path= self.reconstruct_path(predecessor, current)
                cost = problem.evaluate(path)
                fitness = len(path) - 1 # not take weight in account
                logger.log("cost", cost)
                logger.finish(best_solution=path, best_fitness=fitness)
                return {"time(ms)":logger.meta["runtime"],
                        "result":{"path": path, "cost": cost, "nodes_expanded": len(logger.history["visited_edges"]), "logger": logger}}

            neighbors = problem.get_neighbors(current)

            for neigh in neighbors:
                if neigh not in visited:
                    visited.add(neigh)
                    predecessor[neigh] = current
                    queue.append(neigh)

                    # Log edge for visualization
                    logger.history["visited_edges"].append((current, neigh))

        # No path found
        logger.finish(best_solution=[], best_fitness=float('inf'))
        return {"time(ms)":logger.meta["runtime"],
                "result": {"path": [], "cost": float('inf'),"nodes_expanded": len(logger.history["visited_edges"]), "logger": logger}}


    def reconstruct_path(self, previous, current):
        complete_path = [current]
        while current in previous:
            current = previous[current]
            if current is None:
                break
            complete_path.append(current)
        complete_path.reverse()
        return complete_path