import time
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger
import numpy as np

# BFS solves: shortest path finding, knapsack problem, graph coloring

class BFS(BaseAlgorithm):
    def __init__(self, params=None):

        super().__init__("Breadth-First Search", params if params else {})

    def solve(self, problem, seed=None):
        # Route the problem to the correct solver logic
        # not handle wrong pass case
        if problem.getName() == "Knapsack Problem":
            return self._solve_knapsack(problem, seed)
        elif problem.getName() == "Graph Coloring":
            return self._solve_graph_coloring(problem, seed)
        else:
            return self._solve_pathfinding(problem, seed)

    def _solve_graph_coloring(self, problem, seed):
        # State: (node index, tuple of assigned colors)
        start_state = (0, ())
        queue = [start_state]

        # init logger
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []

        # fallback for timeout
        best_solution = [0] * problem.dimension
        best_cost = problem.evaluate(np.array(best_solution))
        nodes_expanded = 0

        start_time = time.perf_counter()
        time_limit = 10.0  # time limit

        while len(queue) > 0:
            if time.perf_counter() - start_time > time_limit: # return when time limit reached
                print(f"  [BFS] Time limit ({time_limit}s) reached")
                break

            current = queue.pop(0) # take the frontier state
            idx, path = current # idx of node
            nodes_expanded += 1

            # If leaf of the decision tree reached (all nodes colored)
            if idx == problem.dimension:
                current_solution = np.array(path)
                cost = problem.evaluate(current_solution)

                if cost < best_cost: # if better cost found, update best_solution
                    best_cost = cost
                    best_solution = list(path)
                    logger.history["current_best"].append(np.array(best_solution))
                continue

            # Only try existing colors + 1 new. consider (0,1) and (1,0) same
            max_color_used = max(path) if path else -1

            for c in range(max_color_used + 2): # +2 due to exclusive for loop
                # Ensure less or equal (num_nodes - 1)
                if c < problem.dimension:
                    queue.append((idx + 1, path + (c,))) #add new color to path

        final_solution = np.array(best_solution)
        logger.log("cost", best_cost)
        logger.finish(best_solution=final_solution, best_fitness=best_cost)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {"path": best_solution, "cost": best_cost, "best_fitness": best_cost,
                       "nodes_expanded": nodes_expanded, "logger": logger}
        }

    def _solve_knapsack(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []

        # (item idx, current weight, current value, selection tuple)
        start_state = (0, 0.0, 0.0, ())
        queue = [start_state]

        # Fallback solution
        best_value = -1
        best_solution = [0] * problem.dimension
        nodes_expanded = 0

        start_time = time.perf_counter()
        time_limit = 10.0  # Time limit

        while len(queue) > 0:
            if time.perf_counter() - start_time > time_limit:
                print(f"  [BFS] Time limit ({time_limit}s) reached!")
                break

            current = queue.pop(0)  # FIFO, take frontier
            idx, cur_wei, cur_val, path = current
            nodes_expanded += 1

            # If leaf of the decision tree reached
            if idx == problem.dimension:
                if cur_val > best_value:
                    best_value = cur_val
                    best_solution = list(path)
                    # Log the 1D numpy arr for later used by KnapsackViz
                    logger.history["current_best"].append(np.array(best_solution))
                continue

            # Branch 1: Exclude the current item (0)
            queue.append((idx + 1, cur_wei, cur_val, path + (0,)))

            # Branch 2: Include the current item (1)
            # Only branch if avaible capacity
            if cur_wei + problem.weights[idx] <= problem.capacity:
                queue.append(
                    (idx + 1, cur_wei + problem.weights[idx], cur_val + problem.values[idx], path + (1,)))

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

        queue = [start]  # queue of frontier
        visited = {start}  # visited nodes
        predecessor = {start: None}  # reconstruct path from root to current node

        logger = Logger(self.name, run_id=seed)
        logger.history["visited_edges"] = []
        logger.history["visited_edges"].append((start, start))
        ite = 0  # iteration count

        start_time = time.perf_counter()
        time_limit = 10.0  # Time limit to prevent freeze on massive open mazes

        while len(queue) > 0:
            if time.perf_counter() - start_time > time_limit:
                print(f"  [BFS] Time limit ({time_limit}s) reached! Aborting early...")
                break

            current = queue.pop(0)  # pop front
            ite += 1

            if current == goal:
                path = self.reconstruct_path(predecessor, current)
                cost = problem.evaluate(path)
                fitness = len(path) - 1  # not take weight in account
                logger.log("cost", cost)
                logger.finish(best_solution=path, best_fitness=fitness)
                return {"time(ms)": logger.meta["runtime"],
                        "result": {"path": path, "cost": cost, "nodes_expanded": len(logger.history["visited_edges"]),
                                   "logger": logger}}

            neighbors = problem.get_neighbors(current)

            for neigh in neighbors:
                if neigh not in visited:
                    visited.add(neigh)
                    predecessor[neigh] = current
                    queue.append(neigh)

                    # Log edge for visualization
                    logger.history["visited_edges"].append((current, neigh))

        # No path found or timeout
        logger.finish(best_solution=[], best_fitness=float('inf'))
        return {"time(ms)": logger.meta["runtime"],
                "result": {"path": [], "cost": float('inf'), "nodes_expanded": len(logger.history["visited_edges"]),
                           "logger": logger}}

    def reconstruct_path(self, previous, current):
        complete_path = [current]
        while current in previous:
            current = previous[current]
            if current is None:
                break
            complete_path.append(current)
        complete_path.reverse()
        return complete_path