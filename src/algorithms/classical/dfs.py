import time
import numpy as np
from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class DFS(BaseAlgorithm):
    def __init__(self, params=None):
        super().__init__("Depth-First Search", params if params else {})

    def solve(self, problem, seed=None):
        if problem.getName() == "Knapsack Problem":
            return self._solve_knapsack(problem, seed)
        elif problem.getName() == "Graph Coloring":
            return self._solve_graph_coloring(problem, seed)
        else:
            return self._solve_pathfinding(problem, seed)

    # ==========================================================
    # GRAPH COLORING (DFS)
    # ==========================================================
    def _solve_graph_coloring(self, problem, seed):
        # State: (node_index, color_assignment)
        stack = [(0, ())]

        logger = Logger(self.name, run_id=seed)
        logger.history["best_fitness"] = []
        logger.history["current_best"] = []

        best_solution = [0] * problem.dimension
        best_cost = problem.evaluate(np.array(best_solution))
        nodes_expanded = 0

        num_iters = self.params.get("num_iters", 100)

        while stack:
            if nodes_expanded > num_iters:
                print(f"  [{self.name}] Iteration limit ({num_iters}) reached!")
                break

            idx, path = stack.pop()
            nodes_expanded += 1
            solution = np.array(path)
            logger.history["best_fitness"].append(best_cost)
            logger.history["current_best"].append(solution)

            if idx == problem.dimension:

                cost = problem.evaluate(solution)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = list(path)
                continue

            max_color_used = max(path) if path else -1

            # DFS: push ngược để giữ thứ tự giống BFS
            for c in reversed(range(max_color_used + 2)):
                if c < problem.dimension:
                    stack.append((idx + 1, path + (c,)))

        final_solution = np.array(best_solution)
        logger.log("best_fitness", best_cost)
        logger.finish(best_solution=final_solution, best_fitness=best_cost)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "path": best_solution,
                "best_fitness": best_cost,
                "nodes_expanded": nodes_expanded,
                "logger": logger
            }
        }

    # ==========================================================
    # KNAPSACK (DFS)
    # ==========================================================
    def _solve_knapsack(self, problem, seed):
        logger = Logger(self.name, run_id=seed)
        logger.history["current_best"] = []
        logger.history["best_fitness"] = []

        # (idx, weight, value, path)
        stack = [(0, 0.0, 0.0, ())]

        best_value = -1
        best_solution = [0] * problem.dimension
        nodes_expanded = 0

        num_iters = self.params.get("num_iters", 100)

        while stack:
            if nodes_expanded > num_iters:
                print(f"  [{self.name}] Iteration limit ({num_iters}) reached!")
                break

            idx, cur_w, cur_v, path = stack.pop()
            nodes_expanded += 1
            logger.history["best_fitness"].append(best_value)
            logger.history["current_best"].append(np.array(best_solution))

            if idx == problem.dimension:
                if cur_v > best_value:
                    best_value = cur_v
                    best_solution = list(path)
                    #logger.history["current_best"].append(np.array(best_solution))
                continue


            # DFS: push include trước để đi sâu
            if cur_w + problem.weights[idx] <= problem.capacity:
                stack.append(
                    (idx + 1,
                     cur_w + problem.weights[idx],
                     cur_v + problem.values[idx],
                     path + (1,))
                )

            stack.append((idx + 1, cur_w, cur_v, path + (0,)))

        final_solution = np.array(best_solution)
        cost = problem.evaluate(final_solution)

        logger.log("best_fitness", cost)
        logger.finish(best_solution=final_solution, best_fitness=best_value)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "path": best_solution,
                "best_cost": cost,
                "best_fitness": best_value,
                "nodes_expanded": nodes_expanded,
                "logger": logger
            }
        }

    # ==========================================================
    # PATH FINDING (DFS)
    # ==========================================================
    def _solve_pathfinding(self, problem, seed):
        start = problem.start
        goal = problem.goal

        stack = [start]
        visited = {start}
        predecessor = {start: None}

        nodes_expanded = 0
        logger = Logger(self.name, run_id=seed)
        logger.history["visited_edges"] = [(start, start)]
        logger.history["best_fitness"] = []

        num_iters = self.params.get("num_iters", 100)

        while stack:

            current = stack.pop()
            nodes_expanded += 1

            current_path = self.reconstruct_path(predecessor, current)
            logger.history["best_fitness"].append(problem.evaluate(current_path))

            if current == goal or nodes_expanded > num_iters:
                if nodes_expanded > num_iters:
                    print(f"  [{self.name}] Iteration limit ({num_iters}) reached! Returning progress.")

                path = self.reconstruct_path(predecessor, current)
                cost = problem.evaluate(path)
                fitness = len(path) - 1

                logger.log("best_fitness", cost)
                logger.finish(best_solution=path, best_fitness=fitness)

                return {
                    "time(ms)": logger.meta["runtime"],
                    "result": {"path": path, "cost": cost, "nodes_expanded": len(logger.history["visited_edges"]),
                               "logger": logger}
                }

            for neigh in problem.get_neighbors(current):
                if neigh not in visited:
                    visited.add(neigh)
                    predecessor[neigh] = current
                    stack.append(neigh)
                    logger.history["visited_edges"].append((current, neigh))

        logger.finish(best_solution=[], best_fitness=float('inf'))
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "path": [],
                "cost": float('inf'),
                "nodes_expanded": len(logger.history["visited_edges"]),
                "logger": logger
            }
        }

    def reconstruct_path(self, previous, current):
        path = [current]
        while current in previous:
            current = previous[current]
            if current is None:
                break
            path.append(current)
        return path[::-1]