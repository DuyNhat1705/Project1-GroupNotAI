from src.algorithms.base_algorithm import BaseAlgorithm
import heapq
import numpy as np
from src.utils.logger import Logger


class A_Star(BaseAlgorithm):
    def __init__(self, params=None):
        # params (dictionary) contains array for optimal path
        super().__init__("A Star", params if params else {})
        if "path" not in self.params:
            self.params["path"] = []

    # Metaheuristic function
    def euclidean_distance(self, a, b):
        # Cast to array to prevent tuple subtraction crash
        return np.linalg.norm(np.array(a) - np.array(b))

    # A* algorithm
    def solve(self, problem, seed=None):
        # Problem (dictionary) contains matrix, goal, start
        prior_queue = []
        start = tuple(problem.start)
        goal = tuple(problem.goal)
        maze = problem.maze
        x, y = maze.shape

        move = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

        logger = Logger(self.name, run_id=seed)

        # change name into "visited_edges" for the visualizer
        logger.history["visited_edges"] = []

        # standard NumPy indexing [row, col]
        if maze[goal[0], goal[1]] == 1 or maze[start[0], start[1]] == 1:
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"cost": float('inf'), "path": [], "logger": logger}}

        came_from = {}
        close_states = set()  # admissible heuristic function: euclidean distance
        best_g = {start: 0}

        g_cost = 0
        heapq.heappush(prior_queue, (g_cost + self.euclidean_distance(start, goal), start))

        # Log start node for the visualizer
        logger.history["visited_edges"].append((start, start))

        while prior_queue:
            curr_state = heapq.heappop(prior_queue)[1]
            g_cost = best_g[curr_state] + 1

            if curr_state in close_states:
                continue

            # log the edge for the visualizer
            parent = came_from.get(curr_state, curr_state)
            logger.history["visited_edges"].append((parent, curr_state))

            if curr_state == goal:
                break
            close_states.add(curr_state)

            for step in move:
                adjacency = np.array(curr_state) + step

                # check bounds and use correct maze indexing
                if (adjacency[0] >= 0 and adjacency[0] < x) and (adjacency[1] >= 0 and adjacency[1] < y) and (
                        maze[adjacency[0], adjacency[1]] == 0) and (tuple(adjacency) not in close_states):

                    f_cost = g_cost + self.euclidean_distance(adjacency, goal)
                    key = tuple(adjacency)

                    if (key not in best_g) or (g_cost < best_g[key]):
                        best_g[key] = g_cost
                        came_from[key] = curr_state
                        heapq.heappush(prior_queue, (f_cost, key))
                        best_g[key] = g_cost

        # g_cost need to be correspond to the curr_state
        self.reconstruct_path(came_from, start, goal)

        # handle cost if goal wasn't reached
        final_cost = best_g[goal] if goal in best_g else float('inf')

        logger.finish(best_solution=self.params["path"], best_fitness=final_cost)

        return {
            "time(ms)": logger.meta["runtime"],
            "result": {
                "cost": final_cost,
                "path": self.params["path"],
                "nodes_expanded": len(logger.history["visited_edges"]),  # <-- DISCRETE METRIC
                "logger": logger
            }
        }

    # Reconstruct optimal path
    def reconstruct_path(self, came_from, start, goal):
        self.params["path"] = []
        if goal not in came_from:
            return

        node = goal
        while node != start:
            self.params["path"].append(node)
            node = came_from[node]
        self.params["path"].append(start)
        self.params["path"].reverse()

    # format matrix with optimal path for visualization
    def formatMatrixWithOptimalPath(self, matrix):
        for node in self.params["path"]:
            matrix[node[0], node[1]] = 8 #modified index
        return matrix