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

    # Heuristic function: Euclidean distance between two points
    def euclidean_distance(self, a, b):
        # Cast to array to prevent tuple subtraction crash
        return np.linalg.norm(np.array(a) - np.array(b)) # linear algebra norm

    # A* algorithm for Maze
    def solveMaze(self, problem, seed=None):
        # Problem (dictionary) contains matrix, goal, start
        prior_queue = []
        start = tuple(problem.start)
        goal = tuple(problem.goal)
        maze = problem.maze
        x, y = maze.shape # x = num_rows, y = num_cols

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
    def mst_cost(self, nodes, dist):
        """Compute MST cost using Prim algorithm."""
        if len(nodes) <= 1:
            return 0

        nodes = list(nodes)
        visited = {nodes[0]}
        total_cost = 0
        edges = []

        for v in nodes[1:]:
            heapq.heappush(edges, (dist[nodes[0]][v], v))

        while edges and len(visited) < len(nodes):
            cost, v = heapq.heappop(edges)

            if v in visited:
                continue

            visited.add(v)
            total_cost += cost

            for u in nodes:
                if u not in visited:
                    heapq.heappush(edges, (dist[v][u], u))

        return total_cost
    # Heuristic function: MST cost of unvisited nodes + min distance from current to unvisited + min distance from unvisited to start
    def mst_heuristic(self, curr, mask, dist, start, n):

        unvisited = [i for i in range(n) if not (mask & (1 << i))]

        if not unvisited:
            return dist[curr][start]

        mst = self.mst_cost(unvisited, dist)

        min_from_curr = min(dist[curr][u] for u in unvisited)

        min_to_start = min(dist[u][start] for u in unvisited)

        return mst + min_from_curr + min_to_start
    # A* algorithm for TSP
    def solveTSP(self, problem, seed=None):

        # Extract parameters and problem details
        num_iters = self.params.get("num_iters", True)
        dist = problem.dist_mat
        n = problem.dimension

        # Initialization
        start = 0
        FULL_MASK = (1 << n) - 1
        prior_queue = []
        heapq.heappush(prior_queue, (0, 0, start, 1 << start))
        best_g = {(start, 1 << start): 0}
        parent = {}
        close_states = set()
        goal_state = None
        start_state = (start, 1 << start)
        parent[start_state] = None
        dummy_partial_path = []

        logger = Logger(self.name, run_id=seed)
        logger.history["best_fitness"] = []
        logger.history["explored"] = []

        while num_iters > 0 and prior_queue:
            f, g, city, mask = heapq.heappop(prior_queue)
            state_id = (city, mask)
            if state_id in close_states:
                continue
            close_states.add(state_id)
            cur_path = self.reconstruct_partial(parent, state_id)

            # Log g (cost of cur_path) as best_fitness, cur_path as explored
            logger.log("best_fitness", g)
            logger.log("explored", ([dummy_partial_path.copy(), cur_path.copy()], g))

            if mask == FULL_MASK:
                goal_state = state_id
                break
            
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue

                new_mask = mask | (1 << nxt)
                new_g = g + dist[city][nxt]

                state = (nxt, new_mask)

                if state not in best_g or new_g < best_g[state]:
                    best_g[state] = new_g
                    parent[state] = (city, mask)
                    h = self.mst_heuristic(nxt,new_mask,dist,start,n)
                    heapq.heappush(prior_queue, (new_g + h, new_g, nxt, new_mask))
            
            num_iters = num_iters - 1 if isinstance(num_iters, int) else num_iters

        if goal_state is not None:
            best_path = self.reconstruct_partial(parent, goal_state)
            best_path.append(start)
        else:
            best_path = []

        cost = problem.evaluate(best_path) if best_path else float("inf")

        logger.finish(best_solution=best_path, best_fitness=cost)
        return {"time(ms)": logger.meta["runtime"],
                "result": {"best_fitness": cost,"best_solution": best_path,"logger": logger}}

    def solve(self, problem, seed=None):
        if problem.getName() == "TSP":
            return self.solveTSP(problem, seed)
        else:
            return self.solveMaze(problem, seed)
    
    # Reconstruct optimal path
    def reconstruct_partial(self, parent, state):
        path = []

        while state is not None:
            city, _ = state
            path.append(city)
            state = parent.get(state)

        path.reverse()
        return path
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
