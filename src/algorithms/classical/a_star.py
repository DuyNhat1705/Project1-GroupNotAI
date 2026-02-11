from ..base_algorithm import BaseAlgorithm
import heapq
import numpy as np
from src.utils.logger import Logger
class A_Star(BaseAlgorithm):
    def __init__(self,params):
        # params (dictionary) contains array for optimal path 
        super().__init__("A Star", params)
    
    # Metaheuristic function
    def euclidean_distance(self, a, b):
        return np.linalg.norm(a-b)
    
    # A* algorithm
    def solve(self, problem, seed=None):
        # Problem (dictionary) contains matrix, goal (np.array(x,y)), start (np.array(x,y))
        prior_queue = []
        start = problem.start
        goal = problem.goal
        maze = problem.maze
        x, y = maze.shape

        move = np.array ([[0,1],[1,0],[0,-1],[-1,0]])
        
        logger = Logger(self.name)
        logger.history["visited_nodes"] = []
        if maze[goal[1]][goal[0]] == 1 or maze[start[1]][start[0]] == 1:
            logger.finish(best_solution=[], best_fitness=float('inf'))
            return {"time(ms)": logger.meta["runtime"],
                    "result": {"cost": 0, "path": [], "logger": logger}}
        
        came_from = {}
        close_states = set() # admissible heuristic function: euclidean distance
        best_g = {tuple(start): 0}

        g_cost= 0
        heapq.heappush(prior_queue, (g_cost + self.euclidean_distance(start, goal), tuple(start)))
        while prior_queue:
            curr_state = heapq.heappop(prior_queue)[1]
            g_cost = best_g[curr_state] + 1
            # print("curr_state", curr_state)
            if curr_state in close_states:
                continue
            logger.history["visited_nodes"].append(curr_state)
            if curr_state == tuple(goal):
                break
            close_states.add(curr_state)

            for step in move:
                adjacency =  np.array(curr_state) + step
                if (adjacency[0] >= 0 and adjacency[0] < x) and (adjacency[1] >= 0 and adjacency[1] < y) and (maze[adjacency[1]][adjacency[0]] == 0) and (tuple(adjacency) not in close_states):
                    f_cost = g_cost + self.euclidean_distance(adjacency, goal)
                    key = tuple(adjacency)
                    if (key not in best_g) or (g_cost < best_g[key]):
                        # print("adjacency", key)
                        best_g[key] = g_cost
                        came_from[key] = curr_state
                        heapq.heappush(prior_queue, (f_cost, key))
                        best_g[key] = g_cost
            
        # g_cost need to be correspond to the curr_state
        self.reconstruct_path(came_from, start, goal)
        logger.finish(best_solution=self.params["path"], best_fitness=best_g[tuple(goal)])

        return {"time(ms)": logger.meta["runtime"],
                "result": {"cost": best_g[tuple(goal)], "path": self.params["path"], "logger": logger}}
    
    # Reconstruct optimal path
    def reconstruct_path(self, came_from, start, goal):
        self.params["path"] = []
        # print(came_from)
        node = tuple(goal)
        while node != tuple(start):
            self.params["path"].append(node)
            node = came_from[node]
        self.params["path"].append(tuple(start))
        self.params["path"].reverse()
        
    # Format matrix with optimal path for visualization
    def formatMatrixWithOptimalPath(self, matrix):
        for node in self.params["path"]:
            matrix[node[1]][node[0]] = 8
        return matrix


