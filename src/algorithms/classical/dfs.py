from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class DFS(BaseAlgorithm):
    def __init__(self, params=None):
        super().__init__("Depth-First Search", params if params else {})

    def solve(self, problem, seed=None):
        start = problem.start
        goal = problem.goal

        stack = [start] 
        visited = {start} # visited nodes
        predecessor = {start: None} # reconstruct path

        logger = Logger(self.name, run_id=seed)
        logger.history["visited_edges"] = [] # Giữ nguyên key logger để visualize
        logger.history["visited_edges"].append((start, start))
        ite = 0  # iteration count

        while len(stack) > 0:
            current = stack.pop() 
            ite += 1

            if current == goal:
                path = self.reconstruct_path(predecessor, current)
                cost = problem.evaluate(path)
                fitness = len(path) - 1
                
                # Log kết quả y chang BFS
                logger.log("cost", cost)
                logger.finish(best_solution=path, best_fitness=fitness)
                return {
                    "time(ms)": logger.meta["runtime"],
                    "result": {"path": path, "cost": cost, "logger": logger}
                }

            neighbors = problem.get_neighbors(current)
            
            for neigh in neighbors:
                if neigh not in visited:
                    visited.add(neigh)
                    predecessor[neigh] = current
                    stack.append(neigh) 

                    logger.history["visited_edges"].append((current, neigh))

        logger.finish(best_solution=[], best_fitness=float('inf'))
        return {
            "time(ms)": logger.meta["runtime"],
            "result": {"path": [], "cost": float('inf'), "logger": logger}
        }

    def reconstruct_path(self, previous, current):
        complete_path = [current]
        while current in previous:
            current = previous[current]
            if current is None:
                break
            complete_path.append(current)
        complete_path.reverse()
        return complete_path