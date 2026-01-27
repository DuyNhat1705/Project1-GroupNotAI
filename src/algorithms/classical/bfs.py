from src.algorithms.base_algorithm import BaseAlgorithm
from src.utils.logger import Logger


class BFS(BaseAlgorithm):
    def __init__(self, params=None):

        super().__init__("Breadth-First Search", params if params else {})


    def solve(self, problem, seed=None):

        start = problem.start
        goal = problem.goal

        queue = [start] # queue of frontier
        visited = {start} # visited nodes
        predecessor = {start: None} # reconstruct path from root to current node

        logger = Logger(self.name, run_id=seed)
        logger.history["visited_edges"] = []
        ite = 0  # iteration count

        while len(queue) > 0:
            current = queue.pop(0) # pop front
            ite += 1

            if current == goal:
                path= self.reconstruct_path(predecessor, current, problem)
                cost = problem.evaluate(path)
                fitness = len(path) - 1 # not take weight in account
                logger.log("cost", cost)
                logger.finish(best_solution=path, best_fitness=fitness)
                return path, cost, logger

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
        return [], float('inf'), logger

    def reconstruct_path(self, previous, current, problem):
        complete_path = [current]
        while current in previous:
            current = previous[current]
            if current is None:
                break
            complete_path.append(current)
        complete_path.reverse()
        return complete_path