from src.algorithms.classical.a_star import A_Star
from src.algorithms.classical.bfs import BFS

def get_algorithm(algorithm_name):
    match algorithm_name:
        case "A Star":
            return A_Star(params = {"path": []})
        case "Breadth-First Search":
            return BFS()
        case _:
            raise ValueError(f"Algorithm '{algorithm_name}' not recognized.")