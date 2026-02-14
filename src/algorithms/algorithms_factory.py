from src.algorithms.classical.a_star import A_Star
from src.algorithms.classical.bfs import BFS
from src.algorithms.classical.dfs import DFS
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithm
from src.algorithms.evolution.differential_evolution import DifferentialEvolution

def get_algorithm(algorithm_name, params=None):
    if params is None: params = {}
    
    match algorithm_name:
        case "A Star":
            return A_Star(params)
        case "BFS":
            return BFS()
        case "DFS":
            return DFS()
        case "Genetic Algorithm":
            return GeneticAlgorithm(params)
        case "Differential Evolution":
            return DifferentialEvolution(params)
        case _:
            raise ValueError(f"Algorithm '{algorithm_name}' not recognized.")