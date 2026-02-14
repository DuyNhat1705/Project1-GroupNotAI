# src/HandleCLI.py

# --- IMPORTS: PROBLEMS ---
from src.problems.discrete.ShortestPathOnGraph import ShortestPathOnGraph
from src.problems.discrete.ShortestPathOnMaze import ShortestPathOnMaze
from src.problems.continuous.continuous import Sphere, Rosenbrock, Ackley, Griewank, Rastrigin, Michalewicz

# --- IMPORTS: ALGORITHMS ---
from src.algorithms.classical.bfs import BFS
from src.algorithms.classical.dfs import DFS
from src.algorithms.classical.a_star import A_Star
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithm
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.classical.hill_climbing import HillClimbing
from src.algorithms.nature.simulated_annealing import SimulatedAnnealing
from src.algorithms.nature.artificial_bee import ArtificialBee


def get_problem(name, **kwargs):
    """
    Factory for Problems.
    name: Tên bài toán (VD: "sphere", "maze")
    kwargs: Các tham số khởi tạo (VD: dimension=2, directed=True)
    """
    name_key = name.lower().strip().replace(" ", "")
    
    problems = {
        # --- Discrete ---
        "shortestpathongraph": ShortestPathOnGraph,
        "graph": ShortestPathOnGraph,
        
        "shortestpathonmaze": ShortestPathOnMaze,
        "maze": ShortestPathOnMaze,

        # --- Continuous ---
        "sphere": Sphere,
        "rosenbrock": Rosenbrock,
        "ackley": Ackley,
        "griewank": Griewank,
        "rastrigin": Rastrigin,
        "michalewicz": Michalewicz,
    }

    if name_key not in problems:
        raise ValueError(f"Problem '{name}' not found. Available: {list(problems.keys())}")

    # Lấy class
    ProblemClass = problems[name_key]

    # Xử lý tham số đặc thù cho từng loại
    # Continuous problems cần 'dimension'
    if hasattr(ProblemClass, 'cont_flag') or name_key in ["sphere", "rosenbrock", "ackley", "griewank", "rastrigin", "michalewicz"]:
        dim = kwargs.get('dimension', 2) # Default dim = 2 nếu không truyền
        return ProblemClass(dimension=dim)
    
    # Discrete problems (Graph/Maze) thường tự load file hoặc có tham số riêng
    # Ví dụ Graph có thể nhận 'directed'
    if name_key in ["graph", "shortestpathongraph"]:
        directed = kwargs.get('directed', True)
        seed = kwargs.get('seed', 42)
        return ProblemClass(directed=directed, seed=seed)

    # Mặc định
    return ProblemClass()


def get_algorithm(name, **kwargs):
    """
    Factory for Algorithms.
    name: Tên thuật toán (VD: "ga", "bfs")
    kwargs: Dictionary chứa params (VD: {'pop_size': 50, 'F': 0.5})
    """
    name_key = name.lower().strip().replace(" ", "").replace("-", "").replace("_", "")

    algos = {
        # --- Classical ---
        "bfs": BFS,
        "breadthfirstsearch": BFS,
        
        "dfs": DFS,
        "depthfirstsearch": DFS,
        
        "astar": A_Star,
        "a_star": A_Star,

        # --- Nature Inspired ---
        "geneticalgorithm": GeneticAlgorithm,
        "ga": GeneticAlgorithm,
        
        "differentialevolution": DifferentialEvolution,
        "de": DifferentialEvolution,
        
        "hillclimbing": HillClimbing,
        "hc": HillClimbing,
        
        "simulatedannealing": SimulatedAnnealing,
        "sa": SimulatedAnnealing,
        
        "artificialbeecolony": ArtificialBee,
        "abc": ArtificialBee,
    }

    if name_key not in algos:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(algos.keys())}")

    # Lấy class
    AlgoClass = algos[name_key]
    
    # kwargs ở đây chính là dict 'params' mà các class Algorithm yêu cầu
    # Một số thuật toán nature nhận params={...}, classical cũng nhận params={...}
    # Ta truyền thẳng kwargs vào tham số 'params' của constructor
    return AlgoClass(params=kwargs)