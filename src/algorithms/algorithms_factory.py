from src.algorithms.classical.bfs import BFS
from src.algorithms.classical.dfs import DFS
from src.algorithms.classical.a_star import A_Star
from src.algorithms.evolution.genetic_algorithm import GeneticAlgorithm
from src.algorithms.evolution.differential_evolution import DifferentialEvolution
from src.algorithms.classical.hill_climbing import HillClimbing
from src.algorithms.physics.simulated_annealing import SimulatedAnnealing
from src.algorithms.biology.artificial_bee import ArtificialBee
from src.algorithms.biology.cuckoo_search import CS
from src.algorithms.human.tlbo import TLBO

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

        "tlbo": TLBO,
        "teachinglearningbasedoptimization": TLBO,

        "cs": CS,
        "cuckoosearch": CS
    }

    if name_key not in algos:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(algos.keys())}")

    # Lấy class
    AlgoClass = algos[name_key]
    
    # kwargs ở đây chính là dict 'params' mà các class Algorithm yêu cầu
    # Một số thuật toán nature nhận params={...}, classical cũng nhận params={...}
    # Ta truyền thẳng kwargs vào tham số 'params' của constructor
    return AlgoClass(params=kwargs)