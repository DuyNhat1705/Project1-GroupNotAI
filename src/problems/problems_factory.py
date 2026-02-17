from src.problems.discrete.ShortestPathOnGraph import ShortestPathOnGraph
from src.problems.discrete.ShortestPathOnMaze import ShortestPathOnMaze
from src.problems.continuous.continuous import Sphere, Rosenbrock, Ackley, Griewank, Rastrigin, Michalewicz
from src.problems.discrete.TSP import TravelSalesmanProblem
from src.problems.discrete.knapsack import KnapsackProblem


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

        "knapsack": KnapsackProblem,

        "tsp": TravelSalesmanProblem,


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
    
    # Discrete problems (Graph/Maze) thường tự load file hoặc
    # có tham số riêng
    # Ví dụ Graph có thể nhận 'directed'
    if name_key in ["graph", "shortestpathongraph"]:
        directed = kwargs.get('directed', True)
        seed = kwargs.get('seed', 42)
        return ProblemClass(directed=directed, seed=seed)

    # Mặc định
    return ProblemClass()