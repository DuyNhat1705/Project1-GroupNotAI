# src/HandleCLI.py
import argparse

def parse_param_string(param_list):
    """
    Chuyển list ['key=value', 'a=1'] thành dict {'key': 'value', 'a': 1}
    Tự động ép kiểu int/float.
    """
    params = {}
    if not param_list:
        return params

    for item in param_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        
        # Thử ép kiểu số
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass # Giữ nguyên string nếu không phải số (VD: path="...")
            
        params[key] = value
    return params
def handleCLI():
    parser = argparse.ArgumentParser(description="AI Search Algorithm Runner")

    # Các tham số bắt buộc
    parser.add_argument("--algo", type=str, required=True, help="Tên thuật toán (VD: ga, de, bfs, astar)")
    parser.add_argument("--problem", type=str, required=True, help="Tên bài toán (VD: sphere, maze, graph)")

    # Các tham số tùy chọn
    parser.add_argument("--dim", type=int, default=2, help="Số chiều (cho bài toán Continuous)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Tham số động cho thuật toán (VD: pop_size=100 F=0.8)
    parser.add_argument("--params", nargs='*', help="Các tham số thuật toán dạng key=value (VD: pop_size=50 F=0.8 CR=0.9)")

    args = parser.parse_args()

    # 1. Parse Params
    algo_params = parse_param_string(args.params)
    print(f"\n[INFO] Run Configuration:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Problem  : {args.problem}")
    print(f"  Dimension: {args.dim}")
    print(f"  Params   : {algo_params}")

    return args, algo_params
def get_problem_type(problem):
    """
    Identify the problem type based on problem name and properties.
    Returns: problem type string that matches COMPATIBILITY keys
    """
    # Check if continuous
    if hasattr(problem, 'cont_flag') and problem.cont_flag:
        return "Continuous"
    
    # Check by problem name
    problem_name = problem.name
    if "TSP" in problem_name or "Traveling" in problem_name:
        return "TSP"
    elif "Maze" in problem_name or "ShortestPathOnMaze" in problem_name:
        return "ShortestPathOnMaze"
    elif "Knapsack" in problem_name:
        return "Knapsack"
    elif "Graph Coloring" in problem_name or "Coloring" in problem_name:
        return "GraphColoring"
    elif "Graph" in problem_name or "ShortestPath" in problem_name:
        return "ShortestPathOnGraph"
    
    return problem_name

def check_compatibility(algorithm_name, problem_type):
    """
    Check if algorithm is compatible with problem type.
    Returns: (is_compatible, compatible_algorithms_for_problem, compatible_problems_for_algorithm)
    """
    # Define compatibility mapping between algorithm names and problem types
    COMPATIBILITY = {
        "Continuous": [
            "Artificial Bee Colony", "Simulated Annealing", "Hill Climbing", 
            "Particle Swarm Optimization", "Cuckoo Search", "Firefly Algorithm", 
            "Ant Colony Optimization", "Genetic Algorithm", "Differential Evolution", "TLBO"
        ],
        "TSP": [
            "Simulated Annealing", "Hill Climbing", "Particle Swarm Optimization", 
            "Firefly Algorithm", "Ant Colony Optimization", "Genetic Algorithm", "A Star"
        ],
        "ShortestPathOnMaze": ["A Star", "Breadth-First Search", "Depth-First Search", "Genetic Algorithm"],
        "Knapsack": [
            "Artificial Bee Colony", "Breadth-First Search", "Cuckoo Search", "TLBO", "Breadth-First Search", "Depth-First Search"
        ],
        "GraphColoring": ["Breadth-First Search", "Depth-First Search"],
        "ShortestPathOnGraph": ["Breadth-First Search", "Depth-First Search"]
    }
    
    # Get compatible algorithms for this problem type
    compatible_algos = COMPATIBILITY.get(problem_type, [])
    
    # Get all problem types where this algorithm is compatible
    compatible_problems = []
    for prob_type, algos in COMPATIBILITY.items():
        if algorithm_name in algos:
            compatible_problems.append(prob_type)

    if "Continuous" in compatible_problems:
        compatible_problems.remove("Continuous")
        compatible_problems.extend([
            "Sphere",
            "Rosenbrock",
            "Ackley",
            "Griewank",
            "Rastrigin",
            "Michalewicz"
        ])
    
    # Check compatibility
    is_compatible = algorithm_name in compatible_algos
    
    return is_compatible, compatible_algos, compatible_problems