# Import Factory
from matplotlib.pylab import rint

from src.HandleCLI import handleCLI
from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm
from src.visualization.visualizer_factory import get_visualizer

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
            "Firefly Algorithm", "Ant Colony Optimization", "Genetic Algorithm"
        ],
        "ShortestPathOnMaze": ["A Star", "Breadth-First Search", "Depth-First Search"],
        "Knapsack": [
            "Artificial Bee Colony", "Breadth-First Search", "Cuckoo Search", "TLBO"
        ],
        "GraphColoring": ["Breadth-First Search", "Depth-First Search"],
        "ShortestPathOnGraph": ["A Star", "Breadth-First Search", "Depth-First Search"]
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

def main():
    #1. Xử lý CLI
    args, algo_params = handleCLI()

    # 2. Khởi tạo Problem
    problem = get_problem(args.problem, dimension=args.dim, seed=args.seed)

    # 3. Khởi tạo Algorithm
    algorithm = get_algorithm(args.algo, **algo_params)

    # 4. Kiểm tra tương thích
    problem_type = get_problem_type(problem)
    is_compatible, compatible_algos, compatible_problems = check_compatibility(
        algorithm.name, problem_type
    )

    if is_compatible:
        # 5. Chạy thuật toán
        print(f"\n[INFO] Solving...")
        
        raw_output = algorithm.solve(problem, seed=args.seed)
        
        # Chuẩn hóa dữ liệu để gửi cho Visualizer
        viz_data = {
            "problem": problem,
            "algorithm": algorithm.name,
            "result": raw_output["result"],
            "context": args.problem
        }
        visualizer = get_visualizer(viz_data)
        # In thời gian thực thi ngay sau khi solve xong
        print(f"Time(ms): {raw_output.get('time(ms)', 'N/A')}")
        
        # 5. Visualization
        print(f"\n[INFO] Visualizing...")
        
        if visualizer:
            visualizer.animate()
            # Nếu là bài toán Continuous, có thể gọi thêm analyze_performance nếu có data
            if hasattr(visualizer, 'analyze_performance'):
                # Lưu ý: CLI chạy 1 lần nên không có Grid Search Data để vẽ heatmap
                # Nhưng vẫn vẽ được Convergence Curve nếu logger có lưu history fitness
                pass
        else:
            print("-> No visualizer available for this combination.")

    else: 
        # Incompatibility detected - provide helpful information
        print(f"\n[ERROR] Algorithm '{algorithm.name}' is NOT compatible with '{problem.name}'")
        print(f"Compatible problems for '{algorithm.name}' are: {', '.join(compatible_problems)}")

if __name__ == "__main__":
    main()