from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.maze_visualizer import MazeVisualizer
from src.visualization.continuous_visualizer import ContinuousVisualizer

def get_visualizer(params):
    problem = params.get("problem")
    result = params.get("result", {})
    
    # 1. CONTINUOUS PROBLEMS (Sphere, Ackley...)
    if hasattr(problem, 'cont_flag') and problem.cont_flag:
        best_fit = result.get("best_fitness", None)
        print(f"  -> Done. Best Fitness: {best_fit}")
        # Lấy lịch sử population từ logger
        logger = result.get("logger", None)
        history = logger.history.get("population", []) if logger else []
        
        # Tiêu đề
        algo_name = params.get("algorithm", "Unknown Algo")
        title = f"{algo_name} on {problem.getName()}"
        
        # Lấy dữ liệu Metrics (cho Convergence Plot)
        metrics = {
            "best_fit": result.get("best_hist", []),
            "avg_fit": result.get("avg_hist", [])
        }
        
        # Lấy dữ liệu Grid Search (cho 3 biểu đồ heatmap)
        grid_data = result.get("grid_data", None)

        return ContinuousVisualizer(problem, history, title, metrics, grid_data)

    # 2. DISCRETE PROBLEMS (Graph, Maze)
    else:
        problem_name = problem.getName()
        print(f"  -> Done. Cost: {result.get('cost', 'N/A')}")
        match problem_name:
            case "ShortestPathOnGraph":
                history = result.get("logger", None).history.get("visited_edges", [])
                path = result.get("path", [])
                title = params.get("algorithm") + " Visualization"
                return GraphVisualizer(problem, history, path, title)
                
            case "ShortestPathOnMaze":
                history = result.get("logger", None).history.get("visited_edges", [])
                path = result.get("path", [])
                title = params.get("algorithm") + " Visualization"
                return MazeVisualizer(problem, history, path, title)
            
            case _:
                raise ValueError(f"Visualization not implemented for {problem_name}")