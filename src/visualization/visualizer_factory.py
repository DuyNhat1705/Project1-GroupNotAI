from src.visualization.TSP_viz import TSPVisualizer
from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.maze_visualizer import MazeVisualizer
from src.visualization.continuous_visualizer import ContinuousVisualizer
from src.visualization.knapsack_viz import KnapsackVisualizer
from src.visualization.graph_color_viz import GraphColoringVisualizer

def get_visualizer(params):
    problem = params.get("problem")
    result = params.get("result", {})
    
    # 1. CONTINUOUS PROBLEMS
    if hasattr(problem, 'cont_flag') and problem.cont_flag:
        best_fit = result.get("best_fitness", None)
        print(f"  -> Done. Best Fitness: {best_fit}")
        
        logger = result.get("logger", None)
        if logger:
            history = logger.history.get("population",
                      logger.history.get("explored",
                      logger.history.get("current_best", [])))
        else:
            history = []
        
        algo_name = params.get("algorithm", "Unknown Algo")
        title = f"{algo_name} on {problem.getName()}"
        
        if logger:
            metrics = {
                "best_fitness": logger.history.get("best_fitness", []),
                "avg_fitness": logger.history.get("avg_fitness", [])
            }
        else:
            metrics = {}
        
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
                logger = result.get("logger")
                history = logger.history.get("visited_edges", [])
                path = result.get("path", [])
                title = params.get("algorithm") + " Visualization"
                return MazeVisualizer(problem, history, path, title)
            
            case "Knapsack Problem":
                history = result.get("logger").history.get("current_best", [])
                title = params.get("algorithm") + " Visualization"
                return KnapsackVisualizer(problem, history, path=None, title=title)

            case "TSP":
                logger = result.get("logger")
                history = logger.history.get("explored", [])
                title = params.get("algorithm", "Unknown Algo") + " on TSP"
                return TSPVisualizer(problem, history, path=None, title=title)

            case "Graph Coloring":
                logger = result.get("logger")
                history = result.get("logger").history.get("current_best", [])
                title = params.get("algorithm") + " Visualization"
                return GraphColoringVisualizer(problem, history, path=None, title=title)

            case _:
                raise ValueError(f"Visualization not implemented for {problem_name}")