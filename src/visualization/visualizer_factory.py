from unittest import case

from src.visualization.TSP_viz import TSPVisualizer
from src.visualization.TSP_GA_viz import TSPGAVisualizer
from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.maze_visualizer import MazeVisualizer
from src.visualization.continuous_visualizer import ContinuousVisualizer
from src.visualization.knapsack_viz import KnapsackVisualizer
from src.visualization.graph_color_viz import GraphColoringVisualizer
from src.visualization.maze_ga_visualizer import GAMazeVisualizer

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
        title = f"{algo_name} on {problem.name}"
        
        if logger:
            metrics = {
                "best_fitness": logger.history.get("best_fitness", []),
                "avg_fitness": logger.history.get("avg_fitness", []),
            }
        else:
            metrics = {}
        
        grid_data = result.get("grid_data", None)

        return ContinuousVisualizer(problem, history, title, metrics, grid_data)

    # 2. DISCRETE PROBLEMS (Graph, Maze)
    else:
        problem_name = problem.name
        match problem_name:
            case "ShortestPathOnGraph":
                print(f"  -> Done. Cost: {result.get('cost', 'N/A')}")
                history = result.get("logger", None).history.get("visited_edges", [])
                path = result.get("path", [])
                num = params.get("context", "graph1")[5:]
                title = params.get("algorithm") + " on Graph " + num
                return GraphVisualizer(problem, history, path, title)
                
            case "ShortestPathOnMaze":
                logger = result.get("logger")
                algo_name = params.get("algorithm", "").lower()
                num = params.get("context", "maze1")[4:]
                title = params.get("algorithm") + " on Maze " + num

                # Phân nhánh cho Genetic Algorithm
                if "genetic" in algo_name or "ga" in algo_name:
                    print(f"  -> Done. Best Fitness: {result.get('best_fitness', 'N/A')}")
                    # Đối với GA, history là mảng các best_solution qua từng thế hệ
                    history = logger.history.get("best_solution", [])
                    # Lấy kết quả đường đi tốt nhất cuối cùng
                    path = result.get("best_solution", []) 
                    
                    return GAMazeVisualizer(problem, history, path, title)
                
                # Phân nhánh cho các thuật toán truyền thống (BFS, DFS, A*,...)
                else:
                    print(f"  -> Done. Cost: {result.get('cost', 'N/A')}")
                    history = logger.history.get("visited_edges", [])
                    path = result.get("path", [])
                    
                    return MazeVisualizer(problem, history, path, title)
            
            case "Knapsack Problem":
                print(f"  -> Done. Best Fitness: {result.get('best_fitness', None)}")
                history = result.get("logger").history.get("current_best", [])
                num = params.get("context", "knapsack1")[8:]
                title = params.get("algorithm") + " on Knapsack " + num
                return KnapsackVisualizer(problem, history, path=None, title=title)

            case "TSP":
                print(f"  -> Done. Best Fitness: {result.get('best_fitness', None)}")
                logger = result.get("logger")
                history = logger.history.get("population", []) or logger.history.get("explored", [])
                num = params.get("context", "tsp1")[3:]
                algo_name = params.get("algorithm", "").lower()
                title = params.get("algorithm", "Unknown Algo") + " on TSP " + num

                if "genetic" in algo_name or "ga" in algo_name:
                    return TSPGAVisualizer(
                        problem,
                        history,
                        logger=logger,
                        title=title
                    )
                else:
                    return TSPVisualizer(
                        problem,
                        history,
                        path=None,
                        title=title
                    )

            case "Graph Coloring":
                logger = result.get("logger")
                history = result.get("logger").history.get("current_best", [])
                num = params.get("context", "coloring1")[8:]
                title = params.get("algorithm") + " on Graph Coloring " + num
                return GraphColoringVisualizer(problem, history, path=None, title=title)

            case _:
                raise ValueError(f"Visualization not implemented for {problem_name}")