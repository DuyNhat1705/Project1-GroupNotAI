from src.visualization.graph_visualizer import GraphVisualizer
from src.visualization.maze_visualizer import MazeVisualizer

def get_visualizer(params):
    problem = params.get("problem")
    problem_name = problem.getName()
    match problem_name:
        case "ShortestPathOnGraph":
            history = params.get("result", []).get("logger", None).history.get("visited_edges", [])
            path = params.get("result", []).get("path", [])
            title = params.get("algorithm") + " Visualization"
            return GraphVisualizer(problem, history, path, title)
        case "ShortestPathOnMaze":
            history = params.get("result", []).get("logger", None).history.get("visited_nodes", [])
            path = params.get("result", []).get("path", [])
            title = params.get("algorithm") + " Visualization"
            return MazeVisualizer(problem, history, path, title)
