from src.problems.discrete.ShortestPathOnGraph import ShortestPathOnGraph
from src.problems.discrete.ShortestPathOnMaze import ShortestPathOnMaze

def get_problem(problem_name):
    match problem_name:
        case "Shortest Path on Graph":
            return ShortestPathOnGraph(directed=True, seed=42)
        case "Shortest Path on Maze":
            return ShortestPathOnMaze()
        case _:
            raise ValueError(f"Problem '{problem_name}' not recognized.")