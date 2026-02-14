from src.problems.discrete.ShortestPathOnGraph import ShortestPathOnGraph
from src.problems.discrete.ShortestPathOnMaze import ShortestPathOnMaze
from src.problems.continuous.continuous import Sphere, Rosenbrock, Ackley, Griewank, Rastrigin, Michalewicz

def get_problem(problem_name):
    match problem_name:
        # --- Discrete Problems ---
        case "Shortest Path on Graph":
            return ShortestPathOnGraph(directed=True, seed=42)
        case "Shortest Path on Maze":
            return ShortestPathOnMaze()
            
        # --- Continuous Problems (Update phần này) ---
        case "Sphere":
            return Sphere(dimension=2)
        case "Rosenbrock":
            return Rosenbrock(dimension=2)
        case "Ackley":
            return Ackley(dimension=2)
        case "Griewank":
            return Griewank(dimension=2)
        case "Rastrigin":
            return Rastrigin(dimension=2)
        case "Michalewicz":
            return Michalewicz(dimension=2)
            
        case _:
            raise ValueError(f"Problem '{problem_name}' not recognized.")