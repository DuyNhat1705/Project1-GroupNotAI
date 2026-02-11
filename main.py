from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm
from src.visualization.visualizer_factory import get_visualizer

if __name__ == "__main__":
    algorithm = get_algorithm("A Star")
    problem = get_problem("Shortest Path on Maze")

    answer = algorithm.solve(problem)
    
    visualizer_params = {
        "problem": problem,
        "result": answer["result"],
        "algorithm": algorithm.getAlgorithmName()
    }
    print("Time (in milliseconds):", answer["time(ms)"])
    print("History:", answer["result"]["logger"].history["visited_nodes"])
    visualizer  = get_visualizer(visualizer_params)
    visualizer.animate()

    # print("Algorithm Name: ",algorithm.getAlgorithmName())
    # print("Problem: Shortest Path in 8x8 Maze")
    # print("Time (in milliseconds):", answer["time(ms)"])
    # print("Cost:", answer["result"]["cost"])
    # print("Path:", answer["result"]["path"])
    # print("Matrix:\n", answer["result"]["simulated matrix"])