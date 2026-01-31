from  src.algorithms.classical.a_star import A_Star
import numpy as np
algorithm = A_Star(params={"path": []})
maze = np.array([[0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0]])
problem = {"matrix": maze,
          "start": np.array([0,0]),
          "goal": np.array([0,7])} # Theo x, y la (3,5); matrix[row][col] la (5,3)
algorithm_result = algorithm.runAlgorithm(problem)

print("Algorithm Name: ",algorithm.getAlgorithmName())
print("Problem: Shortest Path in 8x8 Maze")
print("Time (in milliseconds):", algorithm_result["time(ms)"])
print("Cost:", algorithm_result["result"]["cost"])
print("Path:", algorithm_result["result"]["path"])
print("Matrix:\n", algorithm_result["result"]["simulated matrix"])