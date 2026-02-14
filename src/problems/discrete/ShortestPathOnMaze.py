from src.problems.base_problem import BaseProblem
import numpy as np
import os
class ShortestPathOnMaze(BaseProblem):
    def __init__(self):
        super().__init__(name="ShortestPathOnMaze", dimension=2)
        self.maze = None
        self.start = None
        self.goal = None

        self.data_path = os.path.join(BaseProblem.project_root, 'data', '8x8_maze.txt')
        self.load_from_file(self.data_path)
        
        
    def load_from_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines 
             if line.strip() and not line.strip().startswith('#')]
            
            self.start = np.array(tuple(map(int, lines[0].split())))
            self.goal  = np.array(tuple(map(int, lines[1].split())))

            matrix_data = [list(map(int, line.split())) for line in lines[2:]]
            self.maze = np.array(matrix_data)
    # def getProblem(self):
    #     return {
    #         'matrix': self.maze,
    #         'start': self.start,
    #         'goal': self.goal
    #     }