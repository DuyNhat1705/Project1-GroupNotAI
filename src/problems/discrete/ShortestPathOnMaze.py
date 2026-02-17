from src.problems.base_problem import BaseProblem
import numpy as np
import os


class ShortestPathOnMaze(BaseProblem):
    def __init__(self):
        super().__init__(name="ShortestPathOnMaze", dimension=2)
        self.maze = None
        self.start = None
        self.goal = None

        self.data_path = os.path.join(BaseProblem.project_root, 'data', 'maze.txt')
        self.load_from_file(self.data_path)

    def load_from_file(self, filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines
                     if line.strip() and not line.strip().startswith('#')]

            # Read raw X, Y (Col, Row) coordinates
            start_raw = tuple(map(int, lines[0].split()))
            goal_raw = tuple(map(int, lines[1].split()))

            # --- FLIP TO (ROW, COL) ---
            # NumPy expects (y, x)
            self.start = (start_raw[1], start_raw[0])
            self.goal = (goal_raw[1], goal_raw[0])

            # Load maze matrix
            matrix_data = [list(map(int, line.split())) for line in lines[2:]]
            self.maze = np.array(matrix_data)

    def get_neighbors(self, current):
        """
        Required by BFS/DFS to find valid moves.
        """
        r, c = current
        neighbors = []
        # Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Check maze boundaries
            if 0 <= nr < self.maze.shape[0] and 0 <= nc < self.maze.shape[1]:
                # Check if it's a valid path (0 is way)
                if self.maze[nr, nc] == 0:
                    neighbors.append((nr, nc))

        return neighbors

    def evaluate(self, path):
        """
        Required by BFS/DFS to score the final path.
        """
        if not path:
            return float('inf')
        # Cost = the number of steps taken
        return len(path) - 1