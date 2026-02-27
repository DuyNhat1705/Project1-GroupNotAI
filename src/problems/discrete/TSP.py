import numpy as np
import math
import os
from src.problems.base_problem import BaseProblem

class TravelSalesmanProblem(BaseProblem):
    def __init__(self, context = "tsp1", name="TSP"):
        self.city_names = []
        self.coords = None  # coords for visualization
        self.dist_mat = None
        self.best_distance = None

        self.filepath = os.path.join(BaseProblem.project_root, 'data', f'{context}.txt')

        self.load_from_file(self.filepath)
        dim = len(self.dist_mat)
        self.coords = self.generate_circular_layout(dim)

        super().__init__(name=name, dimension=dim, bounds=None, cont_flag=False)

    def load_from_file(self, filepath):
        with open(filepath, "r") as f:
            raw_lines = [l.strip() for l in f.readlines() if l.strip()]

        names = []
        matrix_rows = []
        best_distance = None
        read_best = False

        for line in raw_lines:
            if line.startswith("#"):
                if "Best Distance" in line:
                    read_best = True
                continue

            if read_best:
                best_distance = float(line)
                read_best = False
                continue

            try:
                row = list(map(float, line.split()))
                if len(row) > 1:
                    matrix_rows.append(row)
            except ValueError:
                names.append(line)

        self.dist_mat = np.array(matrix_rows)
        self.dist_mat[np.isinf(self.dist_mat)] = 999999.0

        self.city_names = names
        self.best_distance = best_distance

        if len(self.city_names) != len(self.dist_mat):
            print(f"[WARNING] Mismatch: {len(self.city_names)} names vs {len(self.dist_mat)} matrix rows.")
            if len(self.city_names) == 0:
                self.city_names = [f"City_{i}" for i in range(len(self.dist_mat))]

    def generate_circular_layout(self, n, radius=100):
        """Generate (x, y) coordinates to arrange nodes in circle."""
        coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            coords.append([x, y])
        return np.array(coords)

    def evaluate(self, solution):
        """
        Calculate total distance of the path.
        """
        # Ensure solution is int idx
        path = np.array(solution, dtype=int)

        total_cost = 0
        dim = self.dimension

        for i in range(dim - 1):
            start_node = path[i]
            end_node = path[i + 1]
            total_cost += self.dist_mat[start_node][end_node]

        # cost of last edge of the circuit
        total_cost += self.dist_mat[path[-1]][path[0]] # from end back to start

        return total_cost