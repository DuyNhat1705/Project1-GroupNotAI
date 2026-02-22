import os
import numpy as np
from src.problems.base_problem import BaseProblem


class GraphColoring(BaseProblem):
    def __init__(self, filename="graph.txt", name="Graph Coloring"):
        self.num_nodes = 0
        self.num_edges = 0
        self.edges = []
        self.node_coords = {}  # later use coordinates for Matplotlib

        self.filepath = os.path.join(BaseProblem.project_root, 'data', filename)
        self.load_from_file(self.filepath) # take test case from data

        #  circular layout for the nodes
        for i in range(self.num_nodes):
            angle = 2 * np.pi * i / self.num_nodes
            # Scale by 10
            self.node_coords[i] = (10 * np.cos(angle), 10 * np.sin(angle))

        # Dimension = num nodes
        # Bounds = [0, num_nodes - 1]
        super().__init__(name, dimension=self.num_nodes, bounds=[0, self.num_nodes - 1], cont_flag=False)

    def load_from_file(self, filename):
        with open(filename, "r") as file:
            lines = [line.strip() for line in file.readlines() if line.strip() and not line.strip().startswith('#')]

            if not lines:
                raise ValueError(f"The file {filename} is empty.")

            self.num_nodes, self.num_edges = map(int, lines[0].split())

            for line in lines[1:]:
                u, v = map(int, line.split())
                self.edges.append((u, v))

    def evaluate(self, solution):
        """
        solution: 1D array of integers representing the color assigned to each node.
        """
        colors = np.round(solution).astype(int)

        # count conflicts
        conf_cnt = sum(1 for u, v in self.edges if colors[u] == colors[v])

        # count colors used
        color_cnt = len(np.unique(colors))

        # calc fitness
        # penalty for conflicts: 1000
        return float(color_cnt + (1000 * conf_cnt))