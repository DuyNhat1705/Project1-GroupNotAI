import os
import numpy as np
import random
from src.problems.base_problem import BaseProblem

class ShortestPathOnGraph(BaseProblem):
    def __init__(self, directed=True, seed=None):
        """
        weighted adjacency list (dict)
            {<vertice> :  {<vertice>: <weight>}
        filename: input txt file
        seed: random seed for position
        """
        super().__init__("ShortestPathOnGraph", dimension=1)
        self.adj_list = {}
        self.directed = directed
        self.start = None
        self.goal = None
        self.seed = seed
        self.node_coords = {}  # { 'NodeID': (x, y) }

        self.data_path = os.path.join(BaseProblem.project_root, 'data', 'weighted_graph_ex1.txt')
        self.load_from_file(self.data_path)
        
        self.generate_coords(seed) #assign node position for visualization
        self.dimension = self.__len__()


    def __len__(self):
        return len(self.adj_list)

    def generate_coords(self, seed):
        """
        Assigns random (x, y) coordinate to every node (position on visualizer)
        Seed ensures the graph looks the same every run
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Get all unique nodes
        nodes = list(self.adj_list.keys())

        # Assign coordinates
        for node in nodes:
            # Generate random X, Y between 0 and 100
            self.node_coords[node] = (random.uniform(0, 100), random.uniform(0, 100))


    def add_edge(self, u, v, weight):
        """
        Args:
            u, v (string): nodes
            weight(float)
        """

        if u not in self.adj_list:
            self.adj_list[u] = {}
        if v not in self.adj_list:
            self.adj_list[v] = {}

        self.adj_list[u][v] = float(weight)

        # If undirected, add edge v -> u
        if not self.directed:
            self.adj_list[v][u] = float(weight)

    def load_from_file(self, filename):
        """
        Reads txt file
        Format: "SrcNode DestNode Weight"
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        print(f"Loading graph from {filename}...")

        with open(filename, 'r') as file_input:
            for line_num, line in enumerate(file_input, 1):  # line number for error catch
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                token = line.split()  # split line to tokens

                # skip lines with false format
                if len(token) != 3:
                    continue


                u, v, w = token  # assign node and weight

                try:
                    if token[0] == "$":  # special line to config start & end node
                        self.start = token[1]
                        self.goal = token[2]
                    else:
                        self.add_edge(u, v, float(w))

                except ValueError:
                    print(f"Error on line {line_num}: Invalid weight")

    def get_neighbors(self, node):
        """Returns dict {neighbor: weight}"""
        return self.adj_list.get(node, {})

    def get_edge_cost(self, u, v):
        """Returns weight of the edge u->v"""
        return self.adj_list.get(u, {}).get(v, 0)


    def evaluate(self, solution):
        """
        Args:
            solution (list): node id
        Returns:
            float: Total cost of solution
        """
        cost = 0
        if not solution or len(solution) < 2: # extreme case
            return 0

        for i in range(len(solution) - 1):
            u, v = solution[i], solution[i + 1]
            cost += self.get_edge_cost(u, v)

        return cost



