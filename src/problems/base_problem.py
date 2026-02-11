from abc import ABC, abstractmethod
import numpy as np
import os
class BaseProblem(ABC):
    """
    Inherit ABC
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir)) # project root directory
    
    def __init__(self, name, dimension, bounds=None, is_min=True):
        """
        Args:
            name (str): problem name
            dimension (int): Number of dimensions 
            bounds (list of tuple): Boundaries for continuous problems.
            is_min (bool): minimial cost option
        """
        self.name = name
        self.dimension = dimension # complexity of problem space
        self.bounds = bounds # bound option for objective function
        self.is_min = is_min # the goal is the lowest point if True
    def getName(self):
        return self.name