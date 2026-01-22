from abc import ABC, abstractmethod
import numpy as np

class BaseProblem(ABC):
    """
    Inherit ABC
    Abstract Class for optimization problems.
    """
    
    def __init__(self, name, dimension, bounds=None, is_minimization=True):
        """
        Args:
            name (str): problem name
            dimension (int): Number of dimensions 
            bounds (list of tuple): Boundaries for continuous problems.
            is_minimization (bool): minimial cost option
        """
        self.name = name
        self.dimension = dimension
        self.bounds = bounds
        self.is_minimization = is_minimization

    @abstractmethod
    def evaluate(self, solution):
        """
        Calculates the objective function value
        
        Args:
            solution (np.array): A single candidate solution.
            
        Returns:
            float: The fitness score.
        """
        pass

    @abstractmethod
    def generate_solution(self):
        """
        Generates valid solution
        
        Returns:
            np.array: solution vector.
        """
        pass

    def check_constraints(self, solution):
        """
        To be overriden if strict contains.
        """
        return True