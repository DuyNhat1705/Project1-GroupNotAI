from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """
    Abstract Base Class for all search algorithms.
    """

    def __init__(self, name, params):
        """
        Args:
            name (str): Algorithm name 
            params (dict): Dictionary of hyperparameters
        """
        self.name = name
        self.params = params

    @abstractmethod
    def solve(self, problem, seed=None):
        """
        Execute the search strategy on the problem.
        
        Args:
            problem: problem to solve.
            seed: Random seed for reproducibility (for nature-inspired algos)
            
        Returns:
            tuple: (best_solution, best_fitness, history_log)
        """
        pass
    def getAlgorithmName(self):
        return self.name
    # def runAlgorithm(self,problem,seed=None):
    #     start = time.perf_counter()
    #     result = self.solve(problem, seed)
    #     end = time.perf_counter()

    #     return {
    #         "result": result,
    #         "time(ms)": round((end - start)*1000,3)
    #     }
    # @abstractmethod
    # def reconstruct_path(self, solution):
    #     pass
    #
    # @abstractmethod
    # def evaluate(self, solution):
    #     pass
