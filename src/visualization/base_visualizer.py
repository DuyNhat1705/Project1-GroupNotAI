from abc import ABC, abstractmethod
import os

class BaseVisualizer(ABC):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))

    def __init__(self, problem, history, path, title):
        self.problem = problem
        self.history = history     
        self.path = path            
        self.paths = []
        self.title = title
        self.save_path = os.path.join(
            BaseVisualizer.project_root,
            "output",
            title + ".mp4"
        )

    @abstractmethod
    def animate(self):
        pass