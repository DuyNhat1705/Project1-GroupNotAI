import time

class Logger:
    def __init__(self, algo_name, run_id=None):
        self.algo_name = algo_name
        self.meta = {
            "run_id": run_id, # assigned to distinguish multiple runs
            "start_time": time.time(), # mark start time to calc duration
            "runtime": 0
        }

        self.history = {} 

    def log(self, key, value):
        """
        Agrs:
            key (str):  what to log
            value(...): elements of corresponding 'key' list
        """
        if key not in self.history: # Auto-creates if not exist.
            self.history[key] = [] # new empty list at 'key'
        self.history[key].append(value) # Appends 'value' to the list at 'key'

    def finish(self, best_solution, best_fitness):
        """"
            Called when algo terminates
            log the runtime and performance metrics
        """
        self.meta["runtime"] = time.time() - self.meta["start_time"]
        self.meta["best_solution"] = best_solution # reconstruct the best answer
        self.meta["best_fitness"] = best_fitness # performance of the best solution