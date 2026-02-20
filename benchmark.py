import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import argparse
import threading
import _thread
from src.algorithms.classical.dfs import DFS
from src.algorithms.classical.bfs import BFS
from src.algorithms.classical.a_star import A_Star
from src.problems.problems_factory import get_problem
from src.algorithms.biology.artificial_bee import ArtificialBee
from src.algorithms.physics.simulated_annealing import SimulatedAnnealing
from src.algorithms.classical.hill_climbing import HillClimbing


# ---  PROBLEM-TO-ALGORITHM MAP ---
# Defines which algorithms can attempt which problems
COMPATIBILITY = {
    "continuous": ["ABC", "SA", "HC"],
    "tsp": ["SA", "HC"],
    "maze": ["A*", "BFS", "DFS"],
    "knapsack": ["ABC", "SA", "BFS"]
}

ALGO_CLASSES = {
    "ABC": ArtificialBee, "SA": SimulatedAnnealing, "HC": HillClimbing,
    "A*": A_Star, "BFS": BFS, "DFS": DFS
}

# Define known optimums for Error calculations (Continuous)
KNOWN_OPTIMUMS = {
    "sphere": 0.0, "griewank": 0.0, "rosenbrock": 0.0
}

def timeout_handler(flag):
    """Sets the timeout flag and forces a KeyboardInterrupt in the main thread."""
    flag[0] = True
    _thread.interrupt_main()

def determine_category(prob_name):
    prob = prob_name.lower()
    if prob in ["sphere", "rosenbrock", "griewank", "ackley"]: return "continuous"
    if prob in ["tsp", "travelsalesman"]: return "tsp"
    if prob in ["maze", "shortestpathonmaze"]: return "maze"
    if prob in ["knapsack"]: return "knapsack"
    return "continuous"  # default fallback


def extract_convergence(logger):
    """Extracts the convergence curve, ensuring 1D numeric data."""
    if not logger: return []

    # Check for explicit cost history
    if "best_cost" in logger.history:
        return [float(val) for val in logger.history["best_cost"]]

    # Check for explored history (Used by SA, HC) -> Format: [(solution, fitness), ...]
    elif "explored" in logger.history:
        curve = []
        for step in logger.history["explored"]:
            # Make sure the step is tuple and has second val
            if isinstance(step, (tuple, list)) and len(step) >= 2:
                val = step[1]
                # Only append actual number (ignore edge)
                if isinstance(val, (int, float, np.floating, np.integer)):
                    curve.append(float(val))
        return curve

    return []

def run_benchmark(prob_name, runs=30):
    category = determine_category(prob_name)
    compatible_algos = COMPATIBILITY.get(category, [])

    print(f"\nBenchmarking {prob_name.upper()} against: {compatible_algos}")

    stats = {}

    for algo_name in compatible_algos:
        if algo_name not in ALGO_CLASSES: continue

        print(f" -> Running {algo_name} ({runs} runs)...")
        stats[algo_name] = {'fitness': [], 'time': [], 'memory': [], 'convergence': [], 'nodes': []}

        for i in range(runs):
            problem = get_problem(prob_name, dimension=10, seed=i)
            algo = ALGO_CLASSES[algo_name]()

            # --- TIMEOUT SETUP ---
            timeout_flag = [False]
            # Create a 10-second timer that will call timeout_handler
            timer = threading.Timer(10.0, timeout_handler, args=[timeout_flag])

            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                timer.start()  # Start the countdown!

                raw_output = algo.solve(problem, seed=i)

                timer.cancel()  # If it finishes in time, stop the bomb!

                end_time = time.perf_counter()
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Log the results normally
                res = raw_output["result"]
                final_fit = res.get("best_fitness", res.get("cost", 0))

                stats[algo_name]['fitness'].append(final_fit)
                stats[algo_name]['time'].append((end_time - start_time) * 1000)
                stats[algo_name]['memory'].append(peak_mem / 1024)
                stats[algo_name]['convergence'].append(extract_convergence(res.get("logger")))

                if "nodes_expanded" in res:
                    stats[algo_name]['nodes'].append(res["nodes_expanded"])

                print(f"  Run {i + 1:02d}/{runs} | Fit: {final_fit:.4f} | Time: {(end_time - start_time) * 1000:.1f}ms",
                      end="\r")

            # --- CATCH THE TIMEOUT ---
            except KeyboardInterrupt:
                timer.cancel()
                tracemalloc.stop()

                if timeout_flag[0]:
                    print(f"\n  [TIMEOUT] Run {i + 1} exceeded 10s! BFS/DFS has combinatorial explosion.")
                    break  # Break the loop for this algorithm
                else:
                    raise

            # --- CATCH OTHER ERRORS ---
            except Exception as e:
                timer.cancel()
                tracemalloc.stop()
                print(f"\n  [ERROR] Run {i + 1} failed: {e}")
                break

    generate_svg_reports(stats, prob_name)


def generate_svg_reports(stats, prob_name):
    os.makedirs("output", exist_ok=True)

    # Filter out any algorithms that crashed or returned empty data
    valid_labels = [algo for algo in stats.keys() if len(stats[algo]['fitness']) > 0]

    if not valid_labels:
        print(f"[WARNING] No algorithms successfully solved {prob_name}. Skipping charts.")
        return

    def save_plot(suffix):
        plt.tight_layout()
        filepath = f"output/{prob_name}_{suffix}.svg"
        plt.savefig(filepath, format="svg")
        plt.close()
        print(f"Saved: {filepath}")

    # ==========================================
    # CONVERGENCE (Line Plot)
    # ==========================================
    try:
        plt.figure(figsize=(8, 6))
        for algo in valid_labels:
            curves = [c for c in stats[algo]['convergence'] if c]
            if curves:
                # Ensure the curve data is numeric before plotting
                if not isinstance(curves[0][0], (int, float, np.floating, np.integer)):
                    continue

                max_len = max(len(c) for c in curves)
                padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]
                plt.plot(np.mean(padded, axis=0), label=algo, linewidth=2)

        plt.title(f"Convergence - {prob_name.upper()}")
        plt.xlabel("Iterations")
        plt.ylabel("Score")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        save_plot("convergence")
    except Exception as e:
        print(f" [!] Could not generate Convergence chart: {e}")
        plt.close()

    # ==========================================
    # ROBUSTNESS (Boxplot)
    # ==========================================
    try:
        plt.figure(figsize=(8, 6))
        fitness_data = [stats[algo]['fitness'] for algo in valid_labels]

        # THE FIX: Version-agnostic boxplot. Draw the boxes first, label them after!
        plt.boxplot(fitness_data, patch_artist=True)
        plt.xticks(ticks=range(1, len(valid_labels) + 1), labels=valid_labels)

        plt.title(f"Robustness (Score Distribution) - {prob_name.upper()}")
        plt.ylabel("Final Score Found")
        plt.grid(True, linestyle='--', alpha=0.6)
        save_plot("robustness")
    except Exception as e:
        print(f" [!] Could not generate Robustness chart: {e}")
        plt.close()

    # ==========================================
    # TIME & SPACE COMPLEXITY (Bar Chart)
    # ==========================================
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Extract both metrics
        avg_times = [np.mean(stats[algo]['time']) for algo in valid_labels]
        avg_memory = [np.mean(stats[algo]['memory']) for algo in valid_labels]

        x = np.arange(len(valid_labels))
        width = 0.35  # Width of the bars

        # --- PRIMARY Y-AXIS (TIME) ---
        bars1 = ax1.bar(x - width / 2, avg_times, width, label='Time (ms)', color='skyblue')
        ax1.set_ylabel('Average Time (ms)', color='tab:blue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Add numbers on top of Time bars
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.1f}", ha='center', va='bottom', color='tab:blue', fontsize=9)

        # --- SECONDARY Y-AXIS (SPACE/MEMORY) ---
        ax2 = ax1.twinx()  # This creates a second Y-axis sharing the same X-axis!
        bars2 = ax2.bar(x + width / 2, avg_memory, width, label='Memory (KB)', color='lightcoral')
        ax2.set_ylabel('Average Peak Memory (KB)', color='tab:red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Add numbers on top of Memory bars
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{bar.get_height():.1f}", ha='center', va='bottom', color='tab:red', fontsize=9)

        # Add X-axis labels and Title
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_labels)
        plt.title(f"Empirical Time & Space Complexity - {prob_name.upper()}")

        # Combine the legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        save_plot("complexity")
    except Exception as e:
        print(f" [!] Could not generate Complexity chart: {e}")
        plt.close()

    # ==========================================
    # ERROR (If Optimum is known)
    # ==========================================
    if prob_name.lower() in KNOWN_OPTIMUMS:
        try:
            plt.figure(figsize=(8, 6))
            optimum = KNOWN_OPTIMUMS[prob_name.lower()]
            errors = [np.mean(np.abs(np.array(stats[algo]['fitness']) - optimum)) for algo in valid_labels]
            plt.bar(valid_labels, errors, color='salmon')
            plt.title(f"Absolute Error from Optimum ({optimum}) - {prob_name.upper()}")
            plt.ylabel("Mean Absolute Error")
            save_plot("error_deviation")
        except Exception as e:
            print(f" [!] Could not generate Error chart: {e}")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, help="Problem to benchmark (e.g., sphere, maze, tsp)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for stochastic algorithms")
    args = parser.parse_args()

    run_benchmark(args.problem, args.runs)