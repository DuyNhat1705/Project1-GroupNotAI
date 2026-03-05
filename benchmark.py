import os
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import threading
import _thread
import itertools
from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm
from src.HandleCLI import parse_param_string

# ==========================================
# CONFIGURATIONS
# ==========================================
COMPATIBILITY = {
    "continuous": ["abc", "sa", "hc", "pso", "cs", "fa", "aco", "ga", "de", "tlbo"],
    "tsp": ["sa", "hc", "pso", "fa", "aco", "ga", "astar"],
    "maze": ["astar", "bfs", "dfs", "ga"],
    "knapsack": ["abc", "bfs", "cs", "tlbo"],
    "graphcoloring": ["bfs", "dfs", "sa", "hc"]
} #match problem with available algorithms

def timeout_handler(flag): #time limit = 30s (set below)
    """Sets the timeout flag and forces a KeyboardInterrupt in the main thread"""
    flag[0] = True
    _thread.interrupt_main()

def route_solver(prob_name): # route input testcase to problem
    prob = prob_name.lower()
    if prob in ["tsp1", "tsp2"]: return "tsp"
    if prob in ["maze1", "maze2"]: return "maze"
    if prob in ["knapsack1", "knapsack2"]: return "knapsack"
    if prob in ["graph1"]: return "graph"
    if prob in ["coloring1", "coloring2"]: return "graphcoloring"
    return "continuous" # fall back

def extract_convergence(logger):
    """Extracts convergence curve, ensure 1D numeric data."""
    if not logger: return []

    if "best_fitness" in logger.history:
        return [float(val) for val in logger.history["best_fitness"]]
    return []


def run_benchmark(prob_name, runs=30, dim=10, algo_params=None):
    if algo_params is None:
        algo_params = {}

    category = route_solver(prob_name)
    compatible_algos = COMPATIBILITY.get(category, [])

    print(f"\nBenchmarking {prob_name.upper()} (Dim: {dim}) against: {compatible_algos}")
    if algo_params:
        print(f"Algorithm Params: {algo_params}")

    stats = {}
    optimum = None

    for algo_name in compatible_algos:
        print(f" -> Running {algo_name.upper()} ({runs} runs)...")
        stats[algo_name] = {'fitness': [], 'time': [], 'memory': [], 'convergence': [], 'nodes': []}

        for i in range(runs):
            problem = get_problem(prob_name, dimension=dim, seed=i)

            if optimum is None and hasattr(problem, "global_min"):
                optimum = problem.global_min # extract global min from continuous problem

            algo = get_algorithm(algo_name, **algo_params)

            timeout_flag = [False]
            timer = threading.Timer(300.0, timeout_handler, args=[timeout_flag]) #time out = 90s

            tracemalloc.start()
            start_time = time.perf_counter()

            try:
                timer.start()
                raw_output = algo.solve(problem, seed=i)
                timer.cancel()

                end_time = time.perf_counter()
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

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

            except KeyboardInterrupt:
                timer.cancel()
                tracemalloc.stop()
                if timeout_flag[0]:
                    print(f"\n  [TIMEOUT] Run {i + 1} exceeded 90s!")
                    break
                else:
                    raise
            except Exception as e:
                timer.cancel()
                tracemalloc.stop()
                print(f"\n  [ERROR] Run {i + 1} failed: {e}")
                break

    generate_reports(stats, prob_name, optimum)


def generate_reports(stats, prob_name, optimum=None):
    # Create a subfolder for each problem
    out_dir = f"output/{prob_name.lower()}"
    os.makedirs(out_dir, exist_ok=True)

    valid_labels = [algo for algo in stats.keys() if len(stats[algo]['fitness']) > 0]

    if not valid_labels:
        print(f"\n[WARNING] No algorithms successfully solved {prob_name}. Skipping charts.")
        return

    # ==========================================
    # 1. JSON DATA EXPORT
    # ==========================================
    report_data = {}
    for algo in valid_labels:
        fit_arr = np.array(stats[algo]['fitness'])
        time_arr = np.array(stats[algo]['time'])
        mem_arr = np.array(stats[algo]['memory'])

        report_data[algo] = {
            "fitness": {
                "best": float(np.min(fit_arr)),
                "worst": float(np.max(fit_arr)),
                "mean": float(np.mean(fit_arr)),
                "median": float(np.median(fit_arr)),
                "std": float(np.std(fit_arr))
            },
            "time_ms": {
                "mean": float(np.mean(time_arr)),
                "std": float(np.std(time_arr))
            },
            "memory_kb": {
                "mean": float(np.mean(mem_arr)),
                "std": float(np.std(mem_arr))
            }
        }

    # Save JSON into the new subfolder
    json_path = f"{out_dir}/{prob_name}_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\n[+] Saved JSON Statistics: {json_path}")

    # ==========================================
    # 2. PDF CHARTS
    # ==========================================
    def save_plot(suffix):
        filepath = f"{out_dir}/{prob_name}_{suffix}.pdf"
        # bbox_inches='tight' guarantees external legends aren't cropped out
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
        plt.close()
        print(f"[+] Saved PDF Chart: {filepath}")

    # --- CONVERGENCE (Line Plot) ---
    try:
        plt.figure(figsize=(8, 6))
        line_styles = ['-', '--', '-.'] # roulette many styles of line so that one will not cover others

        for idx, algo in enumerate(valid_labels):
            curves = [c for c in stats[algo]['convergence'] if c]
            if curves:
                if not isinstance(curves[0][0], (int, float, np.floating, np.integer)):
                    continue

                # Safety padding: just in case it stops early
                max_len = max(len(c) for c in curves)
                padded = [c + [c[-1]] * (max_len - len(c)) for c in curves]

                padded_arr = np.array(padded, dtype=float)
                padded_arr[np.isinf(padded_arr)] = np.nan

                mean_curve = np.nanmean(padded_arr, axis=0)
                std_curve = np.nanstd(padded_arr, axis=0)

                # X-axis starts at 1
                x_axis = np.arange(1, len(mean_curve) + 1)

                # Pick a style based on the algorithm's index
                style = line_styles[idx % len(line_styles)]

                # Plot with linestyle and alpha=0.8
                line, = plt.plot(x_axis, mean_curve, label=algo.upper(), linewidth=2, linestyle=style, alpha=0.8)
                plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve,
                                 color=line.get_color(), alpha=0.15)

        if optimum is not None:
            plt.axhline(y=optimum, color='black', linestyle=':', linewidth=1.5, label=f"True Optimum ({optimum})")

        if "knapsack" in prob_name or "tsp" in prob_name  or "maze" in prob_name:
            plt.yscale("linear")
        else:
            plt.yscale("symlog", linthresh=1e-3)

        plt.title(f"Convergence - {prob_name.upper()}")
        plt.xlabel("Iterations")
        plt.ylabel("Objective value")

        if plt.gca().get_legend_handles_labels()[0]:
            # Moved legend completely outside the chart area
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.grid(True, linestyle='--', alpha=0.6)
        save_plot("convergence")
    except Exception as e:
        print(f" [!] Could not generate Convergence chart: {e}")
        plt.close()

    # --- ROBUSTNESS (Boxplot) ---
    try:
        plt.figure(figsize=(8, 6))
        fitness_data = [stats[algo]['fitness'] for algo in valid_labels]
        plt.boxplot(fitness_data, patch_artist=True)
        plt.xticks(ticks=range(1, len(valid_labels) + 1), labels=[l.upper() for l in valid_labels])

        plt.title(f"Robustness - {prob_name.upper()}")
        plt.ylabel("Best Fitness")
        plt.grid(True, linestyle='--', alpha=0.6)
        save_plot("robustness")
    except Exception as e:
        print(f" [!] Could not generate Robustness chart: {e}")
        plt.close()

    # --- TIME & SPACE COMPLEXITY (Bar Chart) ---
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        avg_times = [np.mean(stats[algo]['time']) for algo in valid_labels]
        std_times = [np.std(stats[algo]['time']) for algo in valid_labels]  #Calc Time Std Dev

        avg_memory = [np.mean(stats[algo]['memory']) for algo in valid_labels]
        std_memory = [np.std(stats[algo]['memory']) for algo in valid_labels]  # Calc Memory Std Dev

        x = np.arange(len(valid_labels))
        width = 0.35

        # yerr and capsize to show the variance safely
        bars1 = ax1.bar(x - width / 2, avg_times, width, yerr=std_times, capsize=5, label='Time (ms)', color='skyblue')

        for i, bar in enumerate(bars1):
            yval = bar.get_height() + std_times[i]
            ax1.text(bar.get_x() + bar.get_width() / 2, yval, f"{bar.get_height():.1f}",
                     ha='center', va='bottom', color='tab:blue', fontsize=9)

        ax1.set_ylabel('Average Time (ms)', color='tab:blue', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, avg_memory, width, yerr=std_memory, capsize=5, label='Memory (KB)',
                        color='lightcoral')

        for i, bar in enumerate(bars2):
            yval = bar.get_height() + std_memory[i]
            ax2.text(bar.get_x() + bar.get_width() / 2, yval, f"{bar.get_height():.1f}",
                     ha='center', va='bottom', color='tab:red', fontsize=9)

        ax2.set_ylabel('Average Peak Memory (KB)', color='tab:red', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        ax1.set_xticks(x)
        ax1.set_xticklabels([l.upper() for l in valid_labels])

        plt.title(f"Time & Space Complexity - {prob_name.upper()}")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.10, 1), loc='upper left')

        save_plot("complexity")
    except Exception as e:
        print(f" [!] Could not generate Complexity chart: {e}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, help="Problem to benchmark (e.g., sphere, maze, tsp)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for stochastic algorithms")
    parser.add_argument("--dim", type=int, default=10, help="Dimension for continuous problems")
    parser.add_argument("--params", nargs='*', help="Algorithm parameters like iteration=500 limit=20")

    args = parser.parse_args()
    algo_params = parse_param_string(args.params)

    print("\n========================================")
    print("      AI SEARCH ALGORITHM BENCHMARK     ")
    print("========================================")
    run_benchmark(args.problem, runs=args.runs, dim=args.dim, algo_params=algo_params)