import os
import math
import numpy as np
import matplotlib.pyplot as plt

from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm

SAVE_DIR = os.path.join(os.getcwd(), "output", "sensitivity_analysis")
os.makedirs(SAVE_DIR, exist_ok=True)

# Lược bỏ các thuật toán duyệt chính xác hoặc parameter-less (TLBO, BFS, DFS, A*)
COMPATIBILITY = {
    "continuous": ["abc", "sa", "hc", "pso", "cs", "fa", "aco", "ga", "de"],
    "tsp": ["sa", "pso", "fa", "aco", "ga", "hc"], 
    "maze": ["ga"], 
    "knapsack": ["abc", "cs"], 
    "graphcoloring": [] 
}

def get_algo_param_grid(algo_name, problem):
    """
    Trả về số lượng tham số (1 hoặc 2) cùng với tên và mảng giá trị tương ứng.
    - Trả về (2, p1_name, p1_vals, p2_name, p2_vals) -> Vẽ Heatmap
    - Trả về (1, p1_name, p1_vals, None, None)      -> Vẽ Line Plot
    - Trả về None                                   -> Bỏ qua
    """
    is_cont = getattr(problem, 'cont_flag', False)
    
    # Tính toán search range cho Logspace (nếu là bài toán liên tục)
    search_range = (problem.max_range - problem.min_range) if is_cont else 1.0
    # Scale nhiễu không gian từ 0.1% đến 20% search range
    logspace_fluctuations = np.logspace(-3, np.log10(0.2), 5) * search_range

    if algo_name == "aco":
        return 2, "alpha", np.linspace(0.5, 3.0, 5), "beta", np.linspace(1.0, 5.0, 5)
        
    elif algo_name == "ga":
        if is_cont:
            return 2, "CR", np.linspace(0.1, 0.9, 5), "F", logspace_fluctuations
        else:
            return 1, "CR", np.linspace(0.1, 0.9, 5), None, None
            
    elif algo_name == "de":
        return 2, "F", np.linspace(0.1, 1.0, 5), "CR", np.linspace(0.1, 0.9, 5)
        
    elif algo_name == "pso":
        return 2, "w_max", np.linspace(0.4, 0.9, 5), "c1", np.linspace(1.0, 3.0, 5)
        
    elif algo_name == "sa":
        return 2, "temperature", np.logspace(1, 3, 5), "decay", np.linspace(0.8, 0.99, 5)
        
    elif algo_name == "fa":
        return 2, "alpha", np.linspace(0.1, 1.0, 5), "gamma", np.linspace(0.1, 5.0, 5)
        
    elif algo_name == "cs":
        return 2, "pa", np.linspace(0.1, 0.5, 5), "beta", np.linspace(1.0, 2.0, 5)
        
    elif algo_name == "abc":
        return 1, "limit", np.linspace(5, 50, 5, dtype=int), None, None
        
    elif algo_name == "hc":
        if is_cont:
            return 1, "step", logspace_fluctuations, None, None
        return None

    return None

def set_axis_scale_and_label(ax, axis, name, vals):
    """Hàm phụ trợ để set label và log scale công bằng cho các giá trị dao động"""
    if np.max(vals) < 1.0 and name in ['F', 'step']:
        getattr(ax, f"set_{axis}scale")('log')
        getattr(ax, f"set_{axis}label")(f"{name} (Log Scale)", fontweight='bold')
    elif name == 'temperature':
        getattr(ax, f"set_{axis}scale")('log')
        getattr(ax, f"set_{axis}label")(f"{name} (Log Scale)", fontweight='bold')
    else:
        getattr(ax, f"set_{axis}label")(name, fontweight='bold')

def get_fitness_safely(output_dict):
    """Hàm trích xuất fitness an toàn, chống lỗi KeyError giữa Cost và Fitness"""
    res = output_dict["result"]
    # Thử lấy best_fitness, nếu không có thì lấy cost
    fit = res.get("best_fitness", res.get("cost", None))
    # Nếu vẫn None (hiếm), thử móc từ bên trong logger (Logger class luôn gán best_fitness)
    if fit is None and "logger" in res:
        fit = res["logger"].meta.get("best_fitness")
    return fit

def run_sensitivity_analysis(problem_name, problem_type):
    print(f"\n{'='*50}\nStarting Parameter Sensitivity Analysis\nProblem: {problem_name.upper()}\n{'='*50}")
    problem = get_problem(problem_name)
    
    compatible_algos = COMPATIBILITY.get(problem_type, [])
    results = [] 
    
    # 1. THỰC THI GRID SEARCH
    for algo_name in compatible_algos:
        grid_params = get_algo_param_grid(algo_name, problem)
        if not grid_params:
            continue
            
        dim, p1_name, p1_vals, p2_name, p2_vals = grid_params
        
        print(f" -> Testing {algo_name.upper()} ({'1D Line Plot' if dim == 1 else '2D Heatmap'})...")
        
        if dim == 2:
            Z = np.zeros((len(p1_vals), len(p2_vals)))
            for i, v1 in enumerate(p1_vals):
                for j, v2 in enumerate(p2_vals):
                    algo = get_algorithm(algo_name, **{p1_name: v1, p2_name: v2})
                    output = algo.solve(problem, seed=42)
                    Z[i, j] = get_fitness_safely(output) # SỬ DỤNG HÀM LẤY AN TOÀN
            results.append((algo_name.upper(), dim, p1_name, p1_vals, p2_name, p2_vals, Z))
            
        elif dim == 1:
            Z = np.zeros(len(p1_vals))
            for i, v1 in enumerate(p1_vals):
                algo = get_algorithm(algo_name, **{p1_name: v1})
                output = algo.solve(problem, seed=42)
                Z[i] = get_fitness_safely(output) # SỬ DỤNG HÀM LẤY AN TOÀN
            results.append((algo_name.upper(), dim, p1_name, p1_vals, None, None, Z))

    # 2. VẼ BIỂU ĐỒ VÀO 1 FIGURE
    N = len(results)
    if N == 0:
        print(f"No tunable algorithms found for {problem_name}.")
        return

    cols = min(3, N)
    rows = math.ceil(N / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    fig.suptitle(f"Parameter Sensitivity Analysis - {problem.name}", fontsize=18, fontweight='bold', y=1.02)
    
    axes_flat = [axes] if N == 1 else axes.flatten()
        
    for idx, ax in enumerate(axes_flat):
        if idx < N:
            algo_name, dim, p1_name, p1_vals, p2_name, p2_vals, Z = results[idx]
            
            if dim == 2:
                # VẼ HEATMAP
                c = ax.contourf(p1_vals, p2_vals, Z.T, levels=20, cmap='viridis_r')
                fig.colorbar(c, ax=ax, label="Best Fitness")
                set_axis_scale_and_label(ax, 'x', p1_name, p1_vals)
                set_axis_scale_and_label(ax, 'y', p2_name, p2_vals)
                
            elif dim == 1:
                # VẼ LINE PLOT
                ax.plot(p1_vals, Z, marker='o', color='royalblue', linewidth=2.5, markersize=8)
                ax.set_ylabel("Best Fitness", fontweight='bold')
                set_axis_scale_and_label(ax, 'x', p1_name, p1_vals)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            ax.set_title(f"{algo_name}", fontsize=14, fontweight='bold')
        else:
            # Ẩn grid phụ dư thừa
            ax.axis('off')
            
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{problem_name}_Sensitivity_Analysis.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved all-in-one figure to: {save_path}")

def main():
    test_cases = [
        ("sphere", "continuous"),
        ("tsp1", "tsp"),
        ("knapsack1", "knapsack")
    ]
    
    for prob_name, prob_type in test_cases:
        run_sensitivity_analysis(prob_name, prob_type)

if __name__ == "__main__":
    main()