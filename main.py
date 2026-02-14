import argparse
import sys
import os

# Import Factory
from src.HandleCLI import get_problem, get_algorithm
from src.visualization.visualizer_factory import get_visualizer

def parse_param_string(param_list):
    """
    Chuyển list ['key=value', 'a=1'] thành dict {'key': 'value', 'a': 1}
    Tự động ép kiểu int/float.
    """
    params = {}
    if not param_list:
        return params

    for item in param_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        
        # Thử ép kiểu số
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass # Giữ nguyên string nếu không phải số (VD: path="...")
            
        params[key] = value
    return params

def main():
    parser = argparse.ArgumentParser(description="AI Search Algorithm Runner")

    # Các tham số bắt buộc
    parser.add_argument("--algo", type=str, required=True, help="Tên thuật toán (VD: ga, de, bfs, astar)")
    parser.add_argument("--problem", type=str, required=True, help="Tên bài toán (VD: sphere, maze, graph)")

    # Các tham số tùy chọn
    parser.add_argument("--dim", type=int, default=2, help="Số chiều (cho bài toán Continuous)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Tham số động cho thuật toán (VD: pop_size=100 F=0.8)
    parser.add_argument("--params", nargs='*', help="Các tham số thuật toán dạng key=value (VD: pop_size=50 F=0.8 CR=0.9)")

    args = parser.parse_args()

    # 1. Parse Params
    algo_params = parse_param_string(args.params)
    print(f"\n[INFO] Run Configuration:")
    print(f"  Algorithm: {args.algo}")
    print(f"  Problem  : {args.problem}")
    print(f"  Dimension: {args.dim}")
    print(f"  Params   : {algo_params}")

    try:
        # 2. Khởi tạo Problem
        # Truyền dimension và seed vào kwargs cho get_problem
        problem = get_problem(args.problem, dimension=args.dim, seed=args.seed)
        
        # 3. Khởi tạo Algorithm
        # Truyền algo_params vào
        algorithm = get_algorithm(args.algo, **algo_params)

        # 4. Chạy thuật toán
        print(f"\n[INFO] Solving...")
        
        # Gọi hàm solve (Lưu ý: các thuật toán nature của bạn trả về tuple (sol, fit, log), 
        # còn classical trả về dict kết quả. Cần xử lý thống nhất ở đây hoặc trong visualizer)
        
        result = None
        logger = None
        
        # Kiểm tra loại thuật toán để gọi hàm solve phù hợp hoặc xử lý kết quả trả về
        # Dựa trên code bạn cung cấp, BaseAlgorithm trả về khác nhau một chút giữa classical/nature
        # Nhưng visualizer_factory cần một cấu trúc chung.
        
        raw_output = algorithm.solve(problem, seed=args.seed)
        
        # Chuẩn hóa dữ liệu để gửi cho Visualizer
        viz_data = {
            "problem": problem,
            "algorithm": algorithm.name,
            "result": {}
        }

        if isinstance(raw_output, tuple) and len(raw_output) == 3:
            # Nature Algorithms (GA, DE, HC...) trả về (best_sol, best_fit, logger)
            best_sol, best_fit, logger = raw_output
            print(f"  -> Done. Best Fitness: {best_fit}")
            
            viz_data["result"]["logger"] = logger
            # Nếu cần vẽ biểu đồ convergence, visualizer mới cần best_hist/avg_hist
            # Bạn có thể trích xuất từ logger nếu logger lưu
            
        elif isinstance(raw_output, dict):
            # Classical Algorithms (BFS, A*...) trả về dict {"result": ..., "time": ...}
            print(f"  -> Done. Cost: {raw_output['result']['cost']}")
            viz_data["result"] = raw_output["result"]
        else:
            # Trường hợp Hill Climbing trả về (cur, cur_fit, logger)
            # Code HC của bạn trả về 3 biến, khớp với Nature
            print(f"  -> Done. Result: {raw_output[1]}")
            viz_data["result"]["logger"] = raw_output[2]

        # 5. Visualization
        print(f"\n[INFO] Visualizing...")
        visualizer = get_visualizer(viz_data)
        
        if visualizer:
            visualizer.animate()
            # Nếu là bài toán Continuous, có thể gọi thêm analyze_performance nếu có data
            if hasattr(visualizer, 'analyze_performance'):
                # Lưu ý: CLI chạy 1 lần nên không có Grid Search Data để vẽ heatmap
                # Nhưng vẫn vẽ được Convergence Curve nếu logger có lưu history fitness
                pass
        else:
            print("  -> No visualizer available for this combination.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()