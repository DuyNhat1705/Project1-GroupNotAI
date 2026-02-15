
# Import Factory
from src.HandleCLI import handleCLI
from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm
from src.visualization.visualizer_factory import get_visualizer

def main():
    #1. Xử lý CLI
    args, algo_params = handleCLI()
    try:
        # 2. Khởi tạo Problem
        problem = get_problem(args.problem, dimension=args.dim, seed=args.seed)
        
        # 3. Khởi tạo Algorithm
        algorithm = get_algorithm(args.algo, **algo_params)

        # 4. Chạy thuật toán
        print(f"\n[INFO] Solving...")
        
        raw_output = algorithm.solve(problem, seed=args.seed)
        
        # Chuẩn hóa dữ liệu để gửi cho Visualizer
        viz_data = {
            "problem": problem,
            "algorithm": algorithm.name,
            "result": raw_output["result"]
        }
        visualizer = get_visualizer(viz_data)
        # In thời gian thực thi ngay sau khi solve xong
        print(f"Time(ms): {raw_output.get('time(ms)', 'N/A')}")
        
        # 5. Visualization
        print(f"\n[INFO] Visualizing...")
        
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