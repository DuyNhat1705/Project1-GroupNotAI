# Import Factory

from src.HandleCLI import handleCLI, get_problem_type, check_compatibility
from src.problems.problems_factory import get_problem
from src.algorithms.algorithms_factory import get_algorithm
from src.visualization.visualizer_factory import get_visualizer

def main():
    #1. Xử lý CLI
    args, algo_params = handleCLI()

    # 2. Khởi tạo Problem
    problem = get_problem(args.problem, dimension=args.dim, seed=args.seed)

    # 3. Khởi tạo Algorithm
    algorithm = get_algorithm(args.algo, **algo_params)

    # 4. Kiểm tra tương thích
    problem_type = get_problem_type(problem)
    is_compatible, compatible_algos, compatible_problems = check_compatibility(
        algorithm.name, problem_type
    )

    if is_compatible:
        # 5. Chạy thuật toán
        print(f"\n[INFO] Solving...")
        
        raw_output = algorithm.solve(problem, seed=args.seed)
        
        # Chuẩn hóa dữ liệu để gửi cho Visualizer
        viz_data = {
            "problem": problem,
            "algorithm": algorithm.name,
            "result": raw_output["result"],
            "context": args.problem
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
            print("-> No visualizer available for this combination.")

    else: 
        # Incompatibility detected - provide helpful information
        print(f"\n[ERROR] Algorithm '{algorithm.name}' is NOT compatible with '{problem.name}'")
        print(f"Compatible problems for '{algorithm.name}' are: {', '.join(compatible_problems)}")

if __name__ == "__main__":
    main()