# src/HandleCLI.py
import argparse


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
def handleCLI():
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

    return args, algo_params