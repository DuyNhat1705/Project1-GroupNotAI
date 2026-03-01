import os
from src.visualization.base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

class GAMazeVisualizer(BaseVisualizer):
    def __init__(self, problem, history, path, title):
        super().__init__(problem, history, path, title)

    def animate_maze(self):
        # --- TẠO KỊCH BẢN FRAME (STORYBOARD) ---
        animation_frames = []
        last_path = None
        goal = tuple(self.problem.goal) 
        
        for gen_idx, path in enumerate(self.history):
            if path is None:
                continue
            
            path_tuples = [tuple(p) for p in path]
                
            if last_path is None or path_tuples != last_path:
                # 1. Reset: Tạo 3 frame rỗng để xóa sạch ô vuông và đường vẽ
                if last_path is not None:
                    for _ in range(3):
                        animation_frames.append((gen_idx, [], f"Gen {gen_idx + 1} | Better path found! Resetting...", False))
                    
                # 2. Khám phá: Vẽ từ từ từng ô vuông
                for step in range(1, len(path_tuples) + 1):
                    partial_path = path_tuples[:step]
                    # Kiểm tra xem đường đi ở bước này đã chạm đích chưa
                    is_goal_reached = (partial_path[-1] == goal)
                    
                    animation_frames.append((gen_idx, partial_path, f"Gen {gen_idx + 1} | Exploring step {step}...", is_goal_reached))
                    
                last_path = path_tuples
            else:
                # Nếu không có đường mới tốt hơn, duy trì trạng thái hiện tại
                is_goal_reached = (last_path[-1] == goal)
                animation_frames.append((gen_idx, last_path, f"Gen {gen_idx + 1} | Best Length: {len(last_path)-1} (Searching...)", is_goal_reached))
                
        # Frame kết thúc
        final_gen = len(self.history)
        for _ in range(30):
            if last_path is not None:
                is_goal_reached = (last_path[-1] == goal)
                animation_frames.append((final_gen, last_path, f"DONE! Final Best Length: {len(last_path)-1}", is_goal_reached))

        print(f"Visualization: Created storyboard with {len(animation_frames)} frames...")

        if not animation_frames:
            print("ERROR: Nothing to animate!")
            return

        maze = self.problem.maze
        start = self.problem.start
        rows, cols = maze.shape

        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        ax.set_title(self.title, fontsize=14, fontweight='bold')

        cmap = ListedColormap(["white", "dimgray"])
        ax.imshow(maze, cmap=cmap, origin="upper") 

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        for x in range(cols + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, rows - 0.5], color='black', linewidth=1)
        for y in range(rows + 1):
            ax.plot([-0.5, cols - 0.5], [y - 0.5, y - 0.5], color='black', linewidth=1)
                    
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # --- STATIC START & GOAL ---
        ax.scatter(start[1], start[0], c="green", s=250, zorder=10, label="Start")
        ax.scatter(goal[1], goal[0], c="red", s=250, zorder=10, label="Goal")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), labelspacing=1, frameon=True)

        status_text = ax.text(
            0.02, 1.02, "Initializing...",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # Khởi tạo một đường thẳng (dành cho lúc chạm đích)
        path_line, = ax.plot([], [], color="royalblue", linewidth=4, zorder=20)

        # TẠO "BỂ CHỨA" CÁC Ô VUÔNG (RECTANGLES POOL) ĐỂ TÔ MÀU
        max_len = max([len(p) for p in self.history if p is not None] + [0])
        rects = [plt.Rectangle((0, 0), 1, 1, facecolor="orange", alpha=0.5, visible=False, zorder=5) for _ in range(max_len)]
        for r in rects:
            ax.add_patch(r)

        # --- UPDATE FUNCTION ---
        def update(frame_idx):
            gen_idx, partial_path, text_str, is_goal_reached = animation_frames[frame_idx]
            
            # 1. Cập nhật các ô vuông tô màu
            for i, r in enumerate(rects):
                if i < len(partial_path):
                    r_row, c_col = partial_path[i]
                    # Dịch tọa độ hiển thị sao cho căn giữa ô
                    r.set_xy((c_col - 0.5, r_row - 0.5))
                    r.set_visible(True)
                else:
                    r.set_visible(False)
            
            # 2. Cập nhật đường vẽ line xanh (chỉ vẽ khi chạm đích)
            if is_goal_reached:
                x_coords = [c for r, c in partial_path]
                y_coords = [r for r, c in partial_path]
                path_line.set_data(x_coords, y_coords)
            else:
                path_line.set_data([], [])

            status_text.set_text(text_str)
            # Hàm blit yêu cầu trả về danh sách các đối tượng thay đổi, ta gộp mảng rects và đường vẽ
            return rects + [path_line, status_text]

        # --- TẠO ANIMATION ---
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(animation_frames),
            interval=30, 
            blit=True
        )

        # --- EXPORT MP4 ---
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        try:
            print(f"Saving to {self.save_path} ...")
            writer = animation.FFMpegWriter(
                fps=30,
                metadata=dict(artist='GAMazeSolver'),
                bitrate=1800
            )
            ani.save(self.save_path, writer=writer)
            print("Done successfully.")

        except FileNotFoundError:
            print("\nERROR: FFmpeg not found!")
        except Exception as e:
            print(f"Error saving video: {e}")

        plt.close()

    def animate(self):
        self.animate_maze()