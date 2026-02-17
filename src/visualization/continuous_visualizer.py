import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer

class ContinuousVisualizer(BaseVisualizer):
    def __init__(self, problem, history, title, metrics=None, grid_search_data=None):
        """
        history: List các quần thể (pop_hist) để làm animation.
        metrics: Dict chứa {'best_fit': [], 'avg_fit': []} để vẽ biểu đồ hội tụ.
        grid_search_data: Dict chứa kết quả Grid Search {'Z_mean', 'Z_std', 'Z_fail', 'x_axis', 'y_axis', 'x_label', 'y_label'}
        """
        # BaseVisualizer init
        super().__init__(problem, history, path=[], title=title)
        
        self.metrics = metrics
        self.grid_data = grid_search_data
        
        # Đường dẫn save SVG
        self.svg_path = os.path.join(BaseVisualizer.project_root, 'output', f"{title}_performance.svg")

    def animate(self):
        """
        Tạo video MP4 minh họa quá trình tìm kiếm trên Contour plot.
        """
        if not self.history:
            print("No history to animate.")
            return

        print(f"Visualization: Generating Animation for {self.title}...")
        
        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Contour Background
        resolution = 100
        # Lấy bounds từ problem
        min_r = self.problem.min_range
        max_r = self.problem.max_range
        
        x = np.linspace(min_r, max_r, resolution)
        y = np.linspace(min_r, max_r, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        # Tính giá trị Z cho contour
        # Lưu ý: problem.evaluate của hệ thống mới nhận vector 1D
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = self.problem.evaluate(np.array([X[i, j], Y[i, j]]))

        # Vẽ contour
        ax.contourf(X, Y, Z, levels=50, cmap="viridis", alpha=0.7)

        # 2. Vẽ Global Minimum (red star)
        if hasattr(self.problem, 'global_x') and self.problem.global_x is not None:
            # Chỉ vẽ nếu global_x có độ dài >= 2
            gx = self.problem.global_x
            if len(gx) >= 2:
                ax.scatter(gx[0], gx[1], 
                           c='red', marker='*', s=200, edgecolors='white', 
                           label='Global Min', zorder=10)
                ax.legend(loc='upper right')

        # 3. Scatter Plot cho quần thể (Màu cam)
        scat = ax.scatter([], [], c="orange", s=30, edgecolors='black', zorder=15)
        title_text = ax.set_title(f"{self.title} - Init")

        def update(frame):
            current_pop = self.history[frame]

            # 2. Standardize the data format!
            if isinstance(current_pop, tuple):
                current_pop = current_pop[0]

            # Convert to numpy array
            current_pop = np.array(current_pop)

            # If a single 1D point (SA or ABC's current_best), wrap in a 2D array => avoid crash
            if current_pop.ndim == 1:
                current_pop = np.expand_dims(current_pop, axis=0)

            scat.set_offsets(current_pop[:, :2]) 
            title_text.set_text(f"{self.title} - Gen {frame}/{len(self.history)}")
            return scat, title_text

        # --- EXPORT VIDEO ---
        ani = animation.FuncAnimation(
            fig, update,
            frames=len(self.history),
            interval=100,
            blit=True
        )

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        try:
            print(f"Saving video to {self.save_path} ...")
            writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='AI_Solver'), bitrate=1800)
            ani.save(self.save_path, writer=writer)
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving MP4 (Check FFmpeg): {e}")
            try:
                # Fallback GIF
                gif_path = self.save_path.replace(".mp4", ".gif")
                ani.save(gif_path, writer="pillow", fps=15)
                print(f"Saved as GIF instead: {gif_path}")
            except:
                pass
        
        plt.close()

    def analyze_performance(self):
        """
        Vẽ 4 biểu đồ: Convergence, Sensitivity, Failure Rate, Robustness.
        Xuất ra file .svg
        """
        if not self.metrics or not self.grid_data:
            print("Thiếu dữ liệu metrics hoặc grid search để vẽ biểu đồ Performance.")
            return

        print(f"Visualization: Plotting Performance Analysis to {self.svg_path}...")

        best_hist = self.metrics.get('best_fit', [])
        avg_hist = self.metrics.get('avg_fit', [])
        
        Z_mean = self.grid_data.get('Z_mean')
        Z_fail = self.grid_data.get('Z_fail')
        Z_std  = self.grid_data.get('Z_std')
        
        x_axis = self.grid_data.get('x_axis') # VD: CR_LIST hoặc FLUCTUATIONS
        y_axis = self.grid_data.get('y_axis') # VD: F_LIST hoặc MUTATION_RATES
        
        x_label = self.grid_data.get('x_label', 'Parameter 1')
        y_label = self.grid_data.get('y_label', 'Parameter 2')
        is_log_scale = self.grid_data.get('log_scale_x', False)

        # --- PLOTTING ---
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Convergence Curve (Top-Left)
        if len(best_hist) > 0:
            axs[0, 0].plot(best_hist, label="Best Fitness (Avg)", color='red', linewidth=2)
            axs[0, 0].plot(avg_hist, label="Population Mean (Avg)", color='blue', alpha=0.5, linestyle='--')
            axs[0, 0].set_title("Convergence Curve (Best Params)")
            axs[0, 0].set_xlabel("Generation")
            axs[0, 0].set_ylabel("Fitness Value")
            axs[0, 0].grid(True, linestyle=':', alpha=0.6)
            axs[0, 0].legend()
            if is_log_scale: # Nếu fitness giảm quá sâu (về 1e-10)
                axs[0, 0].set_yscale('log')

        # Setup Meshgrid cho Heatmap
        # Lưu ý: Contourf cần X, Y đúng chiều.
        # X tương ứng với cột (axis 1 của Z), Y tương ứng với hàng (axis 0 của Z)
        X, Y = np.meshgrid(x_axis, y_axis)

        # Cấu hình log scale cho trục heatmap nếu cần (VD: GA Fluctuation)
        if is_log_scale:
            axs[0, 1].set_xscale('log')
            axs[1, 0].set_xscale('log')
            axs[1, 1].set_xscale('log')

        # 2. Parameter Sensitivity - Mean Fitness (Top-Right)
        if Z_mean is not None:
            # viridis_r: Tím (Thấp - Tốt) -> Vàng (Cao - Xấu)
            c1 = axs[0, 1].contourf(X, Y, Z_mean, levels=20, cmap="viridis_r")
            fig.colorbar(c1, ax=axs[0, 1], label="Mean Best Fitness")
            axs[0, 1].set_title("Sensitivity: Mean Fitness")
            axs[0, 1].set_xlabel(x_label)
            axs[0, 1].set_ylabel(y_label)

        # 3. Failure Rate (Bottom-Left)
        if Z_fail is not None:
            # Reds: Trắng (0) -> Đỏ Đậm (1)
            c2 = axs[1, 0].contourf(X, Y, Z_fail, levels=10, cmap="Reds", vmin=0, vmax=1)
            fig.colorbar(c2, ax=axs[1, 0], label="Failure Rate (0-1)")
            axs[1, 0].set_title("Sensitivity: Failure Rate")
            axs[1, 0].set_xlabel(x_label)
            axs[1, 0].set_ylabel(y_label)

        # 4. Robustness - Std Dev (Bottom-Right)
        if Z_std is not None:
            # Blues: Nhạt (Ổn định) -> Đậm (Biến động)
            c3 = axs[1, 1].contourf(X, Y, Z_std, levels=20, cmap="Blues")
            fig.colorbar(c3, ax=axs[1, 1], label="Std Dev")
            axs[1, 1].set_title("Robustness: Stability")
            axs[1, 1].set_xlabel(x_label)
            axs[1, 1].set_ylabel(y_label)

        plt.suptitle(f"{self.title} – Benchmark Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save SVG
        os.makedirs(os.path.dirname(self.svg_path), exist_ok=True)
        plt.savefig(self.svg_path)
        print(f"Saved performance chart to: {self.svg_path}")
        plt.close()