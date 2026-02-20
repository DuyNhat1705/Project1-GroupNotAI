import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer

class KnapsackVisualizer(BaseVisualizer):
    def __init__(self, problem, history, path=None, title="Knapsack Optimization"):
        # path is not used but kept for BaseVisualizer
        super().__init__(problem, history, path, title)

        self.weights = problem.weights
        self.values = problem.values
        self.capacity = problem.capacity
        self.num_items = problem.dimension

        # Pre-calculate best values for the convergence line plot
        self.best_values = []
        current_max = 0

        for solution in self.history:
            # solution is a binary array
            w = np.sum(solution * self.weights)
            # valid => calculate value / overweight => 0
            v = np.sum(solution * self.values) if w <= self.capacity else 0

            if v > current_max:
                current_max = v
            self.best_values.append(current_max)

    def animate(self):
        print(f"Visualization: {len(self.history)} iterations...")

        if not self.history:
            print("ERROR: Nothing to animate!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.5]})
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # Distinct colors for each item
        colors = plt.cm.tab20(np.linspace(0, 1, self.num_items))

        # --- update function ---
        def update(frame):
            ax1.clear()
            ax2.clear()

            solution = self.history[frame]
            selected_indices = np.where(solution == 1)[0]

            current_weight = 0
            current_value = 0

            # ========================================
            # PANEL 1: KNAPSACK CONTENT (BAR)
            # ========================================
            ax1.set_xlim(-0.5, 0.5)
            # Dynamic Y-limit to show overflow if it happens
            max_y = max(self.capacity * 1.2, np.sum(solution * self.weights) + 10)
            ax1.set_ylim(0, max_y)
            ax1.set_xticks([])
            ax1.set_ylabel("Weight", fontsize=12)
            ax1.set_title(f"Knapsack Arrangement\nIteration: {frame + 1} / {len(self.history)}", fontsize=14)

            # Capacity Limit Line
            ax1.axhline(self.capacity, color='red', linestyle='--', linewidth=3,
                        label=f'Capacity Limit: {self.capacity}')
            ax1.legend(loc="upper left")

            # Stack items
            bottom = 0
            for idx in selected_indices:
                w = self.weights[idx]
                v = self.values[idx]

                # Draw the item block
                ax1.bar(0, w, bottom=bottom, color=colors[idx], edgecolor='black', width=0.6)

                # Text label inside block (if tall enough to fit text)
                if w > (max_y * 0.04):
                    ax1.text(0, bottom + w / 2, f"Item {idx}\nW:{w} | V:{v}", ha='center', va='center', fontsize=9,
                             color='black', fontweight='bold')

                bottom += w
                current_value += v

            current_weight = bottom

            # Status Warning (Overweight/Valid)
            if current_weight > self.capacity:
                status_color = 'red'
                status_msg = f"OVERWEIGHT!\nWeight: {current_weight}\nValue: 0 (Invalid)"
            else:
                status_color = 'green'
                status_msg = f"VALID\nWeight: {current_weight}\nValue: {current_value}"

            # Display total weight/value
            ax1.text(0, max_y * -0.05, status_msg, ha='center', va='top', fontsize=12, fontweight='bold',
                     color=status_color, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

            # ========================================
            # PANEL 2: CONVERGENCE LINE PLOT
            # ========================================
            ax2.set_xlim(0, len(self.history))
            ax2.set_ylim(0, max(max(self.best_values) * 1.1, 10))
            ax2.set_title("Total Max Value Over Time", fontsize=14)
            ax2.set_xlabel("Iterations (Max Loops)", fontsize=12)
            ax2.set_ylabel("Best Value Found", fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # Draw progression line up to the current frame
            ax2.plot(range(frame + 1), self.best_values[:frame + 1], color='royalblue', linewidth=3)
            # Dot on the current frame
            ax2.scatter(frame, self.best_values[frame], color='red', s=80, zorder=5)

            # Text box for Best Solution details
            best_val = self.best_values[frame]
            ax2.text(0.05, 0.95, f"Best Value Found: {best_val}", transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # --- OPTIMIZE RENDER TIME ---
        # Skip frames if > 200, ensure last frame be hit
        total_frames = len(self.history)
        step = max(1, total_frames // 200)
        frames_to_render = list(range(0, total_frames, step))
        if frames_to_render[-1] != total_frames - 1:
            frames_to_render.append(total_frames - 1)

        ani = animation.FuncAnimation(fig, update, frames=frames_to_render, interval=100)

        # --- EXPORT MP4 ---
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        try:
            print(f"Saving to {self.save_path} ...")
            writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='KnapsackVisualizer'), bitrate=1800)
            ani.save(self.save_path, writer=writer)
            print("Done successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")

        plt.close()