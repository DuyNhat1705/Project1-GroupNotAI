import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer


class TSPVisualizer(BaseVisualizer):
    def __init__(self, problem, history, path=None, title="TSP Visualization"):
        super().__init__(problem, history, path, title)

        self.coords = problem.coords
        self.dist_mat = problem.dist_mat  # Get the distance matrix
        self.city_names = getattr(problem, 'city_names', [])
        self.num_cities = problem.dimension

        self.best_costs = [step[1] for step in self.history]
        self.solutions = [step[0] for step in self.history]

    def animate(self):
        print(f"Visualization: {len(self.history)} steps...")

        if not self.history:
            print("ERROR: Nothing to animate!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1.2, 1]})
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        # ==========================================
        # PANEL 1: SETUP MAP & SPANNING PATH
        # ==========================================
        pad = 20
        ax1.set_xlim(self.coords[:, 0].min() - pad, self.coords[:, 0].max() + pad)
        ax1.set_ylim(self.coords[:, 1].min() - pad, self.coords[:, 1].max() + pad)
        ax1.set_title("Tour Map (Spanning Path & Edge Costs)")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axis('off')

        # Draw City Nodes
        ax1.scatter(self.coords[:, 0], self.coords[:, 1], c='red', s=120, edgecolor='black', zorder=5)

        # City Names
        for i, (x, y) in enumerate(self.coords):
            label = self.city_names[i] if i < len(self.city_names) else f"C{i}"
            ax1.text(x * 1.15, y * 1.15, label, fontsize=10, ha='center', va='center', fontweight='bold',
                     color='midnightblue')

        # Highlight the active path
        line, = ax1.plot([], [], color='royalblue', linewidth=3, alpha=0.8, zorder=3)

        # text for the cost of each edge
        edge_labels = [ax1.text(0, 0, "", fontsize=8, color='crimson', fontweight='bold',
                                ha='center', va='center', zorder=4,
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
                       for _ in range(self.num_cities)]

        title_text = ax1.text(0.5, -0.05, "", transform=ax1.transAxes, ha='center', fontsize=13, fontweight='bold')

        # ==========================================
        # PANEL 2: SETUP CONVERGENCE PLOT
        # ==========================================
        ax2.set_xlim(0, len(self.history))
        ax2.set_ylim(min(self.best_costs) * 0.9, max(self.best_costs) * 1.1)
        ax2.set_title("Optimization Progress")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Total Distance")
        ax2.grid(True, linestyle='--', alpha=0.6)

        curve, = ax2.plot([], [], color='green', linewidth=2)
        dot, = ax2.plot([], [], 'ro')

        # ---------- update function (for animation) ----------------

        def update(frame):
            current_path = self.solutions[frame].astype(int)
            # Append start node to end to close the loop
            draw_indices = np.append(current_path, current_path[0])

            # Update Path Line
            path_x = self.coords[draw_indices, 0]
            path_y = self.coords[draw_indices, 1]
            line.set_data(path_x, path_y)

            # Update Edge Costs
            for i in range(self.num_cities):
                u = current_path[i]
                v = current_path[(i + 1) % self.num_cities]

                # midpoint of the edge
                x1, y1 = self.coords[u]
                x2, y2 = self.coords[v]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # actual cost from distance matrix
                cost = self.dist_mat[u][v]

                # Update position and value
                edge_labels[i].set_position((mid_x, mid_y))
                edge_labels[i].set_text(f"{cost:.0f}")

            # Update text and convergence plot
            current_cost = self.best_costs[frame]
            title_text.set_text(f"Iter: {frame} | Total Cost: {current_cost:.2f}")

            curve.set_data(range(frame + 1), self.best_costs[:frame + 1])
            dot.set_data([frame], [current_cost])

            return [line, title_text, curve, dot] + edge_labels

        total_frames = len(self.history)
        step = max(1, total_frames // 300)
        frames = range(0, total_frames, step)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"Saving video to {self.save_path}...")
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(self.save_path, writer=writer)
        print("Done.")
        plt.close()