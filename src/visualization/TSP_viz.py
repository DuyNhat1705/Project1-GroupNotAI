import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer


class TSPVisualizer(BaseVisualizer):
    def __init__(self, problem, history, path=None, title="TSP Visualization"):

        if isinstance(history, dict):
            if "population" in history:
                extracted_history = history["population"] # swarm algo
            elif "explored" in history:
                extracted_history = history["explored"] # SA and HC
            else:
                extracted_history = []
        else:
            extracted_history = history

        super().__init__(problem, extracted_history, path, title)

        self.coords = problem.coords
        self.dist_mat = problem.dist_mat
        self.city_names = getattr(problem, 'city', [])
        print(self.city_names)
        self.num_cities = problem.dimension

        # Safe Data Parsing
        self.best_costs = []
        self.solutions = []

        for item in self.history:
            if isinstance(item, tuple) and len(item) >= 2:
                path_data = item[0]
                cost = float(item[1])  # extract the numeric cost

                # SA passes [current_path, best_path]
                if isinstance(path_data, list) and len(path_data) == 2:
                    best_path = np.array(path_data[1])
                else:
                    best_path = np.array(path_data)

                self.solutions.append(best_path)
                self.best_costs.append(cost)

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
        ax1.set_title("Tour Map")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axis('off')

        # Draw City Nodes (arranged in circle)
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
        # PANEL 2: CONVERGENCE PLOT
        # ==========================================
        ax2.set_xlim(0, len(self.history))

        # Safe bounds using the cleaned numeric array
        if self.best_costs:
            ax2.set_ylim(min(self.best_costs) * 0.9, max(self.best_costs) * 1.1)

        ax2.set_title("Optimization Progress")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Total Distance")
        ax2.grid(True, linestyle='--', alpha=0.6)

        curve, = ax2.plot([], [], color='green', linewidth=2)
        dot, = ax2.plot([], [], 'ro')

        # ---------- update function (for animation) ----------------

        def update(frame):
            # Guaranteed to be a numpy array
            current_path = self.solutions[frame].astype(int)

            # Append start node to end to close the loop
            path_len = len(current_path)
            # print(path_len)
            is_full = path_len == self.num_cities

            # ---- draw path ----
            if is_full:
                draw_indices = np.append(current_path, current_path[0])
            else:
                draw_indices = current_path

            # Update Path Line
            path_x = self.coords[draw_indices, 0]
            path_y = self.coords[draw_indices, 1]
            line.set_data(path_x, path_y)

            if path_len < 2:
                return [line, title_text, curve, dot] + edge_labels

            # Update Edge Costs
            for i in range(path_len - (0 if is_full else 1)):
                u = current_path[i]
                v = current_path[(i + 1) % path_len] # ensure valid node

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
            
            # Hide unused labels
            for j in range(path_len, len(edge_labels)):
                edge_labels[j].set_text("")
            # Update text and convergence plot
            current_cost = self.best_costs[frame]
            title_text.set_text(f"Current Cost: {current_cost:.2f}")

            curve.set_data(range(frame + 1), self.best_costs[:frame + 1])
            dot.set_data([frame], [current_cost])

            return [line, title_text, curve, dot] + edge_labels

        total_frames = len(self.history)
        step = max(1, total_frames // 300) #frame-capping downsampler
        frames = range(0, total_frames, step)

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"Saving video to {self.save_path}...")
        writer = animation.FFMpegWriter(fps=30, bitrate=1800)
        ani.save(self.save_path, writer=writer)
        print("Done.")
        plt.close()