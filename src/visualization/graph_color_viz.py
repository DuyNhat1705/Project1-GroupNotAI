import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer


class GraphColoringVisualizer(BaseVisualizer):
    def __init__(self, problem, history, path=None, title="Graph Coloring"):
        super().__init__(problem, history, path if path else [], title)
        # built-in Matplotlib colormap with 20 distinct colors
        self.cmap = plt.colormaps.get_cmap('tab20')

    def animate(self):
        if not self.history:
            print("ERROR: Nothing to animate!")
            return

        print(f"Visualization: Animating {len(self.history)} frames for {self.title}...")

        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(8, 8))
        coords = self.problem.node_coords

        def update(frame):
            ax.clear()
            ax.axis('off')

            # Extract current solution array from history
            current_solution = self.history[frame]
            if isinstance(current_solution, tuple):
                current_solution = current_solution[0]

            colors = np.round(current_solution).astype(int)

            # Live Metrics
            conflicts = sum(1 for u, v in self.problem.edges if colors[u] == colors[v])
            unique_colors = len(np.unique(colors))

            # Draw Edges
            for u, v in self.problem.edges:
                p1, p2 = coords[u], coords[v]

                # Highlight conflicts in RED, valid as GRAY
                if colors[u] == colors[v]:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=3, zorder=1)
                else:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linewidth=1, zorder=1)

            # Draw Nodes
            xs = [coords[i][0] for i in range(self.problem.num_nodes)]
            ys = [coords[i][1] for i in range(self.problem.num_nodes)]

            # Map the int arr to hex colors
            node_colors = [self.cmap(colors[i] % 20) for i in range(self.problem.num_nodes)]

            ax.scatter(xs, ys, c=node_colors, s=800, zorder=5, edgecolors='black', linewidths=2)

            # Add Node ID Text
            for i in range(self.problem.num_nodes):
                ax.annotate(str(i), (xs[i], ys[i]), ha='center', va='center',
                            color='white', fontweight='bold', fontsize=12, zorder=6)

            # Titles & Final Frame Highlight
            if frame == len(self.history) - 1:
                ax.set_title(f"FINAL SOLUTION REACHED!\nLeast Colors Used: {unique_colors} | Conflicts: {conflicts}",
                             fontsize=16, fontweight='bold', color='darkgreen')
                fig.patch.set_facecolor('#e8f5e9')  # Soft green background success flash
            else:
                ax.set_title(f"{self.title}\nIteration {frame + 1} | Colors: {unique_colors} | Conflicts: {conflicts}",
                             fontsize=14)

        # Generate Animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.history), interval=150, repeat=False)

        # --- EXPORT MP4 ---
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        try:
            print(f"Saving to {self.save_path} ...")
            writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Here_later'), bitrate=1800)
            ani.save(self.save_path, writer=writer)
            print("Done successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")
            try:
                gif_path = self.save_path.replace(".mp4", ".gif")
                ani.save(gif_path, writer="pillow", fps=10)
                print(f"Saved as GIF instead: {gif_path}")
            except:
                pass

        plt.close()