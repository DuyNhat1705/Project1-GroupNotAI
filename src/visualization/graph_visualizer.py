import os
from src.visualization.base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GraphVisualizer(BaseVisualizer):
    def __init__(self,problem,history,path,title):
        super().__init__(problem,history,path,title)

    def animate_graph(self):
        print(f"Visualization: {len(self.history)} search steps + {len(self.path)} path steps...")

        if not self.history and not self.path:
            print("ERROR: Nothing to animate!")
            return

        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        ax.axis('off')

        coords = self.problem.node_coords

        # --- STATIC ELEMENTS: Nodes & Edges ---
        drawn_edges = set()

        # Draw Background Edges
        for u in self.problem.adj_list:
            for v, weight in self.problem.adj_list[u].items():
                edge_id = tuple(sorted((str(u), str(v))))
                if edge_id not in drawn_edges:
                    if str(u) in coords and str(v) in coords:
                        p1 = coords[str(u)]
                        p2 = coords[str(v)]

                        # Grey line
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='lightgray', zorder=1, alpha=0.5)

                        # Weight Label
                        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        ax.text(mid_x, mid_y, str(weight), fontsize=8, color='gray',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

                        drawn_edges.add(edge_id)

        # Draw Nodes
        xs, ys, labels = [], [], []
        for node_id, (x, y) in coords.items():
            xs.append(x)
            ys.append(y)
            labels.append(node_id)

        # set node color
        ax.scatter(xs, ys, c='cornflowerblue', s=200, zorder=10, edgecolors='white')

        # Node Labels
        for i, txt in enumerate(labels):
            ax.annotate(txt, (xs[i], ys[i]), ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white', zorder=11)

        # Highlight Start and Goal
        if self.problem.start and self.problem.goal:
            sx, sy = coords[str(self.problem.start)]
            ex, ey = coords[str(self.problem.goal)]
            ax.scatter([sx], [sy], c='green', s=400, zorder=9, alpha=0.3, label="Start")
            ax.scatter([ex], [ey], c='red', s=400, zorder=9, alpha=0.3, label="Goal")

        # --- DYNAMIC ELEMENT ---
        status_text = ax.text(0.02, 0.98, "Initializing...", transform=ax.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

        lines_explored = []
        lines_path = []

        def update(frame):
        # Traversal phase
            if frame < len(self.history):
                try:
                    u, v = self.history[frame]
                    p1 = coords[str(u)]
                    p2 = coords[str(v)]

                    # Exploration: Orange dashed line
                    line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color='orange', alpha=0.6, linestyle='--', linewidth=2, zorder=5)
                    lines_explored.append(line)
                    status_text.set_text(f"Exploring: {u} -> {v}")
                except Exception as e:
                    print(f"Frame error: {e}")

            # Complete path highlight
            else:
                path_idx = frame - len(self.history)
                if path_idx < len(self.path) - 1:
                    u, v = self.path[path_idx], self.path[path_idx + 1]
                    p1 = coords[str(u)]
                    p2 = coords[str(v)]

                    # Path: Solid Blue line
                    line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color='royalblue', alpha=1.0, linewidth=4, zorder=20)
                    lines_path.append(line)

                    status_text.set_text(f"Reconstructing Path: {u} -> {v}")

            return lines_explored + lines_path + [status_text]

        total_frames = len(self.history) + len(self.path) + 5  # +5 for a pause at the end

        # Create Animation
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=150, blit=True)

        # --- Export MP4 (FFmpeg) ---
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        try:
            print(f"Saving to {self.save_path} ...")

            writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='GraphSolver'), bitrate=1800) # FFmpeg WRITER

            ani.save(self.save_path, writer=writer)
            print("Done successfully.")

        except FileNotFoundError:
            print("\nERROR: FFmpeg not found!")

        except Exception as e:
            print(f"Error saving video: {e}")

        plt.close()
    def animate(self):
        self.animate_graph()