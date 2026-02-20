import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from src.visualization.base_visualizer import BaseVisualizer

class GAMazeVisualizer(BaseVisualizer):
    """
    Visualizer cho Genetic Algorithm trên Maze.
    Kế thừa giao diện & logic từ MazeVisualizer,
    bổ sung hiển thị Best Fitness theo từng generation.
    """

    def __init__(self, problem, history, path, title):
        super().__init__(problem, history, path, title)

    def animate(self):
        self.animate_maze()

    def animate_maze(self):
        print(f"[GA-Maze] Visualization: {len(self.history)} generations, path length = {len(self.path)}")

        maze = self.problem.maze
        start = self.problem.start
        goal = self.problem.goal

        rows, cols = maze.shape

        fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)
        ax.set_title(self.title, fontsize=14, fontweight='bold')

        # --- DRAW MAZE ---
        cmap = ListedColormap(["white", "dimgray"])
        ax.imshow(maze, cmap=cmap, origin="upper")
        ax.set_aspect("equal")

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        # Grid
        for x in range(cols + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, rows - 0.5], color='black', linewidth=1)
        for y in range(rows + 1):
            ax.plot([-0.5, cols - 0.5], [y - 0.5, y - 0.5], color='black', linewidth=1)

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Start & Goal
        ax.scatter(start[1], start[0], c="green", s=250, zorder=10, label="Start")
        ax.scatter(goal[1], goal[0], c="red", s=250, zorder=10, label="Goal")

        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

        # --- TEXT BOX ---
        status_text = ax.text(
            0.02, 1.05,
            "Initializing GA...",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray')
        )

        # GA history: giả định history = list(best_fitness)
        best_fitness_hist = self.history

        explored_patches = []
        path_lines = []

        def update(frame):
            # --- GENERATION UPDATE ---
            if frame < len(best_fitness_hist):
                bf = best_fitness_hist[frame]
                status_text.set_text(f"Generation {frame + 1} | Best Fitness = {bf:.4f}")

            # --- PATH DRAW ---
            else:
                path_idx = frame - len(best_fitness_hist)
                if path_idx < len(self.path) - 1:
                    r1, c1 = self.path[path_idx]
                    r2, c2 = self.path[path_idx + 1]

                    line, = ax.plot(
                        [c1, c2],
                        [r1, r2],
                        color="royalblue",
                        linewidth=4,
                        zorder=20
                    )
                    path_lines.append(line)

                    status_text.set_text(
                        f"Final Path: ({r1},{c1}) → ({r2},{c2}) | Best Fitness = {best_fitness_hist[-1]:.4f}"
                    )

            return explored_patches + path_lines + [status_text]

        total_frames = len(best_fitness_hist) + len(self.path) + 5

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=200,
            blit=True
        )

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        try:
            print(f"Saving GA Maze visualization to {self.save_path}")
            writer = animation.FFMpegWriter(fps=5, bitrate=1800)
            ani.save(self.save_path, writer=writer)
            print("[GA-Maze] Done.")
        except Exception as e:
            print(f"[GA-Maze] Error: {e}")

        plt.close()