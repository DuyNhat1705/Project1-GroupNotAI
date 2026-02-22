import os
import numpy as np
from src.visualization.base_visualizer import BaseVisualizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

class MazeVisualizer(BaseVisualizer):
    def __init__(self,problem,history,path, title):
        super().__init__(problem, history, path, title)

    def animate_maze(self):
        print(f"Visualization: {len(self.history)} search steps + {len(self.path)} path steps...")

        if not self.history and not self.path:
            print("ERROR: Nothing to animate!")
            return

        maze = self.problem.maze
        start = self.problem.start
        goal = self.problem.goal

        rows, cols = maze.shape

        # --- SETUP PLOT ---
        fig, ax = plt.subplots(figsize=(12,10), constrained_layout=True)
        ax.set_title(self.title, fontsize=14, fontweight='bold')

        # Draw Maze
        cmap = ListedColormap(["white", "dimgray"])
        ax.imshow(maze, cmap=cmap, origin="upper") # y-axis upside down
        ax.set_aspect("equal")

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)

        # Grid
        for x in range(cols + 1):
            ax.plot([x - 0.5, x - 0.5], [-0.5, rows - 0.5],
                    color='black', linewidth=1,antialiased=False)

        for y in range(rows + 1):
            ax.plot([-0.5, cols - 0.5], [y - 0.5, y - 0.5],
                    color='black', linewidth=1,antialiased=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # --- STATIC START & GOAL (Swap to X=col, Y=row) ---
        # Note: start[1] is col (x), start[0] is row (y)
        ax.scatter(start[1], start[0], c="green", s=250, zorder=10, label="Start")
        ax.scatter(goal[1], goal[0], c="red", s=250, zorder=10, label="Goal")

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            labelspacing=1,
            frameon=True)

        # --- STATUS TEXT ---
        status_text = ax.text(
            0.02, 1.02,
            "Initializing...",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray')
        )

        explored_patches = []
        path_lines = []

        # --- UPDATE FUNCTION ---
        def update(frame):

            # --- Exploration Phase ---
            if frame < len(self.history):

                # Unpack the edge tuple from BFS/DFS
                current_node, neigh_node = self.history[frame]
                r, c = neigh_node  # We draw the newly visited node

                # Swap to (x, y) -> (c, r)
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5),  # Swapped r and c
                    1, 1,
                    facecolor="orange",
                    alpha=0.5,
                    zorder=5
                )

                ax.add_patch(rect)
                explored_patches.append(rect)

                status_text.set_text(f"Exploring: ({r}, {c})")

            # --- Path Reconstruction ---
            else:
                path_idx = frame - len(self.history)

                if path_idx < len(self.path) - 1:
                    r1, c1 = self.path[path_idx]
                    r2, c2 = self.path[path_idx + 1]

                    # FIX 3: Swap to X=[c1, c2], Y=[r1, r2]
                    line, = ax.plot(
                        [c1, c2],
                        [r1, r2],
                        color="royalblue",
                        linewidth=4,
                        zorder=20
                    )

                    path_lines.append(line)

                    status_text.set_text(f"Reconstructing Path: ({r1},{c1}) → ({r2},{c2})")

            return explored_patches + path_lines + [status_text]

        total_frames = len(self.history) + len(self.path) + 5

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=150,
            blit=True
        )

        # --- EXPORT MP4 ---
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        try:
            print(f"Saving to {self.save_path} ...")

            writer = animation.FFMpegWriter(
                fps=5,
                metadata=dict(artist='MazeSolver'),
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
