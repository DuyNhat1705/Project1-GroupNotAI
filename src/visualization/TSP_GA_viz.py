import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src.visualization.base_visualizer import BaseVisualizer

class TSPGAVisualizer(BaseVisualizer):
    def __init__(self, problem, history, logger=None, title="GA on TSP"):
        """
        Visualization cho GA-TSP
        - Dùng trực tiếp logger.best_solution & logger.best_fitness_gen
        - Đồng bộ hiển thị với TSP_viz cũ
        """
        super().__init__(problem, history, path=None, title=title)

        if logger is None:
            raise ValueError("TSPGAVisualizer requires GA logger")

        # ===== PROBLEM DATA =====
        self.problem = problem
        self.coords = problem.coords
        self.num_cities = problem.dimension

        # ===== READ FROM LOGGER =====
        self.best_solutions = logger.history["best_solution"]
        self.best_costs = logger.history["best_fitness_gen"]

        # ===== GLOBAL BEST =====
        self.reference_best = getattr(problem, "best_distance", None)

    # ======================================================
    # ANIMATION
    # ======================================================
    def animate(self):
        if not self.best_solutions:
            print("Nothing to animate.")
            return

        n_frames = len(self.best_solutions)

        # ================= FIGURE SETUP =================
        fig, (ax_map, ax_curve) = plt.subplots(
            1, 2,
            figsize=(15, 7),
            gridspec_kw={"width_ratios": [1.2, 1]}
        )
        fig.suptitle(self.title, fontsize=16, fontweight="bold")

        # ================= LEFT PANEL: TSP MAP =================
        pad = 20
        ax_map.set_xlim(self.coords[:, 0].min() - pad, self.coords[:, 0].max() + pad)
        ax_map.set_ylim(self.coords[:, 1].min() - pad, self.coords[:, 1].max() + pad)
        ax_map.set_title("Tour Map")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.axis("off")

        # City nodes (style giống TSP_viz cũ)
        ax_map.scatter(
            self.coords[:, 0],
            self.coords[:, 1],
            c="red",
            s=120,
            edgecolor="black",
            zorder=5
        )

        # Path line
        path_line, = ax_map.plot(
            [],
            [],
            color="royalblue",
            linewidth=3,
            alpha=0.8,
            zorder=3
        )

        # Info text (format y chang TSP_viz)
        info_text = ax_map.text(
            0.5,
            -0.06,
            "",
            transform=ax_map.transAxes,
            ha="center",
            fontsize=13,
            fontweight="bold"
        )

        # ================= RIGHT PANEL: CONVERGENCE =================
        ax_curve.set_xlim(0, n_frames)
        y_min = min(self.best_costs)
        y_max = max(self.best_costs)

        if self.reference_best is not None:
            y_min = min(y_min, self.reference_best)
            y_max = max(y_max, self.reference_best)

        ax_curve.set_ylim(y_min * 0.9, y_max * 1.1)
        ax_curve.set_title("Optimization Progress")
        ax_curve.set_xlabel("Iteration")
        ax_curve.set_ylabel("Total Distance")
        ax_curve.grid(True, linestyle="--", alpha=0.6)

        if self.reference_best is not None:
            ax_curve.axhline(
                y=self.reference_best,
                linestyle="--",
                color="gray",
                linewidth=2,
                alpha=0.8,
                label="Optimal (Reference)"
            )

        ax_curve.legend()
        curve, = ax_curve.plot([], [], color="green", linewidth=2)
        dot, = ax_curve.plot([], [], "ro")

        # ================= UPDATE FUNCTION =================
        def update(frame):
            # ---- TOUR ----
            tour = self.best_solutions[frame].astype(int)
            tour = np.append(tour, tour[0])  # close loop

            x = self.coords[tour, 0]
            y = self.coords[tour, 1]
            path_line.set_data(x, y)

            # ---- INFO TEXT ----
            cost = self.best_costs[frame]
            current_best = min(self.best_costs[:frame + 1])

            info_text.set_text(
                f"Iter: {frame} | Cost: {cost:.2f} | Global Best: {current_best:.2f}"
            )

            # ---- CONVERGENCE ----
            curve.set_data(range(frame + 1), self.best_costs[:frame + 1])
            dot.set_data([frame], [cost])

            return path_line, curve, dot, info_text

        # ================= SPEED CONTROL =================
        # 1 generation = 1 frame (KHÔNG skip)
        frames = range(n_frames)
        interval = 120   # ms – chậm, dễ quan sát
        fps = 10         # video không bị tua nhanh

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False
        )

        # ================= SAVE VIDEO =================
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        print(f"Saving GA-TSP visualization to {self.save_path}")

        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(self.save_path, writer=writer)

        plt.close()
        print("Done.")