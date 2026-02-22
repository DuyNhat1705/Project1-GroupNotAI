import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from src.visualization.base_visualizer import BaseVisualizer

class ContinuousVisualizer(BaseVisualizer):
    def __init__(self, problem, history, title, metrics=None, grid_search_data=None):
        super().__init__(problem, history, path=[], title=title)

        self.problem = problem
        self.history = history
        self.title = title

        self.metrics = metrics or {}
        self.grid_search_data = grid_search_data

        self.save_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.save_dir, exist_ok=True)

        self.save_path = os.path.join(self.save_dir, f"{title}.mp4")

    def animate(self):
        if not self.history:
            print("[VIS] No history to animate.")
            return

        print(f"[VIS] Rendering animation: {self.title}")
        print(f"[VIS] Output → {self.save_path}")

        # ===== FIGURE LAYOUT =====
        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 0.6])

        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])
        ax_conv = fig.add_subplot(gs[1, 0])
        ax_text = fig.add_subplot(gs[1, 1])
        ax_text.axis("off")

        # ===== FUNCTION LANDSCAPE =====
        res = 80
        min_r, max_r = self.problem.min_range, self.problem.max_range
        x = np.linspace(min_r, max_r, res)
        y = np.linspace(min_r, max_r, res)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                Z[i, j] = self.problem.evaluate(np.array([X[i, j], Y[i, j]]))

        # ===== STATIC PLOTS =====
        ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85)
        ax3d.set_title("3D Objective Surface")

        ax2d.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
        ax2d.set_title("Search Space (Top View)")

        if hasattr(self.problem, "global_x"):
            gx = self.problem.global_x
            ax2d.scatter(gx[0], gx[1], c="red", marker="*", s=180)

        scat = ax2d.scatter([], [], c="orange", s=35, edgecolors="black")
        best_dot_3d, = ax3d.plot([], [], [], "r*", markersize=12)

        # ===== CONVERGENCE =====
        best_hist = self.metrics.get("best_fitness", [])
        avg_hist  = self.metrics.get("avg_fitness", [])

        conv_best, = ax_conv.plot([], [], "r-", lw=2, label="Best")
        conv_avg, = ax_conv.plot([], [], "b--", lw=1.5, label="Average")

        ax_conv.set_title("Convergence")
        ax_conv.set_xlabel("Generation")
        ax_conv.set_ylabel("Fitness")
        ax_conv.legend()
        ax_conv.grid(True, linestyle=":")

        if best_hist:
            ax_conv.set_xlim(0, len(best_hist))
            ymin, ymax = min(best_hist), max(best_hist)
            if ymin == ymax:
                ymax += 1e-6
            ax_conv.set_ylim(ymin * 0.95, ymax * 1.05)

        # ===== UPDATE =====
        def update(frame):
            pop = self.history[frame]
            if isinstance(pop, tuple):
                pop = pop[0]
            pop = np.asarray(pop)

            if pop.ndim == 1:
                pop = pop.reshape(1, -1)

            scat.set_offsets(pop[:, :2])

            fitness = [self.problem.evaluate(ind) for ind in pop]
            idx = np.argmin(fitness)
            best = pop[idx]
            bz = fitness[idx]

            best_dot_3d.set_data([best[0]], [best[1]])
            best_dot_3d.set_3d_properties([bz])

            ax3d.view_init(elev=35, azim=frame * 3)

            if frame < len(best_hist):
                x = np.arange(frame + 1)
                conv_best.set_data(x, best_hist[:frame + 1])

                if frame < len(avg_hist):
                    conv_avg.set_data(x, avg_hist[:frame + 1])

            ax_text.clear()
            ax_text.axis("off")
            ax_text.text(0.05, 0.85, f"Generation: {frame}", fontsize=12, weight="bold")
            ax_text.text(0.05, 0.65, f"Best fitness: {bz:.6f}")
            ax_text.text(0.05, 0.50, f"x: {best[0]:.4f}")
            ax_text.text(0.05, 0.40, f"y: {best[1]:.4f}")

            return scat, best_dot_3d, conv_best, conv_avg

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=120,
            blit=False,
        )

        writer = animation.FFMpegWriter(fps=15, bitrate=1800)
        ani.save(self.save_path, writer=writer)

        plt.close()
        print("[VIS] Done.")