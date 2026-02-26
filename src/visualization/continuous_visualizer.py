import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
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

        # ===== FIGURE LAYOUT  =====
        fig = plt.figure(figsize=(16, 9))

        fig.suptitle(self.title, fontsize=18, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(3, 2, width_ratios=[1.6, 1], height_ratios=[1.2, 1, 0.3])

        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax2d = fig.add_subplot(gs[0, 1])
        ax_conv = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[2, 1])
        ax_text.axis("off")

        # ===== FUNCTION LANDSCAPE =====
        res = 101  # Odd number for perfect center sampling
        min_r, max_r = self.problem.min_range, self.problem.max_range
        x = np.linspace(min_r, max_r, res)
        y = np.linspace(min_r, max_r, res)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(res):
            for j in range(res):
                Z[i, j] = self.problem.evaluate(np.array([X[i, j], Y[i, j]]))

        # ===== STATIC PLOTS =====
        # 3D View
        ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.40)
        ax3d.set_title("3D Objective Surface", fontsize=14, fontweight='bold')
        ax3d.set_xlabel('X Axis')
        ax3d.set_ylabel('Y Axis')

        # 2D Top View
        ax2d.contourf(X, Y, Z, levels=40, cmap="viridis", alpha=0.8)
        ax2d.set_title("Search Space (Top View)", fontsize=12, fontweight='bold')

        # True Global Minimum (Red Star)
        if hasattr(self.problem, "global_x") and self.problem.global_x is not None:
            gx = self.problem.global_x

            # Draw on 2D Top View
            ax2d.scatter(gx[0], gx[1], c="red", marker="*", s=180, edgecolors="white", zorder=3)
            # Draw on 3D Surface
            gz = self.problem.evaluate(np.array([gx[0], gx[1]]))
            ax3d.scatter(gx[0], gx[1], gz, c="red", marker="*", s=300, edgecolors="black", linewidth=1, zorder=30)

            # Proxy Artists for Legends
            ax2d.plot([], [], marker="*", color="red", markeredgecolor="white", markersize=12, linestyle="None",
                      label="Global Minimum")
            ax3d.plot([], [], [], marker="*", color="red", markeredgecolor="black", markersize=10, linestyle="None",
                      label="Global Minimum")

        # Dynamic Markers Setup
        scat = ax2d.scatter([], [], c="orange", s=35, edgecolors="black", zorder=4)
        scat3d = ax3d.scatter([], [], [], c="orange", s=35, edgecolors="black", alpha=1.0, zorder=4)

        best_dot_2d, = ax2d.plot([], [], marker="D", color="lime", markersize=10, markeredgecolor="black",
                                 linestyle="None", zorder=5)
        best_dot_3d = ax3d.scatter([], [], [], c="lime", marker="D", s=100, edgecolors="black", alpha=1.0, zorder=10)

        # 2D Legend Setup
        ax2d.plot([], [], marker="o", color="orange", markeredgecolor="black", markersize=6, linestyle="None",
                  label="Swarm")
        ax2d.plot([], [], marker="D", color="lime", markeredgecolor="black", markersize=8, linestyle="None",
                  label="Algo Best")
        ax2d.legend(loc="upper right", fontsize=9, facecolor="white", framealpha=0.85, edgecolor="gray")

        # 3D Legend Setup
        ax3d.plot([], [], [], marker="o", color="orange", markeredgecolor="black", markersize=6, linestyle="None",
                  label="Swarm")
        ax3d.plot([], [], [], marker="D", color="lime", markeredgecolor="black", markersize=8, linestyle="None",
                  label="Algo Best")
        ax3d.legend(loc="upper left", fontsize=10, facecolor="white", framealpha=0.85, edgecolor="gray")

        # ===== CONVERGENCE CHART =====
        best_hist = self.metrics.get("best_fitness", [])
        print(best_hist)
        avg_hist = self.metrics.get("avg_fitness", [])

        conv_best, = ax_conv.plot([], [], "royalblue", lw=2.5, label="Best Fitness")

        # Only create the average line if average data actually exists!
        conv_avg = None
        if avg_hist:
            conv_avg, = ax_conv.plot([], [], "darkorange", lw=1.5, linestyle="--", label="Average")
            ax_conv.legend(loc="upper right")
        else:
            # If no average, label the best line clearly without a legend box
            ax_conv.set_title("Convergence Line Chart", fontsize=12, fontweight='bold')

        ax_conv.set_xlabel("num_iters")
        ax_conv.set_ylabel("Fitness")
        ax_conv.grid(True, linestyle=":", alpha=0.7)

        if best_hist:
            ax_conv.set_xlim(0, len(self.history))

            ymin = min(best_hist)
            ymax = max(best_hist)

            if avg_hist:
                ymin = min(ymin, min(avg_hist))
                ymax = max(ymax, max(avg_hist))

            if ymin == ymax:
                ymax += 1e-6

            ax_conv.set_ylim(ymin * 0.95, ymax * 1.05)

        fig.tight_layout(pad=2.0)

        # ===== UPDATE FUNCTION =====
        def update(frame):
            pop = self.history[frame]
            if isinstance(pop, tuple):
                pop = pop[0]
            pop = np.asarray(pop)

            if pop.ndim == 1:
                pop = pop.reshape(1, -1)

            # 1. Update Swarm in 2D
            scat.set_offsets(pop[:, :2])

            # 2. Evaluate fitness for all bees
            fitness = [self.problem.evaluate(ind) for ind in pop]

            # Update Swarm in 3D
            scat3d._offsets3d = (pop[:, 0], pop[:, 1], fitness)

            # 3. Find Current Best
            idx = np.argmin(fitness)
            best = pop[idx]
            bz = fitness[idx]

            # 3. Update Best Markers
            best_dot_2d.set_data([best[0]], [best[1]])
            z_bump = 0.2
            best_dot_3d._offsets3d = (np.array([best[0]]), np.array([best[1]]), np.array([bz + z_bump]))

            # Spin the 3D camera slightly every frame
            ax3d.view_init(elev=35, azim=frame * 3)

            # 4. Update Convergence Chart
            if frame < len(best_hist):
                x = np.arange(frame + 1)
                conv_best.set_data(x, best_hist[:frame + 1])

                if avg_hist and frame < len(avg_hist) and conv_avg:
                    conv_avg.set_data(x, avg_hist[:frame + 1])

            # 5. Update Styled Background Text Box
            ax_text.clear()
            ax_text.axis("off")

            stats_msg = (
                f"Generation : {frame+1} / {len(self.history)}\n"
                f"Best Score : {bz:.6f}\n"
                f"Coordinates: X={best[0]:.3f}, Y={best[1]:.3f}"
            )

            ax_text.text(0.5, 0.5, stats_msg, transform=ax_text.transAxes,
                         ha='center', va='center', fontsize=12, fontweight='bold', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=1.2', facecolor='#eef2f5', edgecolor='#ced4da', linewidth=2))

            if conv_avg:
                return scat, best_dot_2d, best_dot_3d, conv_best, conv_avg
            return scat, best_dot_2d, best_dot_3d, conv_best

        # ===== EXPORT =====
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=120,
            blit=False,
        )

        writer = animation.FFMpegWriter(fps=15, bitrate=2500)
        ani.save(self.save_path, writer=writer)

        plt.close()
        print("[VIS] Video saved successfully.")