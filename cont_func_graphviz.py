import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.problems.continuous.continuous import Sphere, Rosenbrock, Griewank, Ackley, Rastrigin, Michalewicz


def plot_3d_surface(problem):
    print(f"Generating 3D Visualization for {problem.name}...")

    resolution = 201
    x = np.linspace(problem.min_range, problem.max_range, resolution)
    y = np.linspace(problem.min_range, problem.max_range, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = problem.evaluate(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    #  alpha = 0.70 so the extreme point is visible
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.70)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="f(x,y)")

    if hasattr(problem, 'global_x') and problem.global_x is not None:
        gx, gy = problem.global_x[0], problem.global_x[1]
        gz = problem.evaluate(np.array([gx, gy]))

        # Ensure the star is drawn on top
        ax.scatter(gx, gy, gz, color='red', s=350, marker='*', zorder=20,
                   edgecolors='black', label='Global Minimum')

        z_offset = (np.max(Z) - np.min(Z)) * 0.08
        ax.text(gx, gy, gz + z_offset,
                f' Min: ({gx:.2f}, {gy:.2f}, {gz:.2f})',
                color='darkred', fontsize=12, fontweight='bold', zorder=25)

    ax.set_title(f"3D Surface of {problem.name} Function", fontsize=16, fontweight='bold')
    ax.set_xlabel("x", fontweight='bold')
    ax.set_ylabel("y", fontweight='bold')
    ax.set_zlabel("z", fontweight='bold')

    # --- Custom Viewing Angles ---
    # format: "Name": (elevation, azimuth)
    view_angles = {
         "Sphere": (40, 60),
         "Rosenbrock": (30, 110),
         "Griewank": (10, 60),
         "Ackley": (25, 60),
         "Rastrigin": (10, 80),
         "Michalewicz": (20, 60)
    }

    # Get the custom angle / default to (30, 45) if not found
    elev, azim = view_angles.get(problem.name, (0, 90))
    ax.view_init(elev=elev, azim=azim)

    ax.legend(loc='upper right')

    os.makedirs("output", exist_ok=True)
    filename = f"output/{problem.name}_3D.svg"
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

    print(f" -> Saved successfully: {filename}")


if __name__ == "__main__":
    functions_to_plot = [
        Sphere(dimension=2),
        Rosenbrock(dimension=2),
        Griewank(dimension=2),
        Ackley(dimension=2),
        Rastrigin(dimension=2),
        Michalewicz(dimension=2)
    ]

    print("--- Starting 3D Function Visualizer ---")

    for func in functions_to_plot:
        plot_3d_surface(func)

    print("\nAll 3D SVGs generated in the 'output/' folder!")