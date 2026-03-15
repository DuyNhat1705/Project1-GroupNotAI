# Search & Nature-Inspired Algorithms — CSC14003 Course Project

> **Course:** CSC14003 – Fundamentals of Artificial Intelligence  
> **Faculty:** Faculty of Information Technology, University of Science, VNU-HCM  
> **Theme:** Search & Nature-Inspired Algorithms

---

## Highlights

- Modular framework implementing classical search and nature-inspired optimization algorithms across multiple categories.
- Benchmark problems spanning both continuous optimization functions and discrete combinatorial problems.
- Three main workflows: single-run execution via `main.py`, multi-run benchmarking via `benchmark.py`, and parameter sensitivity analysis via `para_sen.py`.
- Animated visualization for supported algorithm–problem combinations.
- Automated benchmark reporting with exported statistics and charts.
- Parameter sensitivity analysis with exported figure and tabular outputs for supported algorithms and problems.

---

## Table of Contents

1. [Course Information](#course-information)
2. [Project Overview](#project-overview)
3. [Objectives](#objectives)
4. [Implemented Algorithms](#implemented-algorithms)
5. [Benchmark Problems](#benchmark-problems)
6. [Repository Structure](#repository-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Usage Examples](#usage-examples)
10. [Visualization & Analysis Features](#visualization--analysis-features)
11. [Output / Results](#output--results)
12. [Team Members](#team-members)
13. [References](#references)
14. [Limitations / Notes](#limitations--notes)

---

## Course Information

| Field | Detail |
|---|---|
| Course Code | CSC14003 |
| Course Name | Fundamentals of Artificial Intelligence |
| Faculty | Faculty of Information Technology |
| University | University of Science, VNU-HCM |
| Project Theme | Search & Nature-Inspired Algorithms |

---

## Project Overview

This repository contains a modular Python framework for implementing and comparing classical search algorithms with nature-inspired metaheuristic optimization algorithms. The framework provides a unified interface for running supported algorithms on compatible problems, generating visualizations, and collecting benchmark statistics across multiple runs.

---

## Objectives

- Implement and study a diverse set of search and optimization algorithms.
- Evaluate algorithm performance on both **continuous** benchmark functions and **discrete** combinatorial problems.
- Provide reproducible multi-run benchmarks with statistical reporting.
- Analyze the sensitivity of algorithm performance to key hyperparameters.
- Produce clear visualizations suitable for academic presentation and reporting.

---

## Implemented Algorithms in This Repository

The following list includes algorithms currently implemented in the repository and documented in the project report.

### Classical Search

| Short Key | Algorithm | Supported Problems |
|---|---|---|
| `bfs` | Breadth-First Search | Shortest Path on Graph, Shortest Path on Maze, Knapsack, Graph Coloring |
| `dfs` | Depth-First Search | Shortest Path on Graph, Shortest Path on Maze, Knapsack, Graph Coloring |
| `astar` | A* Search | Shortest Path on Maze, TSP |

### Local Search / Physics-based

| Short Key | Algorithm | Supported Problems |
|---|---|---|
| `hc` | Hill Climbing | Continuous, TSP, Graph Coloring |
| `sa` | Simulated Annealing | Continuous, TSP, Graph Coloring |

### Evolutionary Algorithms

| Short Key | Algorithm | Supported Problems |
|---|---|---|
| `ga` | Genetic Algorithm | Continuous, TSP, Shortest Path on Maze |
| `de` | Differential Evolution | Continuous |

### Swarm Intelligence / Biology-based

| Short Key | Algorithm | Supported Problems |
|---|---|---|
| `pso` | Particle Swarm Optimization | Continuous, TSP |
| `aco` | Ant Colony Optimization | Continuous, TSP |
| `abc` | Artificial Bee Colony | Continuous, Knapsack |
| `fa` | Firefly Algorithm | Continuous, TSP |
| `cs` | Cuckoo Search | Continuous, Knapsack |

### Human-based

| Short Key | Algorithm | Supported Problems |
|---|---|---|
| `tlbo` | Teaching–Learning-Based Optimization (TLBO) | Continuous, Knapsack |

---

## Benchmark Problems

### Continuous Optimization Functions

| Problem Key | Name | Search Domain | Global Minimum |
|---|---|---|---|
| `sphere` | Sphere | `[-5.12, 5.12]` | 0.0 |
| `rosenbrock` | Rosenbrock | `[-10, 10]` | 0.0 |
| `ackley` | Ackley | `[-32.768, 32.768]` | 0.0 |
| `griewank` | Griewank | `[-50, 50]` | 0.0 |
| `rastrigin` | Rastrigin | `[-5.12, 5.12]` | 0.0 |
| `michalewicz` | Michalewicz | `[0, π]` | -1.8013 (2D) |

Dimension is configurable via `--dim`. Default is 2.

### Discrete / Combinatorial Problems

| Problem Key | Name | Input File | Known Optimum |
|---|---|---|---|
| `tsp1` | Travelling Salesman Problem | `data/tsp1.txt` | 23535.0 |
| `tsp2` | Travelling Salesman Problem | `data/tsp2.txt` | 1272.0 |
| `tsp3` | Travelling Salesman Problem | `data/tsp3.txt` | 33523.0 |
| `tsp4` | Travelling Salesman Problem | `data/tsp4.txt` | 417.6 |
| `maze1` | Shortest Path on Maze | `data/maze1.txt` | — |
| `maze2` | Shortest Path on Maze | `data/maze2.txt` | — |
| `graph1` | Shortest Path on Graph | `data/graph1.txt` | — |
| `knapsack1` | 0/1 Knapsack | `data/knapsack1.txt` | 483.0 |
| `knapsack2` | 0/1 Knapsack | `data/knapsack2.txt` | 47719.0 |
| `knapsack3` | 0/1 Knapsack | `data/knapsack3.txt` | 504948.0 |
| `coloring1` | Graph Coloring | `data/coloring1.txt` | 3 colors |
| `coloring2` | Graph Coloring | `data/coloring2.txt` | 3 colors |
| `coloring3` | Graph Coloring | `data/coloring3.txt` | 2 colors |

---

## Repository Structure

```text
Project1-GroupNotAI/
├── main.py                         # Main entry point (single run)
├── benchmark.py                    # Multi-run benchmark & report generator
├── para_sen.py                     # Parameter sensitivity analysis
├── requirements.txt                # Python dependencies
├── run_program.txt                 # Reference command examples
│
├── data/                           # Input problem files
│   ├── tsp1.txt ... tsp4.txt
│   ├── maze1.txt, maze2.txt
│   ├── graph1.txt
│   ├── knapsack1.txt ... knapsack3.txt
│   └── coloring1.txt ... coloring3.txt
│
├── output/                         # Generated results (auto-created)
│   ├── <problem>/
│   │   ├── <problem>_report.json
│   │   ├── <problem>_convergence.pdf
│   │   ├── <problem>_robustness.pdf
│   │   └── <problem>_complexity.pdf
│   ├── sensitivity_analysis/
│   │   ├── <problem>_Sensitivity_Analysis.pdf
│   │   └── <problem>_Sensitivity_Analysis.csv
│   └── *.mp4
│
└── src/
    ├── HandleCLI.py
    ├── algorithms/
    │   ├── base_algorithm.py
    │   ├── algorithms_factory.py
    │   ├── classical/
    │   │   ├── bfs.py
    │   │   ├── dfs.py
    │   │   ├── a_star.py
    │   │   └── hill_climbing.py
    │   ├── physics/
    │   │   └── simulated_annealing.py
    │   ├── evolution/
    │   │   ├── genetic_algorithm.py
    │   │   └── differential_evolution.py
    │   ├── biology/
    │   │   ├── particle_swarm.py
    │   │   ├── ant_colony_optimization.py
    │   │   ├── artificial_bee.py
    │   │   ├── firefly_algorithm.py
    │   │   └── cuckoo_search.py
    │   └── human/
    │       └── tlbo.py
    ├── problems/
    │   ├── base_problem.py
    │   ├── problems_factory.py
    │   ├── continuous/
    │   │   └── continuous.py
    │   └── discrete/
    │       ├── TSP.py
    │       ├── ShortestPathOnMaze.py
    │       ├── ShortestPathOnGraph.py
    │       ├── knapsack.py
    │       └── GraphColoring.py
    ├── utils/
    │   └── logger.py
    └── visualization/
        ├── base_visualizer.py
        ├── visualizer_factory.py
        ├── continuous_visualizer.py
        ├── maze_visualizer.py
        ├── maze_ga_visualizer.py
        ├── graph_visualizer.py
        ├── graph_color_viz.py
        ├── TSP_viz.py
        ├── TSP_GA_viz.py
        └── knapsack_viz.py
```

---

## Installation

**Python version:** Python 3.10 or later is recommended.

### 1. Clone the repository

```bash
git clone <repository-url>
cd Project1-GroupNotAI
```

### 2. Create and activate a virtual environment

**Windows**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include `numpy` and `matplotlib`.

### 4. Additional system requirement

For exporting MP4 animations, `ffmpeg` may be required depending on your `matplotlib` animation backend and local environment.

- **Windows:** install FFmpeg and add it to your system `PATH`
- **Linux:** install via your package manager, for example `sudo apt install ffmpeg`
- **macOS:** install via Homebrew, for example `brew install ffmpeg`

---

## Usage

All workflows are run from the project root directory.

### Single Run — `main.py`

```bash
python main.py --algo <ALGO> --problem <PROBLEM> [--dim <DIM>] [--seed <SEED>] [--params key=value ...]
```

| Argument | Description | Default |
|---|---|---|
| `--algo` | Algorithm key such as `ga`, `bfs`, or `sa` | Required |
| `--problem` | Problem key such as `sphere`, `tsp1`, or `maze2` | Required |
| `--dim` | Dimension for continuous problems | `2` |
| `--seed` | Random seed for reproducibility | Depends on script configuration |
| `--params` | Algorithm hyperparameters as `key=value` pairs | — |

For supported visualizers, an MP4 animation is saved to the `output/` directory.

### Multi-Run Benchmark — `benchmark.py`

```bash
python benchmark.py --problem <PROBLEM> --runs <N> [--dim <DIM>] [--params key=value ...]
```

| Argument | Description | Default |
|---|---|---|
| `--problem` | Problem key | Required |
| `--runs` | Number of independent runs per algorithm | Depends on script configuration |
| `--dim` | Dimension for continuous problems | Depends on script configuration |
| `--params` | Shared algorithm hyperparameters | — |

Outputs are saved to `output/<problem>/`.

### Parameter Sensitivity Analysis — `para_sen.py`

```bash
python para_sen.py
```

`para_sen.py` currently runs predefined test cases from the `test_cases` list inside the script.
To analyze a specific problem, edit `test_cases` in `para_sen.py` before running.

Outputs are saved to `output/sensitivity_analysis/`.

Expected output files are:
- `output/sensitivity_analysis/<problem>_Sensitivity_Analysis.pdf`
- `output/sensitivity_analysis/<problem>_Sensitivity_Analysis.csv`

With the current default `test_cases` in `para_sen.py`, typical outputs include:
- `output/sensitivity_analysis/michalewicz_Sensitivity_Analysis.pdf`
- `output/sensitivity_analysis/michalewicz_Sensitivity_Analysis.csv`
- `output/sensitivity_analysis/tsp1_Sensitivity_Analysis.pdf`
- `output/sensitivity_analysis/knapsack1_Sensitivity_Analysis.pdf`

---

## Usage Examples

### Discrete problems

```bash
python main.py --algo BFS --problem graph1
python main.py --algo DFS --problem graph1

python main.py --algo astar --problem maze1
python main.py --algo astar --problem maze2
python main.py --algo ga --problem maze1

python main.py --algo sa --problem tsp1
python main.py --algo hc --problem tsp1
python main.py --algo ga --problem tsp1
python main.py --algo astar --problem tsp1
python main.py --algo aco --problem tsp2 --params pop_size=30 num_iters=500

python main.py --algo tlbo --problem knapsack1 --params pop_size=30 num_iters=100
python main.py --algo cs --problem knapsack1 --params pop_size=25 num_iters=100
python main.py --algo abc --problem knapsack1
python main.py --algo bfs --problem knapsack2
```

### Continuous problems

```bash
python main.py --algo ga --problem sphere --params pop_size=500
python main.py --algo de --problem sphere --params pop_size=50 num_iters=100
python main.py --algo tlbo --problem sphere --params pop_size=50 num_iters=90
python main.py --algo cs --problem sphere --params pop_size=50 num_iters=100
python main.py --algo sa --problem sphere
python main.py --algo hc --problem sphere
python main.py --algo abc --problem griewank
python main.py --algo pso --problem michalewicz --params pop_size=50 num_iters=200
python main.py --algo fa --problem rastrigin --dim 10
python main.py --algo aco --problem ackley --dim 10
```

### Benchmarking

```bash
python benchmark.py --problem sphere --runs 10 --dim 30 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995
python benchmark.py --problem ackley --runs 10 --dim 10 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995
python benchmark.py --problem rosenbrock --runs 10 --dim 30 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995
python benchmark.py --problem rastrigin --runs 10 --dim 20 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995
python benchmark.py --problem griewank --runs 10 --dim 20 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995
python benchmark.py --problem michalewicz --runs 10 --dim 10 --params pop_size=30 num_iters=500 step=1.0 temperature=100 decay=0.995

python benchmark.py --problem tsp2 --runs 10 --params pop_size=30 num_iters=500
python benchmark.py --problem knapsack2 --runs 10 --params pop_size=30 num_iters=500
python benchmark.py --problem coloring2 --runs 10 --params pop_size=30 num_iters=500
python benchmark.py --problem maze2 --runs 10 --params pop_size=30 num_iters=100
```

### Parameter sensitivity analysis

```bash
python para_sen.py
```

Then adjust `test_cases` in `para_sen.py` to control which problems are analyzed.


---

## Visualization & Analysis Features

### Single-run animation (`main.py`)
- **Continuous problems:** animated surface/contour visualization with convergence tracking for supported algorithms.
- **TSP:** animated tour construction or tour improvement visualization.
- **Maze:** step-by-step exploration for supported search algorithms and evolutionary visualization for GA where available.
- **Graph:** exploration animation for supported graph-search workflows.
- **Knapsack:** item-selection history visualization where supported.
- **Graph Coloring:** node-coloring progression where supported.

### Benchmark reports (`benchmark.py`)
For supported problems, the benchmark workflow can export files such as:

| File | Content |
|---|---|
| `<problem>_report.json` | Per-algorithm statistics such as best, worst, mean, median, standard deviation, runtime, and memory usage |
| `<problem>_convergence.pdf` | Convergence curves for compatible algorithms |
| `<problem>_robustness.pdf` | Distribution or boxplot-style robustness comparison |
| `<problem>_complexity.pdf` | Time and memory comparison plots |

### Parameter sensitivity analysis (`para_sen.py`)
Depending on the number of tunable parameters:
- one-parameter analysis may be shown as a line plot
- two-parameter analysis may be shown as a heatmap or contour-style figure

Current script coverage is based on the internal compatibility map in `para_sen.py`.
At present, `graphcoloring` and `shortestpathongraph` are not included in sensitivity runs.

Saved outputs may include:
- `<problem>_Sensitivity_Analysis.pdf`
- `<problem>_Sensitivity_Analysis.csv`

---

## Output / Results

Depending on the current repository state, the `output/` directory may contain generated files such as:
- MP4 animations
- JSON benchmark summaries
- PDF convergence, robustness, and complexity plots
- PDF/CSV sensitivity analysis results

Sensitivity analysis outputs are written to:
- `output/sensitivity_analysis/`

---

## Team Members

| # | Full Name | Student ID |
|---|---|---|
| 1 | Nguyễn Lê Hoàng Khải | 24127408 |
| 2 | Vũ Duy Nhất | 24127095 |
| 3 | Trần Lê Hoàng Gia | 24127028 |
| 4 | Phan Lê Anh Minh | 24127082 |

---

## References

- Dorigo, M., Birattari, M., & Stützle, T. (2007). Ant colony optimization. *IEEE Computational Intelligence Magazine*, 1(4), 28-39.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN*.
- Karaboga, D., & Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. *Journal of Global Optimization*, 39(3), 459-471.
- Yang, X. S., & He, X. (2013). Firefly algorithm: recent advances and applications. *International Journal of Swarm Intelligence*, 1(1), 36-50.
- Yang, X. S., & Deb, S. (2014). Cuckoo search: recent advances and applications. *Neural Computing and Applications*, 24(1), 169-174.

---

## Limitations / Notes

- All interaction is command-line based.
- The repository currently relies on `requirements.txt`; no conda environment file is documented here.
- Python 3.10+ is recommended for compatibility.
- MP4 export may require a working `matplotlib` animation backend and `ffmpeg` on some systems.
- Benchmark and sensitivity analysis coverage depends on algorithm–problem compatibility.
