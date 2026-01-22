# Project1-GroupNotAI

##Folder structure
```
Group_3_NguTuNhien/
├── data/                   # Input files
├── output/                 # Saved plots and logs
├── src/
│   ├── algorithms/        
│   │   ├── __init__.py
│   │   ├── base_algorithm.py   # Abstract Base Class 
│   │   ├── classical/
│   │   │   ├── bfs.py
│   │   │   ├── dfs.py
│   │   │   └── a_star.py
│   │   └── nature/
│   │       ├── ga.py       # Genetic Algorithm
│   │       ├── pso.py      # Particle Swarm
│   │       └── sa.py       # Simulated Annealing
│   │
│   ├── problems/          
│   │   ├── __init__.py
│   │   ├── base_problem.py # Abstract Base Class 
│   │   ├── continuous.py   
│   │   └── discrete.py    
│   │
│   ├── utils/
│   │   └── logger.py       # Collecting stats during runs
│   │
│   └── visualization/     
│       └── __init__.py
│
├── main.py                 # The entry point where you link them
└── requirements.txt        
```