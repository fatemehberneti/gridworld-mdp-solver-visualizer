# Gridworld MDP Solver & Visualizer

A Python implementation of **Markov Decision Process (MDP)** planning algorithms in stochastic Gridworld environments[cite: 2, 3, 4]. This project includes exact dynamic programming solvers (**Value Iteration** and **Policy Iteration**)[cite: 2], empirical **Monte Carlo rollout** validation[cite: 1, 2], scalability benchmarks, and an interactive **Pygame** visualizer.

## Features

* **Exact MDP Solvers (`mdp_solvers.py`)**[cite: 2]:
  * **Value Iteration** & **Policy Iteration** supporting customizable discount factors ($\gamma$), transition noise, and living rewards[cite: 2].
  * Stochastic transition model where intended actions succeed with probability $1 - \text{noise}$ and slip perpendicularly with probability $\text{noise}/2$ each[cite: 2].
  * **Monte Carlo Episode Simulator** to empirically verify theoretical state utilities $V(s)$[cite: 1, 2].
* **Interactive Visualizer (`visualizer_pygame.py`)**[cite: 3]:
  * Real-time Pygame rendering of Value Iteration and Policy Iteration convergence[cite: 3].
  * Color-coded utility heatmaps, terminal states (`+1.0` / `-1.0`), obstacles, and optimal policy directional arrows[cite: 2, 3].
  * Interactive playback controls (`SPACE` to pause/resume, `R` to reset).
* **Grid Environments (`grids.py`)**[cite: 4]:
  * Classic $3 \times 4$ Russell & Norvig Gridworld[cite: 4].
  * Procedural grid generators with customizable wall, positive terminal, and negative terminal probabilities[cite: 4].
* **Benchmarks & Empirical Validation (`benchmark_mdp.py`)**:
  * Compares convergence runtime between Value Iteration and Policy Iteration across grid sizes from $10 \times 10$ to $100 \times 100$.
  * Validates theoretical $V(s)$ values against 300-episode Monte Carlo rollouts.

## Project Structure

```text
├── grids.py                 # Classic 3x4 and procedural random Gridworld generators
├── mdp_solvers.py           # GridMDP class (Value Iteration, Policy Iteration, MC rollouts)
├── visualizer_pygame.py     # Real-time Pygame visualization of MDP convergence
├── benchmark_mdp.py         # Runtime scalability & Monte Carlo validation plots
└── README.md
