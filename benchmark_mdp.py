import time
import random
import matplotlib.pyplot as plt
from grids import random_wall_grid
from mdp_solvers import GridMDP

# --- Experiment 1: Runtime Scalability (Value Iteration vs. Policy Iteration) ---
grid_sizes = [10, 25, 50, 75, 100]
vi_times = []
pi_times = []

for size in grid_sizes:
    vi_sum = 0.0
    pi_sum = 0.0
    runs = 5
    for _ in range(runs):
        grid = random_wall_grid(size, size, wall_prob=0.15)
        mdp = GridMDP(grid, discount_factor=0.9, noise=0.2)

        t0 = time.time()
        mdp.value_iteration(theta=1e-3)
        vi_sum += time.time() - t0

        t0 = time.time()
        mdp.policy_iteration(theta=1e-3)
        pi_sum += time.time() - t0

    vi_times.append(vi_sum / runs)
    pi_times.append(pi_sum / runs)

plt.figure(figsize=(9, 5))
plt.plot(grid_sizes, vi_times, label="Value Iteration", color="purple", marker="o")
plt.plot(grid_sizes, pi_times, label="Policy Iteration", color="teal", marker="s", linestyle="--")
plt.title("MDP Convergence Runtime: Value Iteration vs. Policy Iteration")
plt.xlabel("Grid Dimension (N x N)")
plt.ylabel("Average Convergence Time (seconds)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("mdp_runtime_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Experiment 2: Theoretical V(s) vs. Empirical Monte Carlo Rollouts ---
small_grid = random_wall_grid(5, 5, wall_prob=0.1)
mdp_small = GridMDP(small_grid, discount_factor=0.9, noise=0.2, living_reward=0.0)
values, policy, _ = mdp_small.value_iteration(theta=1e-4)

valid_cells = [
    (r, c) for r in range(5) for c in range(5)
    if small_grid[r][c] not in (None, 1.0, -1.0)
]
sample_states = random.sample(valid_cells, min(6, len(valid_cells)))

labels = []
mc_averages = []
theoretical_vs = []

for (sx, sy) in sample_states:
    rewards_list = [mdp_small.simulate_episode(sx, sy, policy) for _ in range(300)]
    avg_mc = sum(rewards_list) / len(rewards_list)
    v_s = values[sx][sy]

    labels.append(f"({sx},{sy})")
    mc_averages.append(avg_mc)
    theoretical_vs.append(v_s)

x_idx = range(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar([i - width/2 for i in x_idx], mc_averages, width=width, label="Empirical Avg Return (300 MC Rollouts)", color="coral")
plt.bar([i + width/2 for i in x_idx], theoretical_vs, width=width, label="Theoretical V(s) (Value Iteration)", color="slateblue")
plt.xticks(list(x_idx), labels)
plt.xlabel("Agent Start State (row, col)")
plt.ylabel("Expected Discounted Return")
plt.title("Empirical Monte Carlo Rollout Return vs. Theoretical V(s) (Noise = 0.2)")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.savefig("mc_vs_vi_validation.png", dpi=300, bbox_inches="tight")
plt.show()