import random

DIRECTIONS = ["^", "v", "<", ">"]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class GridMDP:
    """Markov Decision Process (MDP) solver supporting Value Iteration, Policy Iteration, and Monte Carlo rollouts."""

    def __init__(self, rewards, discount_factor=0.9, noise=0.2, living_reward=0.0):
        self.rewards = rewards
        self.rows = len(rewards)
        self.cols = len(rewards[0])
        self.gamma = discount_factor
        self.noise = noise
        self.living_reward = living_reward
        self.terminals = {1.0, -1.0}

    def is_available(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.rewards[x][y] is not None

    def expected_utility(self, values, x, y, dx, dy):
        """Calculate expected utility of taking action (dx, dy) from (x, y) under stochastic transition noise."""
        main_x, main_y = x + dx, y + dy
        side1_x, side1_y = x + dy, y - dx
        side2_x, side2_y = x - dy, y + dx

        main_val = values[main_x][main_y] if self.is_available(main_x, main_y) else values[x][y]
        side1_val = values[side1_x][side1_y] if self.is_available(side1_x, side1_y) else values[x][y]
        side2_val = values[side2_x][side2_y] if self.is_available(side2_x, side2_y) else values[x][y]

        expected_next = (
            (1.0 - self.noise) * main_val
            + (self.noise / 2.0) * side1_val
            + (self.noise / 2.0) * side2_val
        )
        return self.living_reward + self.gamma * expected_next

    def value_iteration(self, theta=1e-3, max_iters=1000):
        """Run Value Iteration until convergence. Returns (values, policy, iterations)."""
        values = [
            [r if r in self.terminals else (0.0 if r is not None else None) for r in row]
            for row in self.rewards
        ]
        policy = [["" for _ in range(self.cols)] for _ in range(self.rows)]

        iterations = 0
        while iterations < max_iters:
            new_values = [row[:] for row in values]
            delta = 0.0

            for x in range(self.rows):
                for y in range(self.cols):
                    if self.rewards[x][y] is None or self.rewards[x][y] in self.terminals:
                        continue

                    action_utils = [self.expected_utility(values, x, y, dx, dy) for dx, dy in MOVES]
                    best_val = max(action_utils)
                    best_act = DIRECTIONS[action_utils.index(best_val)]

                    new_values[x][y] = best_val
                    policy[x][y] = best_act
                    delta = max(delta, abs(values[x][y] - best_val))

            values = new_values
            iterations += 1
            if delta < theta:
                break

        return values, policy, iterations

    def policy_iteration(self, theta=1e-3, max_iters=500):
        """Run Policy Iteration (Evaluation + Improvement) until policy stabilizes. Returns (values, policy, iterations)."""
        values = [
            [r if r in self.terminals else (0.0 if r is not None else None) for r in row]
            for row in self.rewards
        ]
        policy = [
            ["^" if (r is not None and r not in self.terminals) else "" for r in row]
            for row in self.rewards
        ]

        iterations = 0
        while iterations < max_iters:
            # 1. Policy Evaluation
            while True:
                new_values = [row[:] for row in values]
                delta = 0.0
                for x in range(self.rows):
                    for y in range(self.cols):
                        if self.rewards[x][y] is None or self.rewards[x][y] in self.terminals:
                            continue
                        dx, dy = MOVES[DIRECTIONS.index(policy[x][y])]
                        val = self.expected_utility(values, x, y, dx, dy)
                        new_values[x][y] = val
                        delta = max(delta, abs(new_values[x][y] - values[x][y]))
                values = new_values
                if delta < theta:
                    break

            # 2. Policy Improvement
            policy_stable = True
            for x in range(self.rows):
                for y in range(self.cols):
                    if self.rewards[x][y] is None or self.rewards[x][y] in self.terminals:
                        continue
                    action_utils = [self.expected_utility(values, x, y, dx, dy) for dx, dy in MOVES]
                    best_act = DIRECTIONS[action_utils.index(max(action_utils))]
                    if best_act != policy[x][y]:
                        policy_stable = False
                    policy[x][y] = best_act

            iterations += 1
            if policy_stable:
                break

        return values, policy, iterations

    def simulate_episode(self, start_x, start_y, policy, max_steps=500):
        """Simulate a single stochastic rollout following a given policy."""
        x, y = start_x, start_y
        total_discounted_reward = 0.0
        steps = 0

        while self.rewards[x][y] not in self.terminals and steps < max_steps:
            action = policy[x][y]
            if not action:
                break
            dx, dy = MOVES[DIRECTIONS.index(action)]

            r = random.random()
            if r < (1.0 - self.noise):
                nx, ny = x + dx, y + dy
            elif r < (1.0 - self.noise / 2.0):
                nx, ny = x + dy, y - dx
            else:
                nx, ny = x - dy, y + dx

            if not self.is_available(nx, ny):
                nx, ny = x, y

            x, y = nx, ny
            if self.rewards[x][y] in self.terminals:
                total_discounted_reward += (self.gamma ** (steps + 1)) * self.rewards[x][y]
            else:
                total_discounted_reward += (self.gamma ** steps) * self.living_reward
            steps += 1

        return total_discounted_reward