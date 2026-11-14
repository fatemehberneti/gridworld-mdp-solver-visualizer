import random

def classic_grid():
    """Standard 3x4 Russell & Norvig Gridworld."""
    return [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, None, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

def random_wall_grid(rows, cols, wall_prob=0.2):
    """Generate a grid with random obstacles and fixed terminal states at top-right."""
    grid = []
    for _ in range(rows):
        row = [None if random.random() < wall_prob else 0.0 for _ in range(cols)]
        grid.append(row)

    grid[0][-1] = 1.0   # Positive terminal state
    grid[1][-1] = -1.0  # Negative terminal state
    return grid

def random_everything_grid(rows, cols, wall_prob=0.2, pos_prob=0.05, neg_prob=0.05, default_reward=0.0):
    """Generate a grid with random walls, positive terminals, and negative terminals."""
    grid = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            r = random.random()
            if r < wall_prob:
                row.append(None)
            elif r < wall_prob + pos_prob:
                row.append(1.0)
            elif r < wall_prob + pos_prob + neg_prob:
                row.append(-1.0)
            else:
                row.append(default_reward)
        grid.append(row)
    return grid