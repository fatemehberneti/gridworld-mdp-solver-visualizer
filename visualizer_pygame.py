import pygame
import time
from grids import classic_grid, random_everything_grid
from mdp_solvers import GridMDP, DIRECTIONS, MOVES

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (220, 40, 40)
GRAY = (120, 120, 120)
DARK_GRAY = (50, 50, 50)
BLUE = (40, 120, 220)


def get_cell_color(reward, value):
    """Determine cell background color based on terminal status or utility value."""
    if reward is None:
        return GRAY
    if reward == 1.0:
        return GREEN
    if reward == -1.0:
        return RED

    # Subtle green/red tint for non-terminal states based on utility value
    clamped = max(-1.0, min(1.0, value))
    if clamped >= 0:
        intensity = int(255 - clamped * 110)
        return (intensity, 255, intensity)
    else:
        intensity = int(255 - abs(clamped) * 110)
        return (255, intensity, intensity)


def draw_arrow(surface, center, direction, color=BLUE):
    """Draw a directional policy arrow inside a grid cell."""
    cx, cy = center
    offset = 24
    head = 7

    if direction == "^":
        start, end = (cx, cy + offset // 2), (cx, cy - offset)
        wing1, wing2 = (cx - head, cy - offset + head), (cx + head, cy - offset + head)
    elif direction == "v":
        start, end = (cx, cy - offset // 2), (cx, cy + offset)
        wing1, wing2 = (cx - head, cy + offset - head), (cx + head, cy + offset - head)
    elif direction == "<":
        start, end = (cx + offset // 2, cy), (cx - offset, cy)
        wing1, wing2 = (cx - offset + head, cy - head), (cx - offset + head, cy + head)
    elif direction == ">":
        start, end = (cx - offset // 2, cy), (cx + offset, cy)
        wing1, wing2 = (cx + offset - head, cy - head), (cx + offset - head, cy + head)
    else:
        return

    pygame.draw.line(surface, color, start, end, 3)
    pygame.draw.polygon(surface, color, [end, wing1, wing2])


def visualize_mdp(rewards, algorithm="value", delay=0.8):
    pygame.init()
    rows, cols = len(rewards), len(rewards[0])
    cell_size = 100
    status_height = 50
    width, height = cols * cell_size, rows * cell_size + status_height
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"MDP {algorithm.capitalize()} Iteration Visualizer")

    def init_state(grid_rewards):
        mdp_inst = GridMDP(grid_rewards, discount_factor=0.9, noise=0.2)
        vals = [
            [r if r in mdp_inst.terminals else (0.0 if r is not None else None) for r in row]
            for row in grid_rewards
        ]
        pol = [
            ["^" if (r is not None and r not in mdp_inst.terminals) else "" for r in row]
            for row in grid_rewards
        ]
        return mdp_inst, vals, pol, 0, False

    mdp, values, policy, iteration, converged = init_state(rewards)

    font = pygame.font.Font(None, 24)
    status_font = pygame.font.Font(None, 26)
    paused = False
    running = True
    last_step_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    mdp, values, policy, iteration, converged = init_state(rewards)

        now = time.time()
        if not converged and not paused and (now - last_step_time >= delay):
            last_step_time = now

            if algorithm == "value":
                new_values = [row[:] for row in values]
                delta = 0.0
                for x in range(rows):
                    for y in range(cols):
                        if rewards[x][y] is None or rewards[x][y] in mdp.terminals:
                            continue
                        utils = [mdp.expected_utility(values, x, y, dx, dy) for dx, dy in MOVES]
                        best_val = max(utils)
                        new_values[x][y] = best_val
                        policy[x][y] = DIRECTIONS[utils.index(best_val)]
                        delta = max(delta, abs(values[x][y] - best_val))
                values = new_values
                iteration += 1
                if delta < 1e-3:
                    converged = True
            else:
                # 1. Policy Evaluation
                while True:
                    new_values = [row[:] for row in values]
                    delta = 0.0
                    for x in range(rows):
                        for y in range(cols):
                            if rewards[x][y] is None or rewards[x][y] in mdp.terminals:
                                continue
                            dx, dy = MOVES[DIRECTIONS.index(policy[x][y])]
                            val = mdp.expected_utility(values, x, y, dx, dy)
                            new_values[x][y] = val
                            delta = max(delta, abs(new_values[x][y] - values[x][y]))
                    values = new_values
                    if delta < 1e-3:
                        break

                # 2. Policy Improvement
                stable = True
                for x in range(rows):
                    for y in range(cols):
                        if rewards[x][y] is None or rewards[x][y] in mdp.terminals:
                            continue
                        utils = [mdp.expected_utility(values, x, y, dx, dy) for dx, dy in MOVES]
                        best_act = DIRECTIONS[utils.index(max(utils))]
                        if best_act != policy[x][y]:
                            stable = False
                        policy[x][y] = best_act

                iteration += 1
                if stable:
                    converged = True

        # --- Render Grid ---
        screen.fill(WHITE)
        for x in range(rows):
            for y in range(cols):
                rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
                reward = rewards[x]