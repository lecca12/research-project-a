import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SimpleGridWorld(gym.Env):
    """
    GridWorld with:
    - agent
    - goal
    - obstacles
    - guaranteed reachable layouts
    - BFS shortest path solver
    - orientation-aware agent for egocentric prompts

    Absolute actions:
        0 = up/north
        1 = right/east
        2 = down/south
        3 = left/west
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=5, obstacle_density=0.0, max_steps=50, ensure_path=True):
        super().__init__()

        self.size = size
        self.obstacle_density = obstacle_density
        self.max_steps = max_steps
        self.ensure_path = ensure_path

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            "goal": spaces.Box(low=0, high=size - 1, shape=(2,), dtype=np.int32),
            "obstacles": spaces.Box(low=0, high=1, shape=(size, size), dtype=np.int32),
            "facing": spaces.Discrete(4),  # 0=N, 1=E, 2=S, 3=W
        })

        self.action_to_direction = {
            0: np.array([-1, 0], dtype=np.int32),  # north
            1: np.array([0, 1], dtype=np.int32),   # east
            2: np.array([1, 0], dtype=np.int32),   # south
            3: np.array([0, -1], dtype=np.int32),  # west
        }

        self.action_to_name = {
            0: "north",
            1: "east",
            2: "south",
            3: "west",
        }

        self.facing_to_name = {
            0: "north",
            1: "east",
            2: "south",
            3: "west",
        }

        self.relative_action_to_name = {
            0: "forward",
            1: "right",
            2: "backward",
            3: "left",
        }

        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.agent_facing = None
        self.current_step = 0

    def _get_obs(self):
        return {
            "agent": self.agent_pos.copy(),
            "goal": self.goal_pos.copy(),
            "obstacles": self.obstacles.copy(),
            "facing": int(self.agent_facing),
        }

    def _get_info(self):
        manhattan_distance = int(np.abs(self.agent_pos - self.goal_pos).sum())
        return {
            "manhattan_distance": manhattan_distance,
            "facing_name": self.facing_to_name[self.agent_facing],
        }

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size

    def _is_obstacle(self, pos):
        r, c = pos
        return self.obstacles[r, c] == 1

    def _is_free(self, pos):
        return self._in_bounds(pos) and not self._is_obstacle(pos)

    def _random_empty_cell(self):
        while True:
            pos = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
            if not self._is_obstacle(pos):
                return pos

    def _place_obstacles(self):
        self.obstacles = np.zeros((self.size, self.size), dtype=np.int32)

        total_cells = self.size * self.size
        num_obstacles = int(total_cells * self.obstacle_density)

        if num_obstacles == 0:
            return

        all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        chosen_indices = self.np_random.choice(len(all_positions), size=num_obstacles, replace=False)

        for idx in chosen_indices:
            r, c = all_positions[idx]
            self.obstacles[r, c] = 1

    def _neighbors(self, pos):
        neighbors = []
        for action, direction in self.action_to_direction.items():
            next_pos = pos + direction
            if self._is_free(next_pos):
                neighbors.append((action, next_pos.astype(np.int32)))
        return neighbors

    def _bfs_path(self, start, goal):
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)

        queue = deque([start_tuple])
        came_from = {start_tuple: None}

        while queue:
            current = queue.popleft()

            if current == goal_tuple:
                break

            current_arr = np.array(current, dtype=np.int32)

            for _, neighbor in self._neighbors(current_arr):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in came_from:
                    came_from[neighbor_tuple] = current
                    queue.append(neighbor_tuple)

        if goal_tuple not in came_from:
            return None

        path = []
        current = goal_tuple
        while current is not None:
            path.append(np.array(current, dtype=np.int32))
            current = came_from[current]

        path.reverse()
        return path

    def has_path(self):
        return self._bfs_path(self.agent_pos, self.goal_pos) is not None

    def get_shortest_path(self):
        return self._bfs_path(self.agent_pos, self.goal_pos)

    def get_valid_actions(self):
        return [action for action, _ in self._neighbors(self.agent_pos)]

    def get_optimal_action(self):
        """
        Returns optimal absolute action:
        0=north, 1=east, 2=south, 3=west
        """
        if np.array_equal(self.agent_pos, self.goal_pos):
            return None

        path = self.get_shortest_path()
        if path is None or len(path) < 2:
            return None

        next_pos = path[1]
        move = next_pos - self.agent_pos

        for action, direction in self.action_to_direction.items():
            if np.array_equal(move, direction):
                return action

        return None

    def get_optimal_action_name(self):
        action = self.get_optimal_action()
        if action is None:
            return None
        return self.action_to_name[action]

    def absolute_to_relative_action(self, absolute_action):
        """
        Convert an absolute action into a relative action based on current facing.

        Returns:
            0 = forward
            1 = right
            2 = backward
            3 = left
        """
        if absolute_action is None:
            return None
        return (absolute_action - self.agent_facing) % 4

    def get_optimal_relative_action(self):
        absolute_action = self.get_optimal_action()
        if absolute_action is None:
            return None
        return self.absolute_to_relative_action(absolute_action)

    def get_optimal_relative_action_name(self):
        relative_action = self.get_optimal_relative_action()
        if relative_action is None:
            return None
        return self.relative_action_to_name[relative_action]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        while True:
            self._place_obstacles()
            self.agent_pos = self._random_empty_cell()

            while True:
                self.goal_pos = self._random_empty_cell()
                if not np.array_equal(self.goal_pos, self.agent_pos):
                    break

            if not self.ensure_path or self._bfs_path(self.agent_pos, self.goal_pos) is not None:
                break

        # Random initial facing: 0=N, 1=E, 2=S, 3=W
        self.agent_facing = int(self.np_random.integers(0, 4))

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Action is absolute:
            0=north, 1=east, 2=south, 3=west
        Agent facing updates to the attempted move direction if move succeeds.
        """
        self.current_step += 1

        direction = self.action_to_direction[action]
        proposed_pos = self.agent_pos + direction

        hit_wall = not self._in_bounds(proposed_pos)
        hit_obstacle = False

        if not hit_wall:
            proposed_pos = proposed_pos.astype(np.int32)
            if self._is_obstacle(proposed_pos):
                hit_obstacle = True
            else:
                self.agent_pos = proposed_pos
                self.agent_facing = action  # update facing only on successful move

        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward = 10
        elif hit_wall or hit_obstacle:
            reward = -2
        else:
            reward = -1

        observation = self._get_obs()
        info = self._get_info()
        info["hit_wall"] = bool(hit_wall)
        info["hit_obstacle"] = bool(hit_obstacle)

        return observation, reward, terminated, truncated, info

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        for r in range(self.size):
            for c in range(self.size):
                if self.obstacles[r, c] == 1:
                    grid[r][c] = "X"

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos

        grid[gr][gc] = "G"
        grid[ar][ac] = "A"

        print("\nGrid:")
        for row in grid:
            print(" ".join(row))
        print(f"Facing: {self.facing_to_name[self.agent_facing]}")
        print()

    def close(self):
        pass
