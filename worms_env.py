import gymnasium as gym
from gymnasium import spaces
from utils_functions import build_field, compute_next_actions
import pygame
import numpy as np


class WormsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_file: str, render_mode: str = None,
                 enable_wormholes: bool = False, max_val: int = 0,
                 use_single_worm: bool = True):
        self.window_size = 512
        self.active_worm = 0
        self.active_worm_len = 0
        self._table = []
        self.worms_positions = []
        self.wormholes_enabled = enable_wormholes
        self.active_head = None
        with open(env_file, "r") as f:
            f.readline()
            self.worms_lengths = [int(x) for x in f.readline().split()]
            if use_single_worm:
                self.worms_lengths = self.worms_lengths[:1]
            for line in f:
                self._table.append(line.split())
        self.field, max_val_node = build_field(self._table)
        self.worm_placed_val = max_val_node * sum(self.worms_lengths)
        self.available_movements = [i for i in range(len(self.field.x))]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        # useless attributes
        self.action_space = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.field = build_field(self._table)
        observation = {
            "worms": self.worms_lengths,
            "field": self.field
        }
        self.worms_positions = []
        self.active_worm = 0
        self.active_worm_len = 0
        self.active_head = None
        self.available_movements = [i for i in range(len(self.field.x))]
        if self.render_mode == "human":
            self._render_frame()
        return self.observation_space, observation

    def step(self, action):
        reward = int(self.field.x[action][0])
        self.worms_positions.append(action)
        self.field.x[action][0] = self.worm_placed_val
        self.active_worm_len += 1
        actions = compute_next_actions(action, self.field.x, len(self._table[0]), self.worm_placed_val)
        terminated = False
        self.active_head = action
        if self.active_worm_len == self.worms_lengths[self.active_worm] and self.active_worm + 1 >= len(
                self.worms_lengths):
            terminated = True
            self.active_head = None
        elif len(actions) == 0 and self.active_worm_len != self.worms_lengths[self.active_worm]:
            terminated = True
            self.active_head = None
            reward = self.worm_placed_val 
        elif self.active_worm_len == self.worms_lengths[self.active_worm] and self.active_worm + 1 < len(
                self.worms_lengths):
            self.active_head = None
            self.active_worm += 1
            self.active_worm_len = 0
            actions = [i for i in range(len(self.field.x)) if self.field.x[i][0] != self.worm_placed_val]
        if self.render_mode == "human":
            self._render_frame()
        observation = {
            "worms": self.worms_lengths,
            "field": self.field
        }
        self.available_movements = actions
        return self.observation_space, reward, terminated, False, observation

    def set_active_head(self,val:int):
        self.active_head = val

    def append_worms_positions(self, node: int):
        self.worms_positions.append(node)
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def render_frame(self):
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / max(len(self._table), len(self._table[0]))
        )  # The size of a single grid square in pixels

        # First we draw the target

        for act in self.worms_positions:
            row = act // len(self._table[0])
            column = act % len(self._table[0])
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    (pix_square_size * column, pix_square_size * row),
                    (pix_square_size, pix_square_size),
                ),
            )
        if self.active_head is not None:
            row = self.active_head // len(self._table[0])
            column = self.active_head % len(self._table[0])

            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (pix_square_size * column, pix_square_size * row),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Finally, add some gridlines
        for x in range(len(self._table) + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
        for x in range(len(self._table[0])):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size*len(self._table)),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
