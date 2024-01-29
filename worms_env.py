import gymnasium as gym
from gymnasium import spaces
from utils_functions import apply_wormholes_policy

class WormsEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_file: str, render_mode: str = None, enable_wormholes: bool = False, max_val: int = 0):
        self.field = []
        with open(env_file, "r") as f:
            f.readline()
            self.snakes_lengths = [int(x) for x in f.readline().split()]
            for line in f:
                self.field.append(apply_wormholes_policy(line.split(), enable_wormholes, max_val))
        self.render_mode = render_mode
        self.window = None
        self.clock = None

