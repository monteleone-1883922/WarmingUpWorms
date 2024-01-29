import gymnasium as gym
from gymnasium import spaces
from utils_functions import build_field

class WormsEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, env_file: str, render_mode: str = None, enable_wormholes: bool = False, max_val: int = 0):
        table = []
        with open(env_file, "r") as f:
            f.readline()
            self.snakes_lengths = [int(x) for x in f.readline().split()]
            for line in f:
                table.append(line.split())
        self.field = build_field(table)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

