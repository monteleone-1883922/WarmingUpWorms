from worms_env import WormsEnv
import gymnasium as gym

from gymnasium.envs.registration import register
import pygame

register(
     id="worms_env",
     entry_point="worms_env:WormsEnv",
     max_episode_steps=300,
)
env = gym.make('worms_env',  env_file="Data/00-example.txt", render_mode="human")



env.render_frame()
frame_surface = pygame.display.get_surface()

# Salva il frame come immagine (formato PNG)
pygame.image.save(frame_surface, "frame.png")

env.set_active_head(1)
env.append_worms_positions(0)

env.render_frame()

frame_surface = pygame.display.get_surface()

# Salva il frame come immagine (formato PNG)
pygame.image.save(frame_surface, "frame2.png")

env.close()

