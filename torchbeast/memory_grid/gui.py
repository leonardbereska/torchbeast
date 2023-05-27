import os
import sys

sys.path.append(os.path.abspath(os.curdir))
print(sys.path)
import pygame
import argparse

import memory_grid


class GridMazeGUI:
    def __init__(self, env, seed=None, full_view=True):
        self.env = env
        self.seed = seed
        self.full_view = full_view
        self.key_to_action = {'up': 0,
                              'right': 1,
                              'down': 2,
                              'left': 3}

    def run(self):
        self.reset()
        pygame.key.set_repeat(200, 100)  # delay, interval in ms
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action):
        _, _, terminated, truncated, _ = self.env.step(action)
        if terminated:
            print('terminated')
            self.reset()
        elif truncated:
            print('total reward', self.env.total_reward)
            print('truncated')
            self.reset()
        else:
            self.env.render(full_view=self.full_view)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render(full_view=self.full_view)

    def key_handler(self, event):
        key: str = event.key
        # print('pressed', key)

        if key == 'escape':
            self.env.close()
            sys.exit()
        if key == 'backspace':
            self.reset()

        if key in self.key_to_action.keys():
            action = self.key_to_action[key]
            self.step(action)
        else:
            print(key, 'is not a valid key')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_view', action='store_true', default=False)  
    parser.add_argument('--view_distance', type=int, default=2)
    parser.add_argument('--size', type=int, default=9)
    parser.add_argument('--rand_name', type=str, default='motoaos42')
    args = parser.parse_args()

    env_name = 'GridMaze{}x{}'.format(args.size, args.size)
    env = memory_grid.GridMazeEnv(env_name, rand_name=args.rand_name, view_distance=args.view_distance, render_mode='human')
    gui = GridMazeGUI(env, seed=42, full_view=args.full_view)
    gui.run()
