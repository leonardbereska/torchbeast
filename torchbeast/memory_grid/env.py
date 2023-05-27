import time
import pygame
import numpy as np
# import gymnasium as gym
from collections import namedtuple
import gym  # needs to be uniform for the whole code base

from memory_grid.maze import GridMaze, get_maze_specs

# TODO all the randomization should be done in a separate file
rand_specs = namedtuple('rand_specs', 'seed maze_not_fixed target_not_fixed agent_not_fixed sequence_not_fixed')


def get_rand_specs(rand_name):
    """
    Get randomization specifications from experiment name.
    rand_name: name of the experiment, e.g. 'mxtxaos42'
    return: rand_specs
    """
    # extract everything after the s if contains s, otherwise use 42 as seed
    if 's' in rand_name:
        seed = int(rand_name.split('s')[-1])
    else:
        seed = 42

    # m means maze, t means target, a means agent
    def not_fixed(char):
        return rand_name.split(char)[-1][0] == 'o'  # o means not fixed

    maze_not_fixed = not_fixed('m')
    target_not_fixed = not_fixed('t')
    agent_not_fixed = not_fixed('a')
    sequence_not_fixed = True
    return rand_specs(seed,
                      maze_not_fixed,
                      target_not_fixed,
                      agent_not_fixed,
                      sequence_not_fixed)


class GridMazeEnv():
    """
    Grid-based environment for memory tasks.
    This environment is designed for memory tasks and features a 2D grid layout, as opposed to the 3D layout of the standard memory environment. The agent's observation is partial and egocentric, but from an overhead perspective. Movement is restricted to discrete grid positions, rather than continuous motion.
    This environment is simpler and more compact than the standard memory environment, making it a good choice for quick prototyping or testing memory-related algorithms.
    """

    def __init__(self, env_name, rand_name=None, view_distance=1, verbose=False, render_mode='human'):
        self.verbose = verbose
        self.view_distance = view_distance
        self.observation_image_size = 2 * self.view_distance + 1
        self.rand_specs = get_rand_specs(rand_name)
        self.random_seed = self.rand_specs.seed
        self.max_episode_steps = 200  # TODO make this a hyperparameter?

        self.maze_specs = get_maze_specs(env_name)
        self.n_targets = self.maze_specs.n_targets
        self.current_target_id = None

        self.grid_maze = GridMaze(self.maze_specs, seed=self.get_seed('maze'))
        self.initialize_maze_env()

        self.action_space = gym.spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.action_space.seed(self.random_seed)  # seed for action_space sampler

        self.observable_entities = [' ', '#'] + [str(t) for t in
                                                 range(self.n_targets)]  # ' ' is empty space, '#' is wall
        # self.observation_space = gym.spaces.Dict({
        #     'image': gym.spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8),
        #     'target': gym.spaces.Discrete(self.n_targets),
        # })
        image_size = 2 * self.view_distance + 1 + 2  # size of the image, e.g. 3x3, 5x5, 7x7, etc. (2 for padding with target id)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8)
        self.render_mode = render_mode
        self.window = None

    def seed(self, seed=None):
        self.random_seed = seed
        self.grid_maze.seed(seed)
        self.action_space.seed(seed)
        return seed
    def get_seed(self, for_what):
        """
        Get seed for random number generator.
        for_what: 'environment', 'target', 'agent'
        return: seed if for_what is not fixed, None otherwise
        """
        key = for_what + '_not_fixed'
        not_fixed = getattr(self.rand_specs, key)
        if not_fixed:
            return None
        else:
            return self.random_seed

    def initialize_maze_env(self):
        self.maze = self.grid_maze.entity_layer
        self.agent_position = self.grid_maze.agent_position
        self.target_positions = self.grid_maze.target_positions
        self.current_target_id = self.sample_new_target_id()
        self.total_reward = 0
        self.steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self.grid_maze = GridMaze(self.maze_specs, seed=seed)
        if self.get_seed('maze') is None:
            self.grid_maze = GridMaze(self.maze_specs, seed=self.get_seed('maze'))
        self.initialize_maze_env()

        if self.get_seed('agent') is None:
            self.randomize_agent_position()
        if self.get_seed('target') is None:
            self.randomize_target_position()
            # mark targets in maze entity layer
        for i, target_position in enumerate(self.target_positions):
            self.maze[target_position[0], target_position[1]] = str(i)
        return self.get_observation(), {}

    def randomize_agent_position(self):
        np.random.seed(self.get_seed('agent'))
        self.agent_position = self.sample_free_position()
        if self.verbose:
            print('randomized agent position to: ', *self.agent_position)

    def randomize_target_position(self):
        np.random.seed(self.get_seed('target'))
        self.target_positions = np.random.permutation(self.target_positions)  # permute target positions
        if self.verbose:
            print('randomized target positions to: ', *self.target_positions)

    def sample_free_position(self):
        position = np.random.randint(0, self.maze.shape[0], size=2)
        while self.is_wall(position) or self.is_target(position):
            position = np.random.randint(0, self.maze.shape[0], size=2)
        return position

    def is_wall(self, position):
        return self.maze[position[0]][position[1]] == '#'

    def is_target(self, position):
        """
        Check if position is a target position.
        position: 1D array of 2 elements (e.g. array([1, 2]))
        target_positions: list of 1D arrays of 2 elements (e.g. [array([1, 2]), array([3, 4])])
        return: True if position is a target position, False otherwise
        """
        return any([np.array_equal(position, target_position) for target_position in self.target_positions])

    def sample_new_target_id(self):
        available_target_ids = [t for t in range(self.n_targets) if t != self.current_target_id]
        np.random.seed(None)  # now experiments are not reproducible anymore
        # TODO How to make experiments reproducible again?
        # TODO Answer: use a different random number generator for sampling the target id
        # TODO e.g. self.rng_target_sequence = np.random.default_rng(seed=42), self.rng_target_sequence.choice(available_target_ids)
        return np.random.choice(available_target_ids)

    def step(self, action):
        self.update_agent_position(action)

        reward = self.get_reward()
        observation = self.get_observation()
        terminated = False
        truncated = False
        if self.steps >= self.max_episode_steps:
            truncated = True
        info = {}
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def update_agent_position(self, action):
        new_position = self.agent_position.copy()
        if action == 0:
            new_position[0] -= 1  # up
        elif action == 1:
            new_position[1] += 1  # right
        elif action == 2:
            new_position[0] += 1  # down
        elif action == 3:
            new_position[1] -= 1  # left
        else:
            raise ValueError('Invalid action: {}'.format(action))

        if not self.is_wall(new_position):
            if self.verbose:
                print('moved to new position: ', *new_position)
            self.agent_position = new_position
        else:
            if self.verbose:
                print('hit wall, stay at: ', *self.agent_position)

    def get_reward(self):
        reward = 0
        current_target_position = self.target_positions[self.current_target_id]
        if np.array_equal(self.agent_position, current_target_position):
            reward = self.reward_and_give_new_target()
        return reward

    def reward_and_give_new_target(self):
        self.print_if_verbose('found target', self.current_target_id)
        self.current_target_id = self.sample_new_target_id()
        self.print_if_verbose('new target', self.current_target_id)

        reward = 1
        self.total_reward += reward
        return reward

    def observe_field_of_view(self):
        """
        Get observation of the agent's surroundings and the current target id.
        return: 1D array of entity chars (e.g. [' ', '#', ' ', ' ', '1', ' ', ' ', ' ', '2'])
        """
        a = self.agent_position
        v = self.view_distance  # for example, v=1: 3x3, v=2: 5x5 space around agent
        
        maze = self.maze.copy()
        # add a border of walls around the maze to avoid index out of bounds errors
        maze = np.pad(maze, v-1, 'constant', constant_values='#')
        # observation = self.maze[a[0] - v:a[0] + v + 1, a[1] - v:a[1] + v + 1]
        # shift all coordinates by v-1 to account for the added border
        observation = maze[a[0]-1:a[0] + 2*v, a[1]-1:a[1] + 2*v]
        return observation

    def entity_to_color(self, entity):
        """
        Convert entity char to RGB color. 
        return: 1D array of RGB values (e.g. [255, 0, 0])
        """
        if entity == ' ':
            return np.array([255, 255, 255])  # empty space is white 
        elif entity == '#':
            return np.array([0, 0, 0])  # wall is black 
        else:
            assert entity in ['0', '1', '2', '3', '4', '5'], 'Invalid entity: {}'.format(entity)
            # target colors are red, green, blue, yellow, cyan, magenta
            target_colors = [np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
                             np.array([255, 255, 0]), np.array([0, 255, 255]), np.array([255, 0, 255])]
            return target_colors[int(entity)]


    def get_observation(self):
        visible_entities = self.observe_field_of_view()

        # surround the visible entities with walls

        visible_entities = np.pad(visible_entities, 1, 'constant', constant_values=str(self.current_target_id))


        image = np.zeros((visible_entities.shape[0], visible_entities.shape[1], 3), dtype=np.uint8)
        for i in range(visible_entities.shape[0]):
            for j in range(visible_entities.shape[1]):
                image[i, j] = self.entity_to_color(visible_entities[i, j])
        # create a frame around the image with the current target color
        # pad the image array with 1 pixel on each side
        # pixel = self.entity_to_color(str(self.current_target_id))
        # image = pad_image(image, 1, pixel)
        # observation = {'image': image, 'target': self.current_target_id}
        # swap axes from hwc to chw
        image = np.swapaxes(image, 2, 0)
        return image

    def print_maze(self):
        maze = self.maze.copy()  # copy to avoid changing the original maze
        maze[self.agent_position[0], self.agent_position[1]] = 'A'  # mark agent position
        print(maze)

    def print_observation(self):
        observation = self.observe_field_of_view()
        string = ''
        for row in observation:
            row = ''.join(row) + '\n'
            string = string + row
        string = string[:5] + 'A' + string[6:]  # add agent position in the middle
        observation = string
        print(observation)

    def render(self, full_view=True):

        if self.render_mode == 'rgb_array':
            img = self.get_observation()['image']
            return img

        elif self.render_mode == 'simple':
            if self.window is None:
                pygame.init()
            print('current target id:', self.current_target_id)
            print('total reward: ', self.total_reward)
            time.sleep(0.1)
            if full_view:
                self.print_maze()
            else:
                self.print_observation()

        elif self.render_mode == 'human':
            # TODO refactor: make separate functions for full and partial view

            def get_maze_image():
                a = self.agent_position
                v = self.view_distance
                maze = self.maze.copy()  
                image = np.zeros((maze.shape[0], maze.shape[1], 3))

                for i in range(maze.shape[0]):
                    for j in range(maze.shape[1]):
                        image[i, j] = self.entity_to_color(maze[i, j])

                        if abs(i - a[0]) > v or abs(j - a[1]) > v:  # color non-observed area 
                            image[i, j] = image[i, j] * 0.5  # zero for black, 0.5 for gray 

                return image, maze
            

            def get_image_around_agent():
                a = self.agent_position
                v = self.view_distance
                maze = self.maze.copy()  
                # add a border of walls around the maze to avoid index out of bounds errors
                maze = np.pad(maze, v-1, 'constant', constant_values='#')
                # shift all coordinates by v-1 to account for the added border
                image = np.zeros((2 * v + 1, 2 * v + 1, 3))

                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        # image[i, j] = self.entity_to_color(maze[a[0] - v + i, a[1] - v + j])
                        # shift all coordinates by v-1 to account for the added border
                        image[i, j] = self.entity_to_color(maze[a[0] + i - 1, a[1] + j - 1])
                        # print(a[0] - v + i, a[1] - v + j)
                        # print(i, j, image[i, j])

                return image, maze
            
            if full_view:
                image, maze = get_maze_image()
            else:
                image, maze = get_image_around_agent()
            image = self.get_observation()

            time.sleep(0.03)

            target_id = self.current_target_id
            target_color = self.entity_to_color(str(target_id))

            image = np.transpose(image, (1, 0, 2))  # convert to (width, height, channels) for pygame
            screen_size = 640  # TODO make this a hyperparameter
 
            if self.window is None:
                pygame.init()
                self.window = pygame.display.set_mode((screen_size, screen_size))
                pygame.display.set_caption('Maze')

           
            image_surface = pygame.surfarray.make_surface(image)
            image_surface = pygame.transform.scale(image_surface, (screen_size, screen_size))
            self.window.blit(image_surface, (0, 0))

            def create_marker_surface(marker_size, target_color):
                marker_surface = pygame.Surface((marker_size, marker_size), pygame.SRCALPHA)
                pygame.draw.circle(marker_surface, target_color, (marker_size // 2, marker_size // 2), marker_size // 2)
                return marker_surface

            def calculate_agent_position(agent_position, screen_size, maze_shape):
                factor = screen_size / maze_shape[0]
                agent_position = (factor * agent_position[1], factor * agent_position[0])
                return agent_position

            def center_marker(agent_position, marker_size, factor):
                offset = factor / 2 - marker_size / 2 + 1  # shift marker to center of grid cell
                agent_position = (agent_position[0] + offset, agent_position[1] + offset)
                return agent_position
            
            
            if full_view:
                marker_size = 50 / maze.shape[0] * 11  # scale marker size with maze size
                factor = screen_size / maze.shape[0]

                # Create surfaces for image and marker
                marker_surface = create_marker_surface(marker_size, target_color)
                agent_position = calculate_agent_position(self.agent_position, screen_size, maze.shape)
                agent_position = center_marker(agent_position, marker_size, factor)

                # Blit image and marker surfaces to window
            else:
                marker_size = 400 / (2 * self.view_distance + 1)  # scale marker size with maze size
                marker_surface = create_marker_surface(marker_size, target_color)
                # place agent in the center of the screen
                agent_position = (screen_size / 2 - marker_size / 2, screen_size / 2 - marker_size / 2)

            
            # self.window.blit(marker_surface, agent_position)
            # font = pygame.font.Font('freesansbold.ttf', 32)
            # text = font.render('Score: ' + str(self.total_reward), True, (128, 128, 128))
            # self.window.blit(text, (10, 10))    



            pygame.display.update()

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()

    def print_if_verbose(self, *args):
        if self.verbose:
            print(*args)


if __name__ == '__main__':
    choices = ['GridMaze9x9', 'GridMaze11x11', 'GridMaze13x13', 'GridMaze15x15']
    env = GridMazeEnv(choices[0], verbose=True)
    env.print_maze()
