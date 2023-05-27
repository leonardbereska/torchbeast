import labmaze
import numpy as np
from collections import namedtuple

# The mazes' specifications are the same as in the 3D Memory Maze (Pasukonis et al. 2022).
maze_specs = namedtuple('maze_specs', 'maze_size n_targets max_rooms room_max_size')
memory_maze_7x7   = maze_specs( 7, 2, 6, 5)
memory_maze_9x9   = maze_specs( 9, 3, 6, 5)
memory_maze_11x11 = maze_specs(11, 4, 6, 5)
memory_maze_13x13 = maze_specs(13, 5, 6, 5)
memory_maze_15x15 = maze_specs(15, 6, 9, 3)


def get_maze_specs(env_name):
    choices = ['GridMaze9x9', 'GridMaze11x11', 'GridMaze13x13', 'GridMaze15x15']
    if env_name == 'GridMaze7x7':
        return memory_maze_7x7
    elif env_name == 'GridMaze9x9':
        return memory_maze_9x9
    elif env_name == 'GridMaze11x11':
        return memory_maze_11x11
    elif env_name == 'GridMaze13x13':
        return memory_maze_13x13
    elif env_name == 'GridMaze15x15':
        return memory_maze_15x15
    else:
        raise ValueError('Unknown environment name: {}, must be one of {}'.format(env_name, choices))



class GridMaze():
    """
    2D discrete grid maze generated with labmaze. The specification of the maze is given by maze_specs, taken from the Memory Maze (Pasukonis et al. 2022) environment.
    maze_specs = namedtuple('maze_specs', 'maze_size n_targets max_rooms room_max_size')
    seed: seed for the random maze generation
    returns: entity_layer, target_positions, agent_position as class attributes
    """
    def __init__(self, maze_specs=memory_maze_9x9, seed=None):
        # print(f'Creating GridMaze with seed {seed}')
        self.seed(seed)
        self.n_targets = maze_specs.n_targets
        self.maze = self.get_maze(maze_specs, seed=seed)
        self.set_entity_layer()

    def seed(self, seed):
        self.random_seed = seed
        np.random.seed(seed)

    def get_maze(self, maze_specs, seed): 
        maze = labmaze.RandomMaze(height=maze_specs.maze_size + 2,  # add outer walls (1 on each side)
                                  width=maze_specs.maze_size + 2,   # add outer walls (1 on each side)
                                  max_rooms=maze_specs.max_rooms,
                                  room_min_size=3,
                                  room_max_size=maze_specs.room_max_size,
                                  spawns_per_room=1,
                                  objects_per_room=1,
                                  max_variations=26,
                                  simplify=True,
                                  random_seed=seed)
        return maze

    def set_entity_layer(self):
        self.entity_layer = self.maze.entity_layer
        self.target_positions = self.place_targets()
        self.agent_position = self.place_agent()
        self.update_entity_layer()

    def regenerate(self):
        self.maze.regenerate()
        self.set_entity_layer()

    def place_targets(self):
        possible_target_positions = np.argwhere(self.entity_layer == 'G')
        while self.n_targets > len(possible_target_positions):
            # print(f'Re-generating maze: more targets than possible positions.')
            self.maze.regenerate()
            self.entity_layer = self.maze.entity_layer
            possible_target_positions = np.argwhere(self.entity_layer == 'G')

        idx = np.random.choice(len(possible_target_positions), size=self.n_targets, replace=False)
        target_positions = list(possible_target_positions[idx])
        return target_positions

    def place_agent(self):
        possible_spawn_positions = np.argwhere(self.entity_layer == 'P')
        idx = np.random.choice(len(possible_spawn_positions), size=1, replace=False)[0]
        agent_position = possible_spawn_positions[idx]
        return agent_position

    def update_entity_layer(self):
        # remove possible target positions
        self.entity_layer[self.entity_layer == 'G'] = ' '
        # remove possible agent spawn positions
        self.entity_layer[self.entity_layer == 'P'] = ' '  
        # mark walls with '#'
        self.entity_layer[self.entity_layer == '*'] = '#'

    def test_entity_layer(self):
        # contains n targets
        for i in range(self.n_targets):
            assert len(np.argwhere(self.entity_layer == str(i))) == 1
        # contains walls
        assert len(np.argwhere(self.entity_layer == '#')) > 0
        # contains empty spaces
        assert len(np.argwhere(self.entity_layer == ' ')) > 0
        # contains no other targets
        assert len(np.argwhere(self.entity_layer == 'G')) == 0
        # contains no other spawns
        assert len(np.argwhere(self.entity_layer == 'P')) == 0

    def test_regenerate(self):
        self.regenerate()
        self.test_entity_layer()

    def test_seed(self):
        grid_maze1 = GridMaze(seed=self.seed)
        grid_maze2 = GridMaze(seed=self.seed)
        assert np.array_equal(grid_maze1.entity_layer, grid_maze2.entity_layer)

if __name__ == '__main__':
    grid_maze = GridMaze(seed=0)
    grid_maze.test_entity_layer()
    grid_maze.test_regenerate()
    grid_maze.test_seed()
    print('All tests passed')
