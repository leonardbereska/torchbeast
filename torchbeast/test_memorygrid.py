import memory_grid
from memory_grid.env import GridMazeEnv

def test_gridmazeenv():
    choices = ['GridMaze9x9', 'GridMaze11x11', 'GridMaze13x13', 'GridMaze15x15']
    test_env = choices[0]
    env = GridMazeEnv(test_env, rand_name='motoaos42', view_distance=2, render_mode='human')
    print(env.observation_space)
    env.render()

def test_gui():
    from memory_grid.gui import GridMazeGUI
    env_name = 'GridMaze9x9'
    rand_name = 'motoaos42'
    view_distance = 2
    full_view = False
    env = memory_grid.GridMazeEnv(env_name, rand_name=rand_name, view_distance=view_distance, render_mode='human')
    gui = GridMazeGUI(env, seed=42, full_view=full_view)
    gui.run()


if __name__ == '__main__':
    # test_gridmazeenv()
    test_gui()