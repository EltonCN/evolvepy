import gym
from gym.wrappers import Monitor
import numpy as np
import tensorflow.keras as keras
from matplotlib import pyplot as plt
import gc


from evolvepy.integrations.tf_keras import ProcessTFKerasFitnessFunction


class BipedalWalkerFitnessFunction(ProcessTFKerasFitnessFunction):
    def __init__(self, args, show=False, save=False ) -> None:
        super().__init__(reset=save, args=args)

        self._env = None
        self._show = show
        self._save = save
        self._count = 0
        super().__init__(args=args)
    
    def setup(self) -> None:
        '''
            Initializes the environment.
        '''
        self._env = gym.make("BipedalWalker-v3")

        if self._save:
            self._env = Monitor(self._env, "./video"+str(self._count), force=True)
            self._count += 1

        
    def evaluate(self, model: keras.Model) -> np.ndarray:
        gc.collect()
        keras.backend.clear_session()
        
        obs = self._env.reset()
        done = False
        total_reward = 0
        while not done:
            obs = obs.reshape((1, 24))

            action = model(obs).numpy()[0]
            action = (2.0*action)-1.0

            obs, rew, done, info = self._env.step(action)
    
            total_reward += rew

            if self._show:
                if isnotebook():
                    show_state(self._env)
                else:
                    self._env.render()

        if self._save:
            self._env.close()

        gc.collect()
        keras.backend.clear_session()

        return total_reward


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter    

def show_state(env):
    from IPython import display
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())