from typing import Any
import gym
from gym.wrappers import Monitor
import numpy as np
from matplotlib import pyplot as plt

from evolvepy.evaluator import ProcessFitnessFunction

def compute(individual:np.ndarray, x:np.ndarray):
    '''
        Foward pass of the dense neural network

        Args:
            individual (np.ndarray): Individual with the weights.
            x (np.ndarray): Network input

        Returns:
            The output of the neural network
    '''
    result = x

    n_layer = len(individual.dtype.names)//2
    
    for i in range(n_layer-1):
        b = individual["layer"+str(i)+"b"]
        w = individual["layer"+str(i)+"w"].reshape((len(b), len(result)))

        result = w@result
        result += b
        result = (np.abs(result)+result)/2

    b = individual["layer"+str(n_layer-1)+"b"]
    w = individual["layer"+str(n_layer-1)+"w"].reshape((len(b), len(result)))

    result = (w@result)+b
    result = 1/(1+np.exp(-result)) 

    return result


class BipedalWalkerFitnessFunction(ProcessFitnessFunction):


    def __init__(self, show=False, save=False , args=None) -> None:
        '''
            BipedalWalkerFitnessFunction constructor.

            Args:
                show (bool): whether to show the graphical output of the environment.
                save (bool): whether to save the graphical output in a file.
                args (dict): can contain the key "time_mode", which indicates if the fitness should be the total amount 
                             of iterations performed in the environment.
        '''
        super().__init__(reset=save)

        self._time_mode = False
        if isinstance(args, dict) and "time_mode" in args:
            self._time_mode = args["time_mode"]

        self._env = None
        self._show = show
        self._save = save
        self._count = 0
    
    def setup(self) -> None:
        '''
            Initializes the environment.
        '''
        self._env = gym.make("BipedalWalker-v3")

        if self._save:
            self._env = Monitor(self._env, "./video"+str(self._count), force=True)
            self._count += 1

        
    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        '''
            Evaluates the individual throug the environment.
        '''
        individual = individuals[0]

        obs = self._env.reset()
        done = False
        total_reward = 0
        while not done:
            action = compute(individual, obs)
            action = (2.0*action)-1.0

            obs, rew, done, info = self._env.step(action)
    
            if self._time_mode:
                total_reward += 1
            else:
                total_reward += rew

            if self._show:
                if isnotebook():
                    show_state(self._env)
                else:
                    self._env.render()

        if self._save:
            self._env.close()

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