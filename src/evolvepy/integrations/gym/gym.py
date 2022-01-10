from abc import ABC, abstractmethod
from typing import Dict

import gym
from gym.wrappers import Monitor
import numpy as np
import matplotlib.pyplot as plt

from evolvepy.evaluator import ProcessFitnessFunction

class GymFitnessFunction(ProcessFitnessFunction, ABC):

    def __init__(self, env_name:str, show:bool=False, save:bool=False ) -> None:
        '''
            GymFitnessFunction constructor.

            Args:
                show (bool): whether to show the graphical output of the environment.
                save (bool): whether to save the graphical output in a file.
        '''
        super().__init__(reset=save)

        self._show = show
        self._save = save
        self._env_name = env_name
        
        self._env = None
        self._count = 0

        try:
            get_ipython()
            self._jupyter = True
        except Exception:
            self._jupyter = False
    
    def setup(self) -> None:
        '''
            Initializes the environment.
        '''
        self._env = gym.make(self._env_name)

        if self._save:
            self._env = Monitor(self._env, "./video"+str(self._count), force=True)
            self._count += 1


    @abstractmethod
    def behaviour(self, obs:object, individual:np.ndarray) -> object:
        ...
    
    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        '''
            Evaluates the individual throug the environment.
        '''
        individual = individuals[0]

        obs = self._env.reset()
        done = False
        total_reward = 0
        while not done:
            action = self.behaviour(obs, individual)

            obs, rew, done, info = self._env.step(action)

            if self._show:
                self._show_env()

            total_reward += rew

        if self._save:
            self._env.close()

        return total_reward
    
    def _show_env(self):
        if self._jupyter:
            self._env.render()
        else:
            from IPython import display
            plt.figure(3)
            plt.clf()
            plt.imshow(self._env.render(mode='rgb_array'))
            plt.axis('off')

            display.clear_output(wait=True)
            display.display(plt.gcf())
            self._env.render()
    