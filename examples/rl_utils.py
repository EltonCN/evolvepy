from typing import Any
import gym
from gym.wrappers import Monitor
import numpy as np
from matplotlib import pyplot as plt

from evolvepy.evaluator import ProcessFitnessFunction
from evolvepy.integrations.gym import GymFitnessFunction

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


class BipedalWalkerFitnessFunction(GymFitnessFunction):

    def __init__(self, show=False, save=False) -> None:
        '''
            BipedalWalkerFitnessFunction constructor.

            Args:
                show (bool): whether to show the graphical output of the environment.
                save (bool): whether to save the graphical output in a file.
                args (dict): can contain the key "time_mode", which indicates if the fitness should be the total amount 
                             of iterations performed in the environment.
        '''
        super().__init__(env_name = "BipedalWalker-v3",show=show, save=save)
    
    def behaviour(self, obs: object, individual: np.ndarray) -> object:
        action = compute(individual, obs)
        action = (2.0*action)-1.0

        return action