import multiprocessing as mp

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from evolvepy.integrations.unity_gym.unity import UnityFitnessFunction

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

class Unity3DBallEvaluator(UnityFitnessFunction):

    def behaviour(self, obs: object, individual: np.ndarray) -> object:
        action = compute(individual, obs[0])
        action = (2.0*action)-1.0

        return action