import multiprocessing as mp

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

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

class Unity3DBallEvaluator(ProcessFitnessFunction):


    def __init__(self, show=False, args=None, env=None) -> None:
        '''
            Unity3DBallEvaluator constructor.
        '''
        super().__init__(reset=False)

        self._env = env
        self._path = args["path"]
        self._show = show

    def setup(self) -> None:
        '''
            Initializes the environment.
        '''
        if self._env is not None:
            return
            
        pid = mp.current_process().ident

        unity_env = UnityEnvironment(self._path, no_graphics=not self._show, worker_id=pid)
        self._env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

        
    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        '''
            Evaluates the individual throug the environment.
        '''
        individual = individuals[0]

        obs = self._env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = compute(individual, obs[0])
            action = (2.0*action)-1.0

            obs, rew, done, info = self._env.step(action)

            total_reward += rew

        return total_reward