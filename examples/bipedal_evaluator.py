import numpy as np
from evolvepy.integrations.gym import GymFitnessFunction
from nn import compute

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