from abc import ABC
import multiprocessing as mp
from typing import Optional

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


from evolvepy.integrations.gym import GymFitnessFunction



class UnityFitnessFunction(GymFitnessFunction, ABC):
    '''
    Evaluates individuals using a Unity's environment. Must be inherited by the user.

    Can be used with ProcessEvaluator.
    '''
    
    def __init__(self, env_path: Optional[str]=None, show: bool = False) -> None:
        '''
        UnityFitnessFunction constructor.

        Args:
            env_path (Optional[str], optional): The path for the environment build. Defaults to None (waits for connection with Unity Editor (play button)).
            show (bool, optional): Whether to show the environment graphic output. Defaults to False.
        '''
        super().__init__(env_path, show=show, save=False)

    def setup(self) -> None:
        '''
        Prepares the environment instance.
        '''
        pid = mp.current_process().ident

        unity_env = UnityEnvironment(self._env_name, no_graphics=not self._show, worker_id=pid)
        self._env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

