import numpy as np
from evolvepy.integrations.unity_gym.unity import UnityFitnessFunction
from nn import compute

class Unity3DBallEvaluator(UnityFitnessFunction):

    def behaviour(self, obs: object, individual: np.ndarray) -> object:
        action = compute(individual, obs[0])
        action = (2.0*action)-1.0

        return action