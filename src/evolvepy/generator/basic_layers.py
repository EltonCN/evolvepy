from typing import Tuple
import numpy as np

from evolvepy.generator.context import Context

from .layer import Layer

class Sort(Layer):

    def __init__(self, name: str = None, ):
        super().__init__(name=name)

    def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
        indexs = np.argsort(fitness)
        indexs = np.flip(indexs)

        context.sorted = True

        return population[indexs], fitness[indexs]

class FilterFirsts(Layer):

    def __init__(self, n_to_pass:int=1, name: str = None):
        parameters = {"n_to_pass":n_to_pass}
        dynamic_parameters = {"n_to_pass":True}
        super().__init__(name=name, parameters=parameters, dynamic_parameters=dynamic_parameters)

    def call(self, population: np.ndarray, fitness: np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
        n_to_pass = self.parameters["n_to_pass"]
        
        return population[0:n_to_pass], fitness[0:n_to_pass]