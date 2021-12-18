from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from tensorflow.keras.layers import Dense

class Layer(ABC):

    def __init__(self, name:str=None, dynamic_parameters:Dict[str, bool] = None, parameters:Dict[str, object] = None, chromossome_names : Union[str, List[str], None] = None):
        
        if name is None:
            self._name : str = self.__class__.__name__
        else:
            self._name = name

        if dynamic_parameters is None:
            dynamic_parameters = {}
        self._dynamic_parameters : Dict[str, bool] = dynamic_parameters

        if isinstance(chromossome_names, str):
            self._chromossome_names = [chromossome_names]
        else:
            self._chromossome_names = chromossome_names

        self._next = None
        self._parameters : Dict[str, object] = {}

    @property
    def parameters(self)-> Dict[str, object]:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value:Union[Dict[str, object], Tuple[str, object]]) -> None:
        if isinstance(value, tuple):
            value = {value[0]: value[1]}
        
        keys = list(value.keys())
        for key in keys:
            if key not in self._dynamic_parameters:
                del value[key]
            if self._dynamic_parameters[key] == False:
                del value[key]

        self._parameters.update(value)

    def lock_parameter(self, name:str) -> None:
        if name in self._dynamic_parameters:
            self._dynamic_parameters[name] = False
    
    def unlock_parameter(self, name:str) -> None:
        if name in self._dynamic_parameters:
            self._dynamic_parameters[name] = True


    @property
    def dynamic_parameters(self) -> List[str]:
        return self._dynamic_parameters

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, population:ArrayLike, fitness:Union[ArrayLike, None]=None) -> np.ndarray:        
        population = np.asarray(population)
        result = population.copy()

        if fitness is None:
            fitness = np.zeros(len(population), dtype=np.float32)
        fitness = np.asarray(fitness)

        if self._chromossome_names is None: # Without specified name
            if len(population.dtype) == 0: # and only one chromossome
                result = self.call(population, fitness)
            else:
                for name in population.dtype.names: # and multiple chrmossomes
                    result[name] = self.call(population[name], fitness)
        else:
            for name in self._chromossome_names:
                result[name] = self.call(population[name], fitness)

        return result
    
    def call(self, chromossomes:np.ndarray, fitness:np.ndarray) -> np.ndarray:
        return chromossomes