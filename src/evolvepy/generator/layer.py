from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

class Layer(ABC):

    __layer_count = 0

    def __init__(self, name:str=None, dynamic_parameters:Dict[str, bool] = None, parameters:Dict[str, object] = None):
        
        if name is None:
            name = self.__class__.__name__
        
        self._name = name + str(Layer.__layer_count)
        Layer.__layer_count += 1

        if dynamic_parameters is None:
            dynamic_parameters = {}
        self._dynamic_parameters : Dict[str, bool] = dynamic_parameters

        self._next = None
        self._parameters = parameters

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

        if fitness is None:
            fitness = np.zeros(len(population), dtype=np.float32)
        fitness = np.asarray(fitness)

        population, fitness = self.call(population, fitness)
        return population, fitness

    def call(self, population:np.ndarray, fitness:np.ndarray) -> Tuple(np.ndarray, np.ndarray):
        return population, fitness


class ChromossomeOperator(Layer):
    def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None, chromossome_names: Union[str, List[str], None] = None):
        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

        if isinstance(chromossome_names, str):
            self._chromossome_names = [chromossome_names]
        else:
            self._chromossome_names = chromossome_names

    
    def call(self, population:ArrayLike, fitness:Union[ArrayLike, None]=None) -> np.ndarray:        
        result = population.copy()

        if self._chromossome_names is None: # Without specified name
            if len(population.dtype) == 0: # and only one chromossome
                result = self.call_chromossomes(population, fitness)
            else:
                for name in population.dtype.names: # and multiple chrmossomes
                    result[name] = self.call_chromossomes(population[name], fitness)
        else:
            for name in self._chromossome_names:
                result[name] = self.call_chromossomes(population[name], fitness)

        return result, fitness
    
    def call_chromossomes(self, chromossomes:np.ndarray, fitness:np.ndarray) -> np.ndarray:
        return chromossomes