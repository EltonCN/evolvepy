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

        self._next : List[Layer] = []
        self._parameters = parameters

        self._population = None
        self._fitness = None

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

    @property
    def next(self) -> Layer:
        return self._next
    
    @next.setter
    def next(self, layer:Layer) -> None:
        if layer not in self._next:
            self._next.append(layer)

    @property
    def population(self) -> np.ndarray:
        return self._population
    
    @property
    def fitness(self) -> np.ndarray:
        return self._fitness


    def __call__(self, population:ArrayLike, fitness:Union[ArrayLike, None]=None) -> np.ndarray:          
        population = np.asarray(population)

        if fitness is None:
            fitness = np.zeros(len(population), dtype=np.float32)
        fitness = np.asarray(fitness)

        population, fitness = self.call(population, fitness)

        self._population = population
        self._fitness = fitness

        for layer in self._next:
            layer(population, fitness)

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

    
    def call(self, population:np.ndarray, fitness:np.ndarray) -> np.ndarray:        
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