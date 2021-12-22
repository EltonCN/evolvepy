from __future__ import annotations
from abc import ABC, abstractmethod


from typing import Dict, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from evolvepy.generator.context import Context

class Layer(ABC):

    __layer_count = 0

    def __init__(self, name:str=None, dynamic_parameters:Dict[str, bool] = None, parameters:Dict[str, object] = None):
        
        if name is None:
            name = self.__class__.__name__
        
        self._name = name + str(Layer.__layer_count)
        Layer.__layer_count += 1

        self._parameters = parameters

        if dynamic_parameters is None and parameters is None:
            dynamic_parameters = {}
        elif dynamic_parameters is None:
            dynamic_parameters = dict.fromkeys(list(parameters.keys()), True)
        self._dynamic_parameters : Dict[str, bool] = dynamic_parameters

        self._next : List[Layer] = []
       

        self._population = None
        self._fitness = None
        self._context = None

        self._prev_count : int = 0

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

            layer._prev_count += 1

    @property
    def population(self) -> np.ndarray:
        return self._population
    
    @property
    def fitness(self) -> np.ndarray:
        return self._fitness

    @property
    def context(self) -> Context:
        return self._context

    def __call__(self, population:ArrayLike, fitness:Union[ArrayLike, None]=None, context:Union[Context, None]=None) -> np.ndarray:          
        population = np.asarray(population)

        if fitness is None:
            fitness = np.zeros(len(population), dtype=np.float32)
        fitness = np.asarray(fitness).flatten()

        if context is None:
            context = Context(population.dtype.names)

        population, fitness = self.call(population, fitness, context)

        self._population = population
        self._fitness = fitness

        for layer in self._next:
                next_context = context
                if len(self._next) != 1:
                    next_context = next_context.copy()
                    
                layer(population, fitness, next_context)

        self._context = context

        return population, fitness

    def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> Tuple[np.ndarray, np.ndarray]:
        return population, fitness

class Concatenate(Layer):

    def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None):
        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

        self._received_count = 0

    def __call__(self, population: np.ndarray, fitness: np.ndarray, context:Union[Context, None]=None) -> Tuple[np.ndarray, np.ndarray]:
        population = np.asarray(population)

        if fitness is None:
            fitness = np.zeros(len(population), dtype=np.float32)
        fitness = np.asarray(fitness).flatten()

        if context is None:
            context = Context(population.dtype.names)

        if self._received_count == 0:
            self._population = population
            self._fitness = fitness
        else:
            self._population = np.concatenate((self._population, population))
            self._fitness = np.concatenate((self._fitness, fitness))

        self._received_count += 1

        if self._prev_count == self._received_count:
            self._received_count = 0

            for layer in self._next:
                next_context = context
                if len(self._next) != 1:
                    next_context = next_context.copy()

                layer(population, fitness, next_context)

        self._context = context

        return population, fitness




class ChromossomeOperator(Layer):
    def __init__(self, name: str = None, dynamic_parameters: Dict[str, bool] = None, parameters: Dict[str, object] = None, chromossome_names: Union[str, List[str], None] = None):
        super().__init__(name=name, dynamic_parameters=dynamic_parameters, parameters=parameters)

        if isinstance(chromossome_names, str):
            self._chromossome_names = [chromossome_names]
        else:
            self._chromossome_names = chromossome_names

    
    def call(self, population:np.ndarray, fitness:np.ndarray, context:Context) -> np.ndarray:        
        result = population.copy()

        if self._chromossome_names is None: # Without specified name
            if len(population.dtype) == 0 and not context.blocked: # and only one chromossome
                result = self.call_chromossomes(population, fitness, context)
            else:
                for name in population.dtype.names: # and multiple chrmossomes
                    if not context.blocked[name]:
                        result[name] = self.call_chromossomes(population[name], fitness, context)
        else:
            for name in self._chromossome_names:
                if not context.blocked[name]:
                        result[name] = self.call_chromossomes(population[name], fitness, context)


        return result, fitness
    
    def call_chromossomes(self, chromossomes:np.ndarray, fitness:np.ndarray, context:Context) -> np.ndarray:
        return chromossomes