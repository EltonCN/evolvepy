from __future__ import annotations
from abc import ABC, abstractmethod


from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike

class Layer(ABC):

    def __init__(self, name:str=None, dynamic_parameters:bool = False, chromossome_names : Union[str, List[str], None] = None):
        
        if name is None:
            self._name : str = self.__class__.__name__
        else:
            self._name = name

        self._dynamic_parameters : bool = dynamic_parameters

        if isinstance(chromossome_names, str):
            self._chromossome_names = [chromossome_names]
        else:
            self._chromossome_names = chromossome_names

        self._next = None


    @property
    def dynamic_parameters(self):
        return self._dynamic_parameters

    def __call__(self, arg:ArrayLike) -> np.ndarray:        
        population = np.asarray(arg)
        result = population.copy()

        if self._chromossome_names is None: # Without specified name
            if len(population.dtype) == 0: # and only one chromossome
                result = self.call(population)
            else:
                for name in population.dtype.names: # and multiple chrmossomes
                    result[name] = self.call(population[name])
        else:
            for name in self._chromossome_names:
                result[name] = self.call(population[name])

        return result
    
    def call(self, chromossomes:np.ndarray) -> np.ndarray:
        return chromossomes