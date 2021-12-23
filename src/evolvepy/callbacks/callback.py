from __future__ import annotations

from typing import Dict, List
import numpy as np
from evolvepy.configurable import Configurable

from evolvepy.generator import Generator
from evolvepy.evaluator import Evaluator

class Callback(Configurable):

    def __init__(self, parameters:Dict[str, object]=None, dynamic_parameters:Dict[str,bool]= None):
        super().__init__(parameters, dynamic_parameters)
        self._generator : Generator = None
        self._evaluator : Evaluator = None
        self._callbacks : List[Callback] = []

    @property
    def generator(self) -> Generator:
        return self._generator
    
    @generator.setter
    def generator(self, value:Generator) -> None:
        if isinstance(value, Generator):
            self._generator = value
        else:
            raise ValueError("Generator must be a evolvepy Generator instance.")
    
    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator
    
    @evaluator.setter
    def evaluator(self, value:Evaluator) -> None:
        if isinstance(value, Evaluator):
            self._evaluator = value
        else:
            raise ValueError("Evaluator must be a evolvepy Evaluator instance.")

    @property
    def callbacks(self) -> List[Callback]:
        return self._callbacks
        
    @callbacks.setter
    def callbacks(self, value:List[Callback]) -> None:
        if not isinstance(value, list): 
            raise ValueError("callbacks must be a list")
        
        for callback in value:
            if not isinstance(callback, Callback):
                raise ValueError("All callbacks elements must be a evolvepy Callback instance.")
        
        self._callbacks = value

        

    def on_start(self) -> None: #NOSONAR
        pass

    def on_generator_start(self) -> None: #NOSONAR
        pass

    def on_generator_end(self, population:np.ndarray) -> None: #NOSONAR
        pass

    def on_evaluator_end(self, fitness:np.ndarray) -> None: #NOSONAR
        pass

    