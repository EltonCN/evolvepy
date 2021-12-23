from abc import ABC


import numpy as np

from evolvepy.generator import Generator
from evolvepy.evaluator import Evaluator

class Callback(ABC):

    def __init__(self):
        self._generator : Generator = None
        self._evaluator : Evaluator = None

    @property
    def generator(self) -> Generator:
        return self._generator
    
    @generator.setter
    def generator(self, value:Generator) -> None:
        if isinstance(value, Generator):
            self._generator = value
        else:
            raise ValueError("Generator must be a evolvepy Generator instance")
    
    @property
    def evaluator(self) -> Evaluator:
        return self._evaluator
    
    @evaluator.setter
    def evaluator(self, value:Evaluator) -> None:
        if isinstance(value, Evaluator):
            self._evaluator = value
        else:
            raise ValueError("Evaluator must be a evolvepy Evaluator instance")
        

    def on_start(self) -> None: #NOSONAR
        pass

    def on_generator_start(self) -> None: #NOSONAR
        pass

    def on_generator_end(self) -> None: #NOSONAR
        pass

    def on_evaluator_end(self, fitness:np.ndarray) -> None: #NOSONAR
        pass

    