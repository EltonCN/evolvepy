from typing import List, Union
import numpy as np

from evolvepy.generator import Generator
from evolvepy.evaluator import Evaluator
from evolvepy.callbacks import Callback

class Evolver:
    def __init__(self, generator:Generator, evaluator:Evaluator, generation_size:int, callbacks:Union[Callback, List[Callback]]=None):
        self._generator = generator
        self._evaluator = evaluator
        self._generation_size = generation_size
        
        if callbacks is None:
            callbacks = []
        elif isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self._callbacks = callbacks

        for callback in self._callbacks:
            callback.generator = generator
            callback.evaluator = evaluator

    def evolve(self, generations:int):

        self._history = np.empty((generations, self._generation_size), np.float64)

        for i in range(generations):
            for callback in self._callbacks:
                callback.on_generator_start()

            population = self._generator.generate(self._generation_size)

            for callback in self._callbacks:
                callback.on_generator_end()

            fitness = self._evaluator(population)

            for callback in self._callbacks:
                callback.on_evaluator_end(fitness)

            self._generator.fitness = fitness

            self._history[i] = fitness.flatten()
        
        return self._history, population

