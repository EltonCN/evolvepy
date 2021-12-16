import numpy as np

from evolvepy.generator import Generator
from evolvepy.evaluator import Evaluator

class Evolver:
    def __init__(self, generator:Generator, evaluator:Evaluator, generation_size:int):
        self._generator = generator
        self._evaluator = evaluator
        self._generation_size = generation_size

    def evolve(self, generations:int):

        self._history = np.empty((generations, self._generation_size), np.float64)

        for i in range(generations):
            population = self._generator.generate(self._generation_size)

            fitness = self._evaluator(population)

            self._generator.fitness = fitness

            self._history[i] = fitness.flatten()
        
        return self._history, population

