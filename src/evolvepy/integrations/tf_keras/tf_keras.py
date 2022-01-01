from typing import Callable, List

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from evolvepy.evaluator import Evaluator
from evolvepy.generator import Descriptor

def transfer_weights(individual:np.ndarray, model:keras.Model) -> None:
    weights:tf.Variable
    for weights in model.weights:
        weights.assign(individual[weights.name].reshape(weights.shape))

class TFKerasEvaluator(Evaluator):

    def __init__(self, model:keras.Model, fitness_function:Callable[[List[keras.Model]], np.ndarray], individual_per_call:int=1, n_scores:int = 1) -> None:
        parameters = {"model":model.name, "fitness_function_name": fitness_function.__name__}
        super().__init__(n_scores=n_scores, individual_per_call=individual_per_call, other_parameters=parameters)

        self._model = model
        self._fitness_function = fitness_function
        self._individual_per_call = individual_per_call

        self._model_clones = [model]
        for _ in range(individual_per_call-1):
            self._model_clones.append(keras.models.clone_model(model))

    @property
    def descriptor(self):
        return self.get_descriptor()

    def get_descriptor(self) -> Descriptor:
        chromossome_sizes = []
        chromossome_ranges = []
        types = []
        names = []

        for weights in self._model.weights:
            chromossome_sizes.append(weights.shape.num_elements())
            chromossome_ranges.append((-1.0, 1.0))
            types.append(np.float32)
            names.append(weights.name)

        descriptor = Descriptor(chromossome_sizes, chromossome_ranges, types, names)
        return descriptor

    def _construct_models(self, individuals:np.ndarray) -> None:
        for i in range(self._individual_per_call):
            individual = individuals[i]
            model = self._model_clones[i]
            transfer_weights(individual, model)


    def __call__(self, population: np.ndarray) -> np.ndarray:
        n = population.shape[0]//self._individual_per_call

        fitness = np.empty((population.shape[0], self._n_scores), dtype=np.float64)

        for i in range(n):
            index = i*self._individual_per_call
            first = index
            last = index+self._individual_per_call

            self._construct_models(population[first:last])

            scores = self._fitness_function(self._model_clones)

            fitness[first:last] = np.asarray(scores).reshape((self._individual_per_call, self._n_scores))

        self._scores = fitness

        return fitness

class LossFitnessFunction:

    def __init__(self, loss:keras.losses.Loss, x:np.ndarray, y:np.ndarray, name:str="LossFitnessFunction"):
        self._loss = loss
        self._x = x
        self._y = y
        self.__name__ = name
    
    def __call__(self, models:List[keras.Model]) -> np.ndarray:
        model = models[0]

        if model.compiled_loss is None:
            model.compile(optimizer="sgd", loss=self._loss)

        prediction = model(self._x)

        score = self._loss(self._y, prediction)

        return -np.array(score)
        