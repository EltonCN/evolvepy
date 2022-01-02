from typing import Callable, List, Optional, Union

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from evolvepy.evaluator import Evaluator
from evolvepy.generator import Descriptor, Generator
from evolvepy import Evolver
from evolvepy.callbacks import Callback

def transfer_weights(individual:np.ndarray, model:keras.Model) -> None:
    weights:tf.Variable
    for weights in model.weights:
        weights.assign(individual[weights.name].reshape(weights.shape))

def get_descriptor(model:keras.Model) -> Descriptor:
    chromossome_sizes = []
    chromossome_ranges = []
    types = []
    names = []

    for weights in model.weights:
        chromossome_sizes.append(weights.shape.num_elements())
        chromossome_ranges.append((-1.0, 1.0))
        types.append(np.float32)
        names.append(weights.name)

    descriptor = Descriptor(chromossome_sizes, chromossome_ranges, types, names)
    return descriptor

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
        return get_descriptor(self._model) 

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

        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x)
        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y)

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

class EvolutionaryModel(keras.Sequential):

    def __init__(self, layers=None, name=None):
        super().__init__(layers=layers, name=name)
        
        self._evolver = None

        

    @property
    def descriptor(self) -> Descriptor:
        return get_descriptor(self)

    def compile(self, generator:Generator=None, population_size:int=None, ep_callbacks:Optional[List[Callback]]=None, optimizer=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, steps_per_execution=None, **kwargs):
        evaluator = TFKerasEvaluator(self, self._fitness_function, 1, 1)
        
        
        if generator is not None and population_size is not None:
            self._evolver = Evolver(generator, evaluator, population_size, ep_callbacks)

        return super().compile(run_eagerly=True, optimizer="sgd", loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, steps_per_execution=steps_per_execution, **kwargs)

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        self._x = x
        self._y = y
        self._sample_weight = sample_weight

        hist, last_pop = self._evolver.evolve(1)

        best = last_pop[np.argmax(hist[-1])]
        transfer_weights(best, self)

        y_pred = self(x, training=True)

        self.compiled_loss.reset_state()
        self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def _fitness_function(self, model:List[keras.Model]=None) -> np.ndarray:
        y_pred = self(self._x, training=True)

        self.compiled_loss.reset_state()

        loss = self.compiled_loss(self._y, y_pred, sample_weight=self._sample_weight, regularization_losses=self.losses)
        self._debug_loss = loss
        return -np.array(loss)
