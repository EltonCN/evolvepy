from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union, Type
import warnings

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from evolvepy.evaluator import Evaluator
from evolvepy.evaluator.process_evaluator import ProcessEvaluator, ProcessFitnessFunction
from evolvepy.generator import Descriptor, Generator
from evolvepy import Evolver
from evolvepy.callbacks import Callback

def transfer_weights(individual:np.ndarray, model:keras.Model) -> None:
    '''
    Transfer weights from individual to model.

    Args:
        individual (np.ndarray): EvolvePy's individual. Chromosomes must contains. layers weights and biases.
                                 Chromosomes names must be model's weights names (model.weights[i].name), or 
                                 in same sequence of model's weights, with "weights+str(i)" name (weights0, weights1, ...).
        model (keras.Model): [description]
    '''
    if len(individual.dtype.names[0]) >7  and individual.dtype.names[0][:7] == "weights": 
        index = 0
        
        weights:tf.Variable
        for weights in model.weights:
            weights.assign(individual["weights"+str(index)].reshape(weights.shape))
            index += 1
    else:    
    
        weights:tf.Variable
        for weights in model.weights:
            weights.assign(individual[weights.name].reshape(weights.shape))

def get_descriptor(model:keras.Model) -> Descriptor:
    '''
    Creates a descriptor from a keras Model.

    Args:
        model (keras.Model): Model to create descriptor.

    Returns:
        Descriptor: Descriptor for individuals containing the model weights and bias.
    '''
    chromossome_sizes = []
    chromossome_ranges = []
    types = []
    names = []

    index = 0

    for weights in model.weights:
        chromossome_sizes.append(weights.shape.num_elements())
        chromossome_ranges.append((-1.0, 1.0))
        types.append(np.float32)
        names.append("weights"+str(index))
        index += 1

    descriptor = Descriptor(chromossome_sizes, chromossome_ranges, types, names)
    return descriptor

class TFKerasEvaluator(Evaluator):
    '''
    Evaluates individuals representing TFKeras models with a arbitrary function.
    '''

    def __init__(self, model:keras.Model, fitness_function:Callable[[List[keras.Model]], np.ndarray], individual_per_call:int=1, n_scores:int = 1) -> None:
        '''
        TFKerasEvaluator constructor.

        Args:
            model (keras.Model): Model to evaluate.
            fitness_function (Callable[[List[keras.Model]], np.ndarray]): Fitness function to evaluate the individuals.
            individual_per_call (int, optional): Number of individuals that the fitness function receives. Defaults to 1.
            n_scores (int, optional): Number of scores that the fitness function generates.. Defaults to 1.
        '''
        parameters = {"model":model.name, "fitness_function_name": fitness_function.__name__}
        super().__init__(n_scores=n_scores, individual_per_call=individual_per_call, other_parameters=parameters)

        self._model = model
        self._fitness_function = fitness_function
        self._individual_per_call = individual_per_call

        self._model_clones = [model]
        for _ in range(individual_per_call-1):
            self._model_clones.append(keras.models.clone_model(model))

    @property
    def descriptor(self) -> Descriptor:
        '''
        Returns the descriptor of an individual that represents the model weights.

        Returns:
            Descriptor: The generated descriptor.
        '''
        return get_descriptor(self._model) 

    def _construct_models(self, individuals:np.ndarray) -> None:
        '''
        Transfers the weights of individuals to the models.

        Args:
            individuals (np.ndarray): [description]
        '''
        for i in range(self._individual_per_call):
            individual = individuals[i]
            model = self._model_clones[i]
            transfer_weights(individual, model)


    def __call__(self, population: np.ndarray) -> np.ndarray:
        '''
        Evaluates a population.

        Args:
            population (np.ndarray): Population to be evaluated.

        Returns:
            np.ndarray: Population fitness.
        '''
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
    '''
    Fitness function to evaluate models with a keras loss function.

    Prefer to use an EvolutionaryModel to use other tensorflow/keras features such as batching and validation.
    '''

    def __init__(self, loss:keras.losses.Loss, x:Union[np.ndarray, tf.Tensor], y:Union[np.ndarray, tf.Tensor], name:str="LossFitnessFunction"):
        '''
        LossFitnessFunction constructor.

        Args:
            loss (keras.losses.Loss): The loss to evaluate the individuals.
            x (Union[np.ndarray, tf.Tensor]): Train features.
            y (Union[np.ndarray, tf.Tensor]): Train targets.
            name (str, optional): Fitness function name. Defaults to "LossFitnessFunction".
        '''
        self._loss = loss

        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x)
        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y)

        self._x = x
        self._y = y
        self.__name__ = name
    
    def __call__(self, models:List[keras.Model]) -> np.ndarray:
        '''
        Evaluates one model.

        Args:
            models (List[keras.Model]): Model to evaluate. Only evaluates the first model in the list.

        Returns:
            np.ndarray: The negative of the model loss
        '''
        model = models[0]

        if model.compiled_loss is None:
            model.compile(optimizer="sgd", loss=self._loss)

        prediction = model(self._x)

        score = self._loss(self._y, prediction)

        return -np.array(score)

class EvolutionaryModel(keras.Sequential):
    '''
    Model class to optimize using EvolvePy.

    Each generation will evolve by one batch (total generations = epochs*batches)
    '''

    def __init__(self, layers=None, name=None):
        '''
        EvolutionaryModel constructor.

        Args:
            layers ([type], optional): Sequential layers of the model. Defaults to None.
            name ([type], optional): Model name. Defaults to None.
        '''

        super().__init__(layers=layers, name=name)
        
        self._evolver = None

        

    @property
    def descriptor(self) -> Descriptor:
        '''
        Returns the descriptor of an individual that represents the model weights.

        Returns:
            Descriptor: The generated descriptor.
        '''
        return get_descriptor(self)

    def compile(self, generator:Generator=None, population_size:int=None, ep_callbacks:Optional[List[Callback]]=None, loss=None, metrics=None, loss_weights=None, weighted_metrics=None, steps_per_execution=None, **kwargs):
        '''
        Compiles the model.

        Undescribed arguments must be used in the same way as the tf/keras Model base class.

        Args:
            generator (Generator, optional): Individual generator to optimize the model. Defaults to None (optimize won't work).
            population_size (int, optional): Population size for the generator. Defaults to None.
            ep_callbacks (Optional[List[Callback]], optional): EvolvePy's callbacks to use during optimization. Defaults to None.
        '''
        evaluator = TFKerasEvaluator(self, self._fitness_function, 1, 1)
        
        
        if generator is not None and population_size is not None:
            self._evolver = Evolver(generator, evaluator, population_size, ep_callbacks)

        super().compile(run_eagerly=True, optimizer="sgd", loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, steps_per_execution=steps_per_execution, **kwargs)

    def train_step(self, data):
        '''
        Evolves the model one step. If you use multiple batches, each generation will evolve by one batch.

        Arguments and returns are the same of tf/keras Model train_step.
        '''
        super().train_step()
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
        '''
        Fitness function for the evolver, computes the actual batch loss.

        Args:
            model (List[keras.Model], optional): Model to evaluate. It's not used, it's here for compatibility only, 
                                                 as the TFKeras evaluator will transfer the weights to this model instance. 
                                                 Defaults to None.

        Returns:
            np.ndarray: Model's batch loss.
        '''
        y_pred = self(self._x, training=True)

        self.compiled_loss.reset_state()

        loss = self.compiled_loss(self._y, y_pred, sample_weight=self._sample_weight, regularization_losses=self.losses)
        self._debug_loss = loss
        return -np.array(loss)


class ProcessTFKerasFitnessFunction(ProcessFitnessFunction):
    '''
    TFKeras Fitness Function for multiprocessing.

    It's under development, it doesn't work properly.

    Args:
        ProcessFitnessFunction ([type]): [description]
    '''
    def __init__(self, args:Dict, reset:bool=False) -> None:
        warnings.warn("Feature not yet fully implemented, is high on memory. Not recommended to use.", RuntimeWarning)
        
        super().__init__(reset=reset, args=args)
        
        config = args
        self._model = EvolutionaryModel.from_config(config)
        
        self._model.compile("sgd", keras.losses.MeanSquaredError())
    
    def __call__(self, individuals: np.ndarray) -> np.ndarray:
        individual = individuals[0]
        transfer_weights(individual, self._model)

        return super().__call__(self._model)
    
    @abstractmethod
    def evaluate(self, model: keras.Model) -> np.ndarray:
        ...

class ProcessTFKerasEvaluator(ProcessEvaluator):

    def __init__(self, fitness_function: Type[ProcessFitnessFunction], model:EvolutionaryModel, n_process: int = None, timeout: int = None, n_scores: int = 1, individual_per_call: int = 1) -> None:
        warnings.warn("Feature not yet fully implemented, is high on memory. Not recommended to use.", RuntimeWarning)
        super().__init__(fitness_function, n_process=n_process, timeout=timeout, n_scores=n_scores, individual_per_call=individual_per_call, args=model.get_config())