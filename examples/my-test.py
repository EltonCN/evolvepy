import numpy as np

import matplotlib.pyplot as plt
import multiprocessing as mp
import evolvepy as ep

from nn import compute
import nn, gym

from abc import ABC, abstractmethod
from typing import Dict
from gym.wrappers.record_video import RecordVideo

from evolvepy.evaluator import ProcessFitnessFunction, FunctionEvaluator
from evolvepy.integrations.gym import GymFitnessFunction
   
class BipedalWalkerFitnessFunction(GymFitnessFunction):

    def __init__(self, show=False, save=False) -> None:
        '''
            BipedalWalkerFitnessFunction constructor.

            Args:
                show (bool): whether to show the graphical output of the environment.
                save (bool): whether to save the graphical output in a file.
                args (dict): can contain the key "time_mode", which indicates if the fitness should be the total amount 
                             of iterations performed in the environment.
        '''
        super().__init__(env_name = "BipedalWalker-v3",show=show, save=save)
    
    def behaviour(self, obs: object, individual: np.ndarray) -> object:
        action = compute(individual, obs)
        action = (2.0*action)-1.0
        return action

if __name__ == "__main__":

    bipedal_evaluator = BipedalWalkerFitnessFunction()

    evaluator = ep.evaluator.FunctionEvaluator(bipedal_evaluator.__call__, mode = FunctionEvaluator.PYTHON)
    multiple_evaluation = ep.evaluator.MultipleEvaluation(evaluator, 10, discard_max=True, discard_min=True)
    descriptor = nn.create_descriptor(input_size=24, output_size=4, units=[20,20])
    population_size = (100//mp.cpu_count())*mp.cpu_count()

    print(descriptor.dtype)

    first = ep.generator.Layer()
    combine = ep.generator.CombineLayer(ep.generator.selection.tournament, ep.generator.crossover.one_point)
    mutation = ep.generator.mutation.NumericMutationLayer(ep.generator.mutation.sum_mutation, 1.0, 0.9, (-0.5, 0.5))
    filter0 = ep.generator.FilterFirsts(int(np.floor(0.95*population_size)))
    sort = ep.generator.Sort()
    filter1 = ep.generator.FilterFirsts(int(np.ceil(0.05*population_size)))
    concat = ep.generator.Concatenate()

    first.next = combine
    combine.next = mutation
    mutation.next = filter0
    filter0.next = concat

    first.next = sort
    sort.next = filter1
    filter1.next = concat

    generator = ep.generator.Generator(first_layer=first, last_layer=concat, descriptor=descriptor)

    dyn_mut = ep.callbacks.DynamicMutation([mutation.name], refinement_patience=5, exploration_patience=5, refinement_steps=5)

    evolver = ep.Evolver(generator, multiple_evaluation, population_size, [dyn_mut])

    hist, last_pop = evolver.evolve(3)

    raise ValueError

    plt.plot(hist.max(axis=1))
    plt.plot(np.mean(hist, axis=1))
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution History")
    plt.legend(["Best", "Mean"])
    plt.show()

    best = last_pop[np.argmax(hist[-1])]
    test_evaluator = BipedalWalkerFitnessFunction(save=True)
    test_evaluator([best])