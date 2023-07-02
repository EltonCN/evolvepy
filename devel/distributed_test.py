from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import ray, gym, nn

import evolvepy as ep
from evolvepy.evaluator import ProcessFitnessFunction
from evolvepy.integrations.ray import DistributedEvaluator
from evolvepy.integrations.gym import GymFitnessFunction

from nn import compute

# ActorDispatcher: manda os indivíduos para os atores, e recebe os scores
# Actor = Runner: recebe os indivíduos, instanci
# ProcessFitnessFunction


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
    ray.shutdown()
    ray.init()
    
    evaluator = DistributedEvaluator(fitness_function = BipedalWalkerFitnessFunction)
    multiple_evaluation = ep.evaluator.MultipleEvaluation(evaluator, 10, discard_max=True, discard_min=True)
    descriptor = nn.create_descriptor(input_size=24, output_size=4, units=[20,20])
    population_size = (100//mp.cpu_count())*mp.cpu_count()
    population_size = 12


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
    print("iniciou evolver")
    evolver = ep.Evolver(generator, multiple_evaluation, population_size, [dyn_mut])

    print("iniciou evolver.evolve")
    hist, last_pop = evolver.evolve(3)
