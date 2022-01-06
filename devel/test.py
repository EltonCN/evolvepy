import gym
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import evolvepy as ep
import matplotlib.pyplot as plt

from evolvepy.integrations.tf_keras import ProcessTFKerasEvaluator, EvolutionaryModel

if __name__ == "__main__":

    model = EvolutionaryModel([keras.layers.Dense(24, activation="relu", input_shape=(24,)),
                            keras.layers.Dense(4, activation="sigmoid")])

    from rl_utils import BipedalWalkerFitnessFunction

    evaluator = ProcessTFKerasEvaluator(BipedalWalkerFitnessFunction, model, n_process=2)
    multiple_evaluation = ep.evaluator.MultipleEvaluation(evaluator, 5, discard_max=True, discard_min=True)

    population_size = 2

    first = ep.generator.Layer()
    combine = ep.generator.CombineLayer(ep.generator.selection.tournament, ep.generator.crossover.one_point)
    mutation = ep.generator.mutation.NumericMutationLayer(ep.generator.mutation.sum_mutation, 1.0, 0.5, (-0.5, 0.5))
    filter0 = ep.generator.FilterFirsts(int(np.floor(0.95*population_size)))
    sort = ep.generator.Sort()
    filter1 = ep.generator.FilterFirsts(int(np.ceil(0.05*population_size)))
    concat = ep.generator.Concatenate()

    first.next = combine
    combine.next = mutation
    combine.next = filter0
    filter0.next = concat

    first.next = sort
    sort.next = filter1
    filter1.next = concat

    generator = ep.generator.Generator(first_layer=first, last_layer=concat, descriptor=model.descriptor)

    from evolvepy.integrations.wandb import WandbLogger

    wandb_log = WandbLogger("BipedalWalker", "EvolvePy Example") 
    evolver = ep.Evolver(generator, multiple_evaluation, population_size) #[ wandb_log])

    del model
    keras.backend.clear_session()

    hist, last_pop = evolver.evolve(1)

    print(hist)