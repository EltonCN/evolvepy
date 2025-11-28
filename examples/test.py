#from mlagents_envs.environment import UnityEnvironment
#from gym_unity.envs import UnityToGymWrapper


import evolvepy as ep
import numpy as np
from matplotlib import pyplot as plt
#import unity_utils as uutils


if __name__ == "__main__":

    args = {"env_path": "D:\\Github\\ml-agents\\builds\\3dball_single\\UnityEnvironment.exe"}
    def fitness_function(individuals):
        individual = individuals[0]["chr0"][0] 

        score = 0

        if individual < 500:
            score = individual
        elif individual < 1000:
            score = 1000 - individual
        elif individual < 2000:
            score = individual - 1000
        elif individual < 3000:
            score = 3000 - individual
        elif individual < 4000:
            score = 2.0*(individual - 3000)
        else:
            score = 2.0*(5000-individual)

        return score
    evaluator = ep.evaluator.ProcessEvaluator(fitness_function=)




    sizes = [20, 20,  2] # Units of each layer
    input_size = 8 # Network input size (= observation size)

    names = []
    chr_sizes = []
    types = []
    ranges = []

    for i in range(len(sizes)):
        total_weights = input_size*sizes[i]

        names.append("layer"+str(i)+"w")
        names.append("layer"+str(i)+"b")

        chr_sizes.append(total_weights)
        chr_sizes.append(sizes[i])

        ranges.append((-1.0, 1.0))
        ranges.append((-1.0, 1.0))

        types.append(np.float32)
        types.append(np.float32)

        input_size = sizes[i]

    descriptor = ep.generator.Descriptor(chr_sizes, ranges, types, names)


    import multiprocessing as mp

    population_size = (100//mp.cpu_count())*mp.cpu_count()
    population_size = 12


    first = ep.generator.Layer()
    sort = ep.generator.Sort()

    first.next = sort

    concat = ep.generator.Concatenate()

    predation = ep.generator.RandomPredation(int(0.75*population_size))
    combine = ep.generator.CombineLayer(ep.generator.selection.tournament, ep.generator.crossover.one_point)
    mutation = ep.generator.mutation.NumericMutationLayer(ep.generator.mutation.sum_mutation, 1.0, 0.9, (-0.5, 0.5))
    filter0 = ep.generator.FilterFirsts(int(np.floor(0.95*population_size)))

    sort.next = predation
    predation.next = combine
    combine.next = mutation
    mutation.next = filter0
    filter0.next = concat

    filter1 = ep.generator.FilterFirsts(int(np.ceil(0.05*population_size)))

    sort.next = filter1
    filter1.next = concat

    generator = ep.generator.Generator(first_layer=first, last_layer=concat, descriptor=descriptor)

    dyn_mut = ep.callbacks.DynamicMutation([mutation.name], refinement_patience=5, exploration_patience=5, refinement_steps=5)

    evolver = ep.Evolver(generator, evaluator, population_size, [dyn_mut])

    evolver.evolve(1)





