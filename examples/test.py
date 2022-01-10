

import gym
import numpy as np
import evolvepy as ep
import matplotlib.pyplot as plt

 
# # Evaluator
# 
# To speed up the individuals evaluation, we are going to use multiple process. Because of that, we need to define a class that will act on what function we evaluate, which, as Jupyter Notebook doesn't handle multiprocessing well, we need to define it in a separate file.
# 
# The [rl_utils.py](rl_utils.py) file contains:
# 
# - compute function: performs the calculation of the output of the neural network (we could use the tensorflow, but we are still working to understand how to correctly parallelize it)
# -  BipedalWalkerFitnessFunction: a ProcessFitnessFunction, evaluates the individual by making him control the agent through the environment, using the total reward obtained as fitness.

if __name__ == "__main__":
    from rl_utils import BipedalWalkerFitnessFunction

    evaluator = ep.evaluator.ProcessEvaluator(BipedalWalkerFitnessFunction)


    sizes = [20, 20 , 4] # Units of each layer
    input_size = 24 # Network input size (= observation size)

    names = []
    chr_sizes = []
    types = []
    ranges = []

    for i in range(3):
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

    population_size = 12

    
    # Let's define a generator with elitism:


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


    
    # To avoid getting stuck in some generation, we're also going to use dynamic mutation.


    dyn_mut = ep.callbacks.DynamicMutation([mutation.name], refinement_patience=5, exploration_patience=5, refinement_steps=5)

    evolver = ep.Evolver(generator, evaluator, population_size, [dyn_mut])

    
    # # Evolve and results
    # 
    # Let's evolve our generation:


    hist, last_pop = evolver.evolve(1)