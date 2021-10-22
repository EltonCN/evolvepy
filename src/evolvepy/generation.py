from numba.types.npytypes import Array
import numpy as np
from numpy.typing import ArrayLike
from evolvepy.generator.mutation import default_mutation
from evolvepy.generator.crossover import default_crossover
from evolvepy.generator.selection.selection import default_selection

class Generation:
    def __init__(self, chromossome_sizes:ArrayLike, n_chromossome:int=1, types:list=[np.float32], names:list=[]):
        dtype = []

        chromossome_sizes = np.asarray(chromossome_sizes)

        self.names = []
        self.mutation_op = []
        self.mutation_params = []

        self.crossover_op = []
        self.crossover_params = []

        self.selection_op = default_selection()
        

        for i in range(n_chromossome):
            name = "chr"+str(i)
            if len(names)-1 >= i:
                name = names[i]
            self.names.append(name)

            size = np.atleast_1d(chromossome_sizes[i])
            size = tuple(size)

            dtype.append((name, types[i], size))

            self.mutation_op.append(default_mutation(types[i]))
            self.crossover_op.append(default_crossover(types[i]))

            self.mutation_params.append({"existence_rate":1.0, "gene_rate": 0.0, "range": (0.0, 1.0)})
            self.crossover_params.append({})

        self.dtype = np.dtype(dtype)
        self.population = None