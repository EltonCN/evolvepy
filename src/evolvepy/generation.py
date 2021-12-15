import numpy as np
from numpy.typing import ArrayLike
from evolvepy.generator.mutation import default_mutation
from evolvepy.generator.crossover import default_crossover
from evolvepy.generator.selection.selection import default_selection

class Population:
    """
    Store the data from a generation of the experiment.

    :ivar names: Names of the chromossomes
    :type names: numpy.ndarray of strings
    
    :ivar mutation_op: Mutation operators
    :type mutation_op: list of functions

    :ivar mutation_params: Mutation parameters
    :type mutation_params: dict of dicts

    :ivar crossover_op: Crossover operators
    :type crossover_op: list of functions

    :ivar crossover_params: Crossover parameters
    :type crossover_params: dict of dicts

    :ivar selection_op: Selection operator
    :type selection_op: function

    :ivar dtype: individual's dtype
    :type dtype: numpy.dtype

    """

    def __init__(self, chromossome_sizes:ArrayLike, n_chromossome:int=1, types:list=[np.float32], names:list=[]):
        dtype = []

        chromossome_sizes = np.asarray(chromossome_sizes)

        self.names = []
        self.mutation_op = {}
        self.mutation_params = {}

        self.crossover_op = {}
        self.crossover_params = {}

        self.selection_op = default_selection()
        

        for i in range(n_chromossome):
            name = "chr"+str(i)
            if len(names)-1 >= i:
                name = names[i]
            self.names.append(name)

            size = np.atleast_1d(chromossome_sizes[i])
            size = tuple(size)

            dtype.append((name, types[i], size))

            self.mutation_op[name] = default_mutation(types[i])
            self.crossover_op[name] = default_crossover(types[i])

            self.mutation_params[name] = {"existence_rate":1.0, "gene_rate": 0.0, "range": (0.0, 1.0)}
            self.crossover_params[name] = {}


        self.names = np.asarray(self.names)

        self.dtype = np.dtype(dtype)
        self.population = None


    def set_mutation(self, chromossome_name="", chromossome_index=-1, operator=None, parameters=None):
        if chromossome_name == "":
            chromossome_name = self.names[chromossome_index]
        
        if operator is not None:
            self.mutation_op[chromossome_name] = operator

        if parameters is not None:
            self.mutation_params[chromossome_name] - parameters
    
    def set_crossover(self, chromossome_name="", chromossome_index=-1, operator=None, parameters=None):
        if chromossome_name == "":
            chromossome_name = self.names[chromossome_index]
        
        if operator is not None:
            self.crossover_op[chromossome_name] = operator

        if parameters is not None:
            self.crossover_params[chromossome_name] - parameters
        
    def set_selection(self, operator):
        self.selection_op = operator
