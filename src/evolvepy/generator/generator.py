from typing import Dict, List, Tuple, Union
import numpy as np

from numpy.typing import ArrayLike, DTypeLike

from evolvepy.generator.layer import Layer

class Generator:

    def __init__(self, chromossome_sizes:ArrayLike, layers:Union[None, List[Layer]]=None, chromossome_ranges:Union[None, List[Union[None, Tuple]], Tuple]=None, types:Union[list, DTypeLike]=[np.float32], names:Union[list, str, None]=None, first_layer:Layer=None, last_layer:Layer=None):    
        chromossome_sizes = np.asarray(chromossome_sizes)

        if chromossome_sizes.shape == ():
            chromossome_sizes = np.array([chromossome_sizes])

        n_chromossome = len(chromossome_sizes)

        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        

        if chromossome_ranges is None:
            chromossome_ranges = [None] * n_chromossome
        elif isinstance(chromossome_ranges, tuple):
            chromossome_ranges = [chromossome_ranges]

        if not isinstance(types, list):
            types = [types]

        self._chromossome_sizes = chromossome_sizes
        self._n_chromossome = n_chromossome
        self._chromossome_ranges = chromossome_ranges
        
        self._create_dtype_names_ranges(names, types)

        self._connected = False

        if layers is None:
            layers = []
        elif first_layer is not None or last_layer is not None:
            raise ValueError("Generator 'layers' parameter must not be used together with 'first_layer' and 'last_layer'")
        else:
            for i in range(len(layers)-1):
                layers[i] = layers[i+1]

        if first_layer is not None and last_layer is not None:
            layers.append(first_layer)
            layers.append(last_layer)
        elif last_layer is not None or first_layer is not None:
            raise ValueError("You must set Generator 'first_layer' with 'last_layer'")

        self._layers = layers

        self._fitness = None
        self._population = None
        self._n_individual = None

    def _create_dtype_names_ranges(self, names, types):
        self._names = []

        dtype = []
        for i in range(self._n_chromossome):
            name = "chr"+str(i)
            if len(names)-1 >= i:
                name = names[i]
            self._names.append(name)

            size = np.atleast_1d(self._chromossome_sizes[i])
            size = tuple(size)

            dtype.append((name, types[i], size))

            if self._chromossome_ranges[i] is None:
                if np.dtype(types[i]).char in np.typecodes["AllFloat"]:
                    self._chromossome_ranges[i] = (0.0, 1.0)
                elif np.dtype(types[i]).char in np.typecodes["AllInteger"]:
                    self._chromossome_ranges[i] = (0, 10)

        self._dtype = np.dtype(dtype)

    def set_parameter(self, layer_name:str, parameter_name:str, value:object) -> None:
        for layer in self._layers:
            if layer.name == layer_name:
                layer.parameters = (parameter_name, value)

    def set_parameters(self, layer_name:str, parameters:Dict[str, object]) -> None:
        for layer in self._layers:
            if layer.name == layer_name:
                layer.parameters = parameters

    def get_parameter(self, layer_name:str, parameter_name:str=None) -> object:
        for layer in self._layers:
            if layer.name == layer_name:
                return layer.parameters[parameter_name]
    
    def get_parameters(self, layer_name:str) -> Dict[str, object]:
        for layer in self._layers:
            if layer.name == layer_name:
                return layer.parameters

    def generate_first(self, n_individual:int) -> np.ndarray:

        population = np.empty(n_individual, self._dtype)

        for i in range(self._n_chromossome):
            n_gene = self._chromossome_sizes[i]
            name = self._names[i]
            dtype = population[name].dtype
            shape = (n_individual, n_gene)

            chromossome_range = self._chromossome_ranges[i]

            if dtype.char in np.typecodes["AllFloat"]:
                population[name] = np.random.rand(n_individual, n_gene)
                population[name] *= chromossome_range[1] - chromossome_range[0]
                population[name] += chromossome_range[0]
            elif dtype.char in np.typecodes["AllInteger"]:
                population[name] = np.random.randint(chromossome_range[0], chromossome_range[1], shape)
            else:
                population[name] = np.random.choice([False, True], shape)

        return population
    
    def generate_evolve(self) -> np.ndarray:
        self._layers[0](self._population, self._fitness)
        
        if len(self._layers[-1].population) != self._n_individual:
            raise RuntimeError("The generator generated a population with wrong size")

        return self._layers[-1].population

    def add(self, layer:Layer) -> None:
        if len(self._layers) != 0:
            self._layers[-1].next = layer

        self._layers.append(layer)
    
    @property
    def fitness(self) -> np.ndarray:
        return self._fitness

    @fitness.setter
    def fitness(self, value:ArrayLike):
        self._fitness = np.asarray(value)

    def generate(self, n_individual:int=None) -> np.ndarray:
        if n_individual is not None:
            self._n_individual = n_individual
        
        if self._n_individual is None:
            raise RuntimeError("Generator generate must be called at least one time with n_individual parameter")

        if self._fitness is None or self._population is None:
            self._population = self.generate_first(n_individual)
        else:
            self._population = self.generate_evolve()

        return self._population