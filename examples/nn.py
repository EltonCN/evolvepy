from typing import List
import numpy as np
import evolvepy as ep

def compute(individual:np.ndarray, x:np.ndarray) -> np.ndarray:
    '''
        Foward pass of a dense neural network

        Args:
            individual (np.ndarray): Individual with the weights.
            x (np.ndarray): Network input

        Returns:
            The output of the neural network
    '''
    result = x

    n_layer = len(individual.dtype.names)//2
    
    for i in range(n_layer-1):
        b = individual["layer"+str(i)+"b"]
        w = individual["layer"+str(i)+"w"].reshape((len(b), len(result)))

        result = w@result
        result += b
        result = (np.abs(result)+result)/2

    b = individual["layer"+str(n_layer-1)+"b"]
    w = individual["layer"+str(n_layer-1)+"w"].reshape((len(b), len(result)))

    result = (w@result)+b
    result = 1/(1+np.exp(-result)) 

    return result

def create_descriptor(input_size:int, output_size:int, units:List[int]):
    sizes = units
    sizes.append(output_size)

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

    return descriptor
