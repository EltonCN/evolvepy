'''
EvolvePy's mutations. Objects that can be called upon during evolution to mutate the individual chromosome.
'''
from .numeric_mutation import sum_mutation, mul_mutation
from .binary_mutation import bit_mutation
from .mutation import default_mutation, NumericMutationLayer, BinaryMutationLayer