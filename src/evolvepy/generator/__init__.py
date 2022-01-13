'''
EvolvePy's callbacks. Objects that can be called upon during evolution to change its behavior.
'''
from .layer import Layer, ChromosomeOperator, Concatenate
from .basic_layers import FilterFirsts, Sort, Block, RandomPredation, ElitismLayer
from .combine import CombineLayer
from .generator import Generator
from .descriptor import Descriptor
from .firstgen import FirstGenLayer
from .context import Context

from . import mutation, selection, crossover