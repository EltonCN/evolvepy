'''
EvolvePy's evaluators. Objects used to evaluate individuals.
'''

from .evaluator import Evaluator
from .multiple import MultipleEvaluation
from .aggregator import FitnessAggregator
from .function_evaluator import FunctionEvaluator
from .cache import FitnessCache
from .process_evaluator import ProcessEvaluator, ProcessFitnessFunction