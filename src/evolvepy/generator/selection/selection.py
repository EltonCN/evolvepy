import numpy as np
from numpy.typing import ArrayLike
from numpy.random import choice
import numba

@numba.njit(fastmath=True)
def isin(val, arr):
    for i in range(arr.shape[0]):
        if arr[i] == val:
            return True
    return False

@numba.njit
def tournament(fitness_array:ArrayLike, n_selection:int) -> np.ndarray:
	'''Select the best individuals in a one vs one test.
		
		:param fitness_array: array with evaluated fitness from the generation
		:type fitness_array: np.typing.ArrayLike
		:param n_selection: number of individuals that will be selected
		:type n_selection: int

		:return: array with indexs of selected individuals
		:rtype: np.ndarray
	'''

	fitness_array = np.asarray(fitness_array)

	selected = np.zeros(n_selection, dtype=np.int32)-1
	size = fitness_array.shape[0]

	for i in range(n_selection):
		best_index = -1

		while best_index == -1 or (isin(best_index, selected)):
			index1 = np.random.randint(size)
			index2 = np.random.randint(size)

			if fitness_array[index1] > fitness_array[index2]:
				best_index = index1
			else:
				best_index = index2

		selected[i] = best_index

	return selected

@numba.njit
def roulette(fitness_array:ArrayLike, n_selection:int) -> np.ndarray:
	'''Select the best individuals stocaticaly with the probability of beign chosen
	equivalent to the individual fitness.
		
		:param fitness_array: array with evaluated fitness from the generation
		:type fitness_array: np.typing.ArrayLike
		:param n_selection: number of individuals that will be selected
		:type n_selection: int

		:return: array with indexs of selected individuals
		:rtype: np.ndarray
	'''

	fitness = np.asarray(fitness_array)
	if np.min(fitness) < 0:
		fitness = fitness - np.min(fitness)
	prob = fitness/np.sum(fitness)

	size = fitness.shape[0]	
	indexs = np.arange(0, size, 1, dtype=np.int32)

	# selected = np.random.choice(indexs, n_selection, p=probability) DON'T WORK WITH NJIT 

	cumsum = np.cumsum(prob)

	selected = np.empty(n_selection, np.int32)
	for i in range(n_selection):
		index = -1

		while index == -1 or (isin(index, selected)):
			index = indexs[np.searchsorted(cumsum, np.random.random(), side="right")]

		selected[i] = index
	#selected = np.searchsorted(cumsum, np.random.rand(n_selection), side="right")

	return selected

@numba.njit
def rank(fitness_array:ArrayLike, n_selection:int) -> np.ndarray:
    '''Select the n best individuals assuming the fitness_array is decreasingly sorted.
        
        :param fitness_array: array with evaluated fitness from the generation
        :type fitness_array: np.typing.ArrayLike
        :param n_selection: number of individuals that will be selected
        :type n_selection: int

        :return: array with indexs of selected individuals
        :rtype: np.ndarray
    '''

    return np.arange(0, n_selection, 1, dtype=np.int32)

tournament.needs_sort = False
roulette.needs_sort = True
rank.needs_sort = True


def default_selection():
	return tournament