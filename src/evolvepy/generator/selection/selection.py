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

		while (isin(best_index, selected)) or best_index == -1:
			index1 = np.random.randint(size)
			index2 = np.random.randint(size)

			if fitness_array[index1] > fitness_array[index2]:
				best_index = index1
			else:
				best_index = index2

		selected[i] = best_index

	return selected

#@numba.njit
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
	selected = np.zeros(n_selection, dtype=np.int32)-1
	size = fitness.shape[0]
	sum = 0
	for i in range(size):
		sum += i
	standard_prob = 1/sum
	selected = choice(np.arange(0, size, 1, dtype=np.int32), n_selection, p=[standard_prob * (size -1 - i) for i in range(size)])
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