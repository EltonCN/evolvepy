def elitism(fitness_array:ArrayLike, n_selection:int) -> np.ndarray:
    '''Elitism selection
		This basic type of selection return the n_selection
		best individuals of the generation to stay the same
		in the next generation

		fitness_array: array with all the individuals
		n_selection: number of top individuals which will stay the same
    '''
    fitness_array = np.array(fitness_array)
    
    selected = np.zeros(n_selection, int)-1
    size = fitness_array.shape[0]

    for i in range(n_selection):
        while (best_index not in selected) and best_index == -1:
            index1 = np.random.randint(size)
            index2 = np.random.randint(size)

            if fitness_array[index1] > fitness_array[index2]:
                best_index = index1
            else:
                best_index = index2

        selected[i] = best_index