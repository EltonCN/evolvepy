import threading
import copy
from typing import Tuple

import numba
import random
import nvtx
import numpy as np
from numpy.typing import ArrayLike

N_FUNC = 50

def thread_func(f):
    f(5)

def func(a):
    return 2*a 


def thread_func2(f):
    c = np.empty(100, np.float32)
    f(c, 1, 0.75, (-10, 10))

def sum_mutation(chromosome:ArrayLike, existence_rate:float, gene_rate:float, mutation_range:Tuple[float, float]):
    '''
    It takes a chromosome and add a random value between the mutation range to its gene, then repeats the process with
    the given probability.

    Args:
        chromosome (np.ArrayLike): array of chromosomes
        existence_rate (float): probability of first mutation
        gene_rate (float): probability of another gene mutation
        mutation_range (Tuple[float, float]):

    Returns:
        new_cromosome (np.ArrayLike): new mutated individual
    '''
    chromosome = np.asarray(chromosome)
    new_chromosome = chromosome.copy()
    
    first = True
    count = 0
    if np.random.rand() < existence_rate:
        while (first or np.random.rand() < gene_rate) and count < chromosome.shape[0]:
            first = False

            index = np.random.randint(0, chromosome.shape[0])
            new_chromosome[index] = chromosome[index] + np.random.uniform(mutation_range[0], mutation_range[1])
            count += 1

    return new_chromosome



funcs = []
def generate_funcs(prefix):
    global funcs
    funcs = []
    for i in range(N_FUNC):
        a = random.random()
        f = lambda o : a*o #sum_mutation#func
        f.__name__ = prefix+"_func"#+str(i)
        f = nvtx.annotate()(numba.njit(nogil=True)(f))
        funcs.append(f)



def parallel_test():
    threads = []
    for f in funcs:
        thread = threading.Thread(target=thread_func, args=(f, ))
        #thread = threading.Thread(target=thread_func2, args=(f, ))
        threads.append(thread)

    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()

def serial_test():
    for f in funcs:
        f(5)
        #c = np.empty(100, np.float32)
        #f(c, 1, 0.75, (-10, 10))

nvtx.annotate("dummy")(numba.njit()(func))(5)

generate_funcs("serial")
with nvtx.annotate("serial"):
    serial_test()


generate_funcs("parallel")
with nvtx.annotate("parallel"):
    parallel_test()
