import threading
import multiprocessing as mp
from queue import Queue
from typing import List, Optional, Tuple, Set

import numpy as np

#job = [Layer, individuals, fitness]

def thread_function(job_queue:Queue, end_queue:Queue):
    '''
    Thread function to get and run jobs.

    Args:
        job_queue (Queue): Queue to receive jobs
        end_queue (Queue): Queue to mark job as done
    '''
    while True:
        job : Tuple[object, np.ndarray, np.ndarray, object] = job_queue.get()

        if job is None:
            break

        job[0](job[1], job[2], job[3])

        end_queue.put(job[0].name)

class ThreadPool:
    '''
    Manager for threads in a multithreaded generator.

    Should not be instantiated, all methods are static.
    '''

    initialized = False #If it is already initialized
    thread_pool : List[threading.Thread] = [] #List of all threads
    job_queue : Optional[Queue] = None #Queue to sending jobs to threads
    end_queue : Optional[Queue] = None #Queue to receiving finished jobs info
    n_thread : Optional[int] = None #Number of threads. If not setted is equal to number of cores.
    waiting : Optional[Set[str]] = None #Set of jobs waiting to get done
    serial = False #If should run the jobs without threads.

    @staticmethod
    def _initialize():
        '''
        Initialize the ThreadPool, creating the threads.
        '''
        if ThreadPool.initialized or ThreadPool.serial:
            return

        if ThreadPool.n_thread is None:
            ThreadPool.n_thread = mp.cpu_count()

        ThreadPool.job_queue = Queue()
        ThreadPool.end_queue = Queue()
        ThreadPool.waiting = set()

        for _ in range(ThreadPool.n_thread):
            thread = threading.Thread(target=thread_function, args=(ThreadPool.job_queue, ThreadPool.end_queue))
            thread.daemon = True
            thread.start()

            ThreadPool.thread_pool.append(thread)

        ThreadPool.initialized = True

    
    @staticmethod
    def _shutdown():
        '''
        Shutdown the ThreadPool, destroying the threads.
        '''
        if not ThreadPool.initialized:
            return

        ThreadPool.initialized = False

        with ThreadPool.job_queue.mutex:
            ThreadPool.job_queue.queue.clear()
        
        for _ in range(ThreadPool.n_thread):
            ThreadPool.job_queue.put(None)
        
        for thread in ThreadPool.thread_pool:
            thread.join()

        ThreadPool.thread_pool = []

    @staticmethod
    def add_job(job : Tuple[object, np.ndarray, np.ndarray, object]):
        '''
        Add a job to be run.

        Args:
            job (Tuple[Layer, np.ndarray, np.ndarray, Context]): Info needed to the job be runned.

        Raises:
            ValueError: If job does not have 4 elements.
        '''
        ThreadPool._initialize()

        if len(job) != 4:
            raise ValueError("Job must have 4 elements, but have {0}".format(len(job)))

        if ThreadPool.serial:
            job[0](job[1], job[2], job[3])
        else:
            ThreadPool.waiting.add(job[0].name)
            ThreadPool.job_queue.put(job)

    @staticmethod
    def wait_for_end():
        '''
        Blocks while waits all jobs to be completed
        '''
        if not ThreadPool.initialized:
            return
        
        while len(ThreadPool.waiting) != 0:
            ended_layer = ThreadPool.end_queue.get()
            ThreadPool.waiting.remove(ended_layer)