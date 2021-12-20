import unittest
import sys

import numpy as np

sys.path.append("..\src")

import evolvepy.generator.selection.selection as evselect

class TestSelection(unittest.TestCase):
    # TODO FIX roulette
    selection_operators = [evselect.tournament,  evselect.rank]#, evselect.roulette]

    n_individuals = 100

    def test_selection(self):

        fitness = np.sort(np.random.uniform(-100, 100, TestSelection.n_individuals))

        for operator in TestSelection.selection_operators:
            for i in range(1, 3):
                selections = operator(fitness, i)

                self.assertEqual(type(selections), np.ndarray) #Correct type
                self.assertEqual(selections.shape[0], i) #Correct shape


                #All indexs between valid range
                self.assertEqual(np.sum(
                    np.bitwise_and(0<=selections, selections<TestSelection.n_individuals)), 
                    i)

                values, counts = np.unique(selections, return_counts=True)

                self.assertEqual(np.sum(counts==1), i) #Don't selects same individuals
