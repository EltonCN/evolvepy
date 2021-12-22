import unittest

import numpy as np
from numpy.testing import assert_equal

from .utils import assert_not_equal

from evolvepy.generator.context import Context

class TestContext(unittest.TestCase):

    def test(self):
        dtype = np.dtype([("chr0", np.float32, 5), ("chr1", bool, 3)])
        
        context = Context(dtype.names)

        assert_equal(context.blocked, {"chr0":False, "chr1": False})
        assert_equal(context.chromossome_names, ["chr0", "chr1"])
        

        assert_equal(context.sorted, False)
        context.sorted = True
        assert_equal(context.sorted, True)

        assert_equal(context.have_value("custom_value"), False)
        context.custom_value = np.ones(5)
        assert_equal(context.have_value("custom_value"), True)
        assert_equal(context.custom_value, np.ones(5))

        context.blocked["chr0"] = True
        assert_equal(context.blocked, {"chr0":True, "chr1": False})