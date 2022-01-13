import unittest

import numpy as np
from numpy.testing import assert_equal
from numpy.testing._private.utils import assert_raises

from .utils import assert_not_equal

from evolvepy.generator.context import Context

class TestContext(unittest.TestCase):

    def test(self):
        dtype = np.dtype([("chr0", np.float32, 5), ("chr1", bool, 3)])
        
        context = Context(1, dtype.names)

        assert_equal(context.blocked, {"chr0":False, "chr1": False})
        assert_equal(context.chromosome_names, ["chr0", "chr1"])
        

        assert_equal(context.sorted, False)
        context.sorted = True
        assert_equal(context.sorted, True)

        assert_equal(context.have_value("custom_value"), False)
        context.custom_value = np.ones(5)
        assert_equal(context.have_value("custom_value"), True)
        assert_equal(context.custom_value, np.ones(5))

        context.blocked["chr0"] = True
        assert_equal(context.blocked, {"chr0":True, "chr1": False})

    def test_copy(self):
        dtype = np.dtype([("chr0", np.float32, 5), ("chr1", bool, 3)])
        
        context = Context(1, dtype.names)
        context.a = "a"

        copy = context.copy()

        assert_equal(list(context.blocked.values()), list(copy.blocked.values()))
        assert_equal(list(context.blocked.keys()), list(copy.blocked.keys()))

        assert_equal(list(context._values.values()), list(copy._values.values()))
        assert_equal(list(context._values.keys()), list(copy._values.keys()))

        assert_equal(context.sorted, copy.sorted)

        context.sorted = True

        assert_not_equal(context.sorted, copy.sorted)

        copy.b = "b"
        assert_not_equal(list(context._values.values()), list(copy._values.values()))
        assert_not_equal(list(context._values.keys()), list(copy._values.keys()))

        context.blocked["chr0"] = True
        assert_not_equal(list(context.blocked.values()), list(copy.blocked.values()))
