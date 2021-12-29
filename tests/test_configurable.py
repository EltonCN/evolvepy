import unittest

import numpy as np
from numpy.testing import assert_equal, assert_raises

from .utils import assert_not_equal

from evolvepy import Configurable

class TestConfigurable(unittest.TestCase):
    def test_dynamic_static(self):
        parameters = {"a":1, "b":2, "c":3}
        dynamic_parameters = {"a":False, "b":False}

        configurable = Configurable(parameters, dynamic_parameters)

        #print(configurable._static_parameter_names)

        assert_equal(configurable._static_parameter_names, ["c"])
        assert_equal(configurable._dynamic_parameter_names, ["a", "b"])
        assert_equal(configurable.dynamic_parameters, {"a":1, "b":2})
        assert_equal(configurable.static_parameters, {"c":3})

    def test_block(self):
        parameters = {"a":1, "b":2, "c":3}
        dynamic_parameters = {"a":True, "b":False}

        configurable = Configurable(parameters, dynamic_parameters)

        configurable.parameters = ("a", 20)
        configurable.parameters = ("b", 10)
        configurable.parameters = ("c", 0)

        assert_equal(configurable.parameters, {"a":20, "b":2, "c":3})