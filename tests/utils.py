from numpy.testing import assert_raises, assert_array_equal

def assert_not_equal(array1, array2):
    assert_raises(AssertionError, assert_array_equal, array1, array2)