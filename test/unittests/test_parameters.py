"""
Tests of the parameters module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyNN.parameters import ParameterSpace, Sequence
from pyNN.core import LazyArray


class ParameterSpaceTest(unittest.TestCase):

    def test_evaluate(self):
        ps = ParameterSpace({'a': [2, 3, 5, 8], 'b': 7, 'c': lambda i: 3*i+2}, shape=(4,))
        self.assertIsInstance(ps['c'], LazyArray)
        ps.evaluate()
        assert_array_equal(ps['c'], np.array([ 2,  5,  8, 11]))

    def test_evaluate_with_mask(self):
        ps = ParameterSpace({'a': [2, 3, 5, 8, 13], 'b': 7, 'c': lambda i: 3*i+2}, shape=(5,))
        ps.evaluate(mask=[1, 3, 4])
        expected = {'a': np.array([ 3,  8, 13]),
                    'c': np.array([ 5, 11, 14]),
                    'b': np.array([7, 7, 7])}
        for key in expected:
            assert_array_equal(expected[key], ps[key])

    def test_evaluate_with_mask_2D(self):
        ps2d = ParameterSpace({'a': [[2, 3, 5, 8, 13], [21, 34, 55, 89, 144]],
                               'b': 7,
                               'c': lambda i, j: 3*i-2*j}, shape=(2, 5))
        ps2d.evaluate(mask=(slice(None), [1, 3, 4]))
        assert_array_equal(ps2d['a'], np.array([[3, 8, 13], [34, 89, 144]]))
        assert_array_equal(ps2d['c'], np.array([[-2, -6, -8], [1, -3, -5]]))

    def test_iteration(self):
        ps = ParameterSpace({'a': [2, 3, 5, 8, 13], 'b': 7, 'c': lambda i: 3*i+2}, shape=(5,))
        ps.evaluate(mask=[1, 3, 4])
        self.assertEqual(list(ps),
                         [{'a': 3, 'c': 5, 'b': 7},
                          {'a': 8, 'c': 11, 'b': 7},
                          {'a': 13, 'c': 14, 'b': 7}])

    def test_iteration_items(self):
        ps = ParameterSpace({'a': [2, 3, 5, 8, 13], 'b': 7, 'c': lambda i: 3*i+2}, shape=(5,))
        ps.evaluate(mask=[1, 3, 4])
        expected = {'a': np.array([3,  8, 13]),
                    'c': np.array([5, 11, 14]),
                    'b': np.array([7, 7, 7])}
        for key, value in ps.items():
            assert_array_equal(expected[key], value)

    def test_columnwise_iteration(self):
        ps2d = ParameterSpace({'a': [[2, 3, 5, 8, 13], [21, 34, 55, 89, 144]],
                               'b': 7,
                               'c': lambda i, j: 3*i-2*j}, shape=(2, 5))
        ps2d.evaluate(mask=(slice(None), [1, 3, 4]))
        expected = [{'a': np.array([3, 34]), 'b': np.array([7, 7]), 'c': np.array([-2, 1])},
                    {'a': np.array([8, 89]), 'b': np.array([7, 7]), 'c': np.array([-6, -3])},
                    {'a': np.array([13, 144]), 'b': np.array([7, 7]), 'c': np.array([-8, -5])}]
        for x, y in zip(ps2d.columns(), expected):
            for key in y:
                assert_array_equal(x[key], y[key])


if __name__ == "__main__":
    unittest.main()