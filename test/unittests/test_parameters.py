"""
Tests of the parameters module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import unittest2 as unittest
except ImportError:
    import unittest
import operator
import numpy as np
from lazyarray import larray
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_equal
from pyNN.parameters import LazyArray, ParameterSpace, Sequence
from pyNN import random, errors
from .mocks import MockRNG


# test LazyArray
def test_create_with_int():
    A = LazyArray(3, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3

def test_create_with_float():
    A = LazyArray(3.0, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3.0

def test_create_with_list():
    A = LazyArray([1,2,3], shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(simplify=True), np.array([1,2,3]))

def test_create_with_array():
    A = LazyArray(np.array([1,2,3]), shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(simplify=True), np.array([1,2,3]))

def test_create_inconsistent():
    assert_raises(ValueError, LazyArray, [1,2,3], shape=4)

def test_create_with_invalid_string():
    A = LazyArray("d+2", shape=3)

def test_create_with_invalid_string():
    assert_raises(errors.InvalidParameterValueError, LazyArray, "x+y", shape=3)
    
def test_setitem_nonexpanded_same_value():
    A = LazyArray(3, shape=(5,))
    assert A.evaluate(simplify=True) == 3
    A[0] = 3
    assert A.evaluate(simplify=True) == 3

def test_setitem_invalid_value():
    A = LazyArray(3, shape=(5,))
    assert_raises(TypeError, A.__setitem__, "abc")

def test_setitem_nonexpanded_different_value():
    A = LazyArray(3, shape=(5,))
    assert A.evaluate(simplify=True) == 3
    A[0] = 4; A[4] = 5
    assert_array_equal(A.evaluate(simplify=True), np.array([4, 3, 3, 3, 5]))

def test_columnwise_iteration_with_flat_array():
    m = LazyArray(5, shape=(4,3)) # 4 rows, 3 columns
    cols = [col for col in m.by_column()]
    assert_equal(cols, [5, 5, 5])

def test_columnwise_iteration_with_structured_array():
    input = np.arange(12).reshape((4,3))
    m = LazyArray(input, shape=(4,3)) # 4 rows, 3 columns
    cols = [col for col in m.by_column()]    
    assert_array_equal(cols[0], input[:,0])
    assert_array_equal(cols[2], input[:,2])

def test_columnwise_iteration_with_random_array_parallel_safe_no_mask():
    orig_get_mpi_config = random.get_mpi_config
    random.get_mpi_config = lambda: (0, 2)
    input = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(parallel_safe=True))
    copy_input = random.RandomDistribution('normal', (0, 1), rng=MockRNG(parallel_safe=True))
    m = LazyArray(input, shape=(4,3))
    cols = [col for col in m.by_column()]
    assert_array_equal(cols[0], copy_input.next(4, mask_local=False))
    assert_array_equal(cols[1], copy_input.next(4, mask_local=False))
    assert_array_equal(cols[2], copy_input.next(4, mask_local=False))
    random.get_mpi_config = orig_get_mpi_config

def test_columnwise_iteration_with_function():
    input = lambda i,j: 2*i + j
    m = LazyArray(input, shape=(4,3))
    cols = [col for col in m.by_column()]
    assert_array_equal(cols[0], np.array([0, 2, 4, 6]))
    assert_array_equal(cols[1], np.array([1, 3, 5, 7]))
    assert_array_equal(cols[2], np.array([2, 4, 6, 8]))
    
def test_columnwise_iteration_with_flat_array_and_mask():
    m = LazyArray(5, shape=(4,3)) # 4 rows, 3 columns
    mask = np.array([True, False, True])
    cols = [col for col in m.by_column(mask=mask)]
    assert_equal(cols, [5, 5])
    
def test_columnwise_iteration_with_structured_array_and_mask():
    input = np.arange(12).reshape((4,3))
    m = LazyArray(input, shape=(4,3)) # 4 rows, 3 columns
    mask = np.array([False, True, True])
    cols = [col for col in m.by_column(mask=mask)]    
    assert_array_equal(cols[0], input[:,1])
    assert_array_equal(cols[1], input[:,2])

def test_columnwise_iteration_with_random_array_parallel_safe_with_mask():
    orig_get_mpi_config = random.get_mpi_config
    random.get_mpi_config = lambda: (0, 2)
    input = random.RandomDistribution('uniform', (0, 1), rng=MockRNG(parallel_safe=True))
    copy_input = random.RandomDistribution('gamma', (2, 3), rng=MockRNG(parallel_safe=True))
    m = LazyArray(input, shape=(4,3))
    mask = np.array([False, False, True])
    cols = [col for col in m.by_column(mask=mask)]
    assert_equal(len(cols), 1)
    assert_array_almost_equal(cols[0], copy_input.next(12, mask_local=False)[8:], 15)
    random.get_mpi_config = orig_get_mpi_config

def test_evaluate_with_flat_array():
    m = LazyArray(5, shape=(4,3))
    assert_array_equal(m.evaluate(), 5*np.ones((4,3)))

def test_evaluate_with_structured_array():
    input = np.arange(12).reshape((4,3))
    m = LazyArray(input, shape=(4,3))
    assert_array_equal(m.evaluate(), input)

def test_evaluate_with_functional_array():
    input = lambda i,j: 2*i + j
    m = LazyArray(input, shape=(4,3))
    assert_array_equal(m.evaluate(),
                        np.array([[0, 1, 2],
                                     [2, 3, 4],
                                     [4, 5, 6],
                                     [6, 7, 8]]))

def test_iadd_with_flat_array():
    m = LazyArray(5, shape=(4,3))
    m += 2
    assert_array_equal(m.evaluate(), 7*np.ones((4,3)))
    assert_equal(m.base_value, 5)
    assert_equal(m.evaluate(simplify=True), 7)

def test_add_with_flat_array():
    m0 = LazyArray(5, shape=(4,3))
    m1 = m0 + 2
    assert_equal(m1.evaluate(simplify=True), 7)
    assert_equal(m0.evaluate(simplify=True), 5)

def test_lt_with_flat_array():
    m0 = LazyArray(5, shape=(4,3))
    m1 = m0 < 10
    assert_equal(m1.evaluate(simplify=True), True)
    assert_equal(m0.evaluate(simplify=True), 5)
    
def test_lt_with_structured_array():
    input = np.arange(12).reshape((4,3))
    m0 = LazyArray(input, shape=(4,3))
    m1 = m0 < 5
    assert_array_equal(m1.evaluate(simplify=True), input < 5)
    
def test_structured_array_lt_array():
    input = np.arange(12).reshape((4,3))
    m0 = LazyArray(input, shape=(4,3))
    comparison = 5*np.ones((4,3))
    m1 = m0 < comparison
    assert_array_equal(m1.evaluate(simplify=True), input < comparison)

def test_multiple_operations_with_structured_array():
    input = np.arange(12).reshape((4,3))
    m0 = LazyArray(input, shape=(4,3))
    m1 = (m0 + 2) < 5
    m2 = (m0 < 5) + 2
    assert_array_equal(m1.evaluate(simplify=True), (input+2)<5)
    assert_array_equal(m2.evaluate(simplify=True), (input<5)+2)
    assert_array_equal(m0.evaluate(simplify=True), input)

def test_apply_function_to_constant_array():
    f = lambda m: 2*m + 3
    m0 = LazyArray(5, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_equal(m1.evaluate(simplify=True), 13)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m1.operations, [(operator.mul, 2), (operator.add, 3)])

def test_apply_function_to_structured_array():
    f = lambda m: 2*m + 3
    input = np.arange(12).reshape((4,3))
    m0 = LazyArray(input, shape=(4,3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_array_equal(m1.evaluate(simplify=True), input*2 + 3)

def test_apply_function_to_functional_array():
    input = lambda i,j: 2*i + j
    m0 = LazyArray(input, shape=(4,3))
    f = lambda m: 2*m + 3
    m1 = f(m0)
    assert_array_equal(m1.evaluate(),
                        np.array([[3, 5, 7],
                                     [7, 9, 11],
                                     [11, 13, 15],
                                     [15, 17, 19]]))

def test_add_two_constant_arrays():
    m0 = LazyArray(5, shape=(4,3))
    m1 = LazyArray(7, shape=(4,3))
    m2 = m0 + m1
    assert_equal(m2.evaluate(simplify=True), 12)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m2.base_value, m0.base_value)
    assert_equal(m2.operations, [(operator.add, m1)])
    
def test_add_incommensurate_arrays():
    m0 = LazyArray(5, shape=(4,3))
    m1 = LazyArray(7, shape=(5,3))
    assert_raises(ValueError, m0.__add__, m1)
    
def test_getitem_from_constant_array():
    m = LazyArray(3, shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4,0))
    assert_raises(IndexError, m.__getitem__, (2,-4))
    
def test_getitem_from_constant_array():
    m = LazyArray(3*np.ones((4,3)), shape=(4,3))
    assert m[0,0] == m[3,2] == m[-1,2] == m[-4,2] == m[2,-3] == 3
    assert_raises(IndexError, m.__getitem__, (4,0))
    assert_raises(IndexError, m.__getitem__, (2,-4))


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

    def test_columnwise_iteration_single_column(self):
        ps2d = ParameterSpace({'a': [[2, 3, 5, 8, 13], [21, 34, 55, 89, 144]],
                               'b': 7,
                               'c': lambda i, j: 3*i-2*j}, shape=(2, 5))
        ps2d.evaluate(mask=(slice(None), 3))
        expected = [{'a': np.array([8, 89]), 'b': np.array([7, 7]), 'c': np.array([-6, -3])}]
        actual = list(ps2d.columns())
        for x, y in zip(actual, expected):
            for key in y:
                assert_array_equal(x[key], y[key])

    def test_create_with_sequence(self):
        schema = {'a': Sequence}
        ps = ParameterSpace({'a': Sequence([1, 2, 3])},
                            schema,
                            shape=(2,))
        ps.evaluate()
        assert_array_equal(ps['a'], np.array([Sequence([1, 2, 3]), Sequence([1, 2, 3])], dtype=Sequence))

    def test_create_with_tuple(self):
        schema = {'a': Sequence}
        ps = ParameterSpace({'a': (1, 2, 3)},
                            schema,
                            shape=(2,))
        ps.evaluate()
        assert_array_equal(ps['a'], np.array([Sequence([1, 2, 3]), Sequence([1, 2, 3])], dtype=Sequence))
    
    def test_create_with_list_of_sequences(self):
        schema = {'a': Sequence}
        ps = ParameterSpace({'a': [Sequence([1, 2, 3]), Sequence([4, 5, 6])]},
                            schema,
                            shape=(2,))
        ps.evaluate()
        assert_array_equal(ps['a'], np.array([Sequence([1, 2, 3]), Sequence([4, 5, 6])], dtype=Sequence))

    def test_create_with_array_of_sequences(self):
        schema = {'a': Sequence}
        ps = ParameterSpace({'a': np.array([Sequence([1, 2, 3]), Sequence([4, 5, 6])], dtype=Sequence)},
                            schema,
                            shape=(2,))
        ps.evaluate()
        assert_array_equal(ps['a'], np.array([Sequence([1, 2, 3]), Sequence([4, 5, 6])], dtype=Sequence))

    def test_create_with_list_of_lists(self):
        schema = {'a': Sequence}
        ps = ParameterSpace({'a': [[1, 2, 3], [4, 5, 6]]},
                            schema,
                            shape=(2,))
        ps.evaluate()
        assert_array_equal(ps['a'], np.array([Sequence([1, 2, 3]), Sequence([4, 5, 6])], dtype=Sequence))  


if __name__ == "__main__":
    unittest.main()