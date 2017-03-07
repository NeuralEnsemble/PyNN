# encoding: utf-8
"""
Unit tests for ``larray`` class

Copyright Andrew P. Davison and Joël Chavas, 2012-2014
"""

from lazyarray import larray, VectorizedIterable, sqrt, partial_shape
import numpy
from nose.tools import assert_raises, assert_equal
from nose import SkipTest
from numpy.testing import assert_array_equal, assert_array_almost_equal
import operator
from copy import deepcopy


class MockRNG(VectorizedIterable):

    def __init__(self, start, delta):
        self.start = start
        self.delta = delta

    def next(self, n):
        s = self.start
        self.start += n * self.delta
        return s + self.delta * numpy.arange(n)


# test larray
def test_create_with_int():
    A = larray(3, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3


def test_create_with_int_and_dtype():
    A = larray(3, shape=(5,), dtype=float)
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3


def test_create_with_float():
    A = larray(3.0, shape=(5,))
    assert A.shape == (5,)
    assert A.evaluate(simplify=True) == 3.0


def test_create_with_list():
    A = larray([1, 2, 3], shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), numpy.array([1, 2, 3]))


def test_create_with_array():
    A = larray(numpy.array([1, 2, 3]), shape=(3,))
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), numpy.array([1, 2, 3]))


def test_create_with_array_and_dtype():
    A = larray(numpy.array([1, 2, 3]), shape=(3,), dtype=int)
    assert A.shape == (3,)
    assert_array_equal(A.evaluate(), numpy.array([1, 2, 3]))


def test_create_with_generator():
    def plusone():
        i = 0
        while True:
            yield i
            i += 1
    A = larray(plusone(), shape=(5, 11))
    assert_array_equal(A.evaluate(),
                       numpy.arange(55).reshape((5, 11)))


def test_create_with_function1D():
    A = larray(lambda i: 99 - i, shape=(3,))
    assert_array_equal(A.evaluate(),
                       numpy.array([99, 98, 97]))


def test_create_with_function1D_and_dtype():
    A = larray(lambda i: 99 - i, shape=(3,), dtype=float)
    assert_array_equal(A.evaluate(),
                       numpy.array([99.0, 98.0, 97.0]))


def test_create_with_function2D():
    A = larray(lambda i, j: 3 * j - 2 * i, shape=(2, 3))
    assert_array_equal(A.evaluate(),
                       numpy.array([[0, 3, 6],
                                    [-2, 1, 4]]))


def test_create_inconsistent():
    assert_raises(ValueError, larray, [1, 2, 3], shape=4)


def test_create_with_string():
    assert_raises(TypeError, larray, "123", shape=3)


def test_create_with_larray():
    A = 3 + larray(lambda i: 99 - i, shape=(3,))
    B = larray(A, shape=(3,), dtype=int)
    assert_array_equal(B.evaluate(),
                       numpy.array([102, 101, 100]))

# def test_columnwise_iteration_with_flat_array():
# m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]
#    assert_equal(cols, [5, 5, 5])
#
# def test_columnwise_iteration_with_structured_array():
#    input = numpy.arange(12).reshape((4,3))
# m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], input[:,0])
#    assert_array_equal(cols[2], input[:,2])
#
# def test_columnwise_iteration_with_function():
#    input = lambda i,j: 2*i + j
#    m = larray(input, shape=(4,3))
#    cols = [col for col in m.by_column()]
#    assert_array_equal(cols[0], numpy.array([0, 2, 4, 6]))
#    assert_array_equal(cols[1], numpy.array([1, 3, 5, 7]))
#    assert_array_equal(cols[2], numpy.array([2, 4, 6, 8]))
#
# def test_columnwise_iteration_with_flat_array_and_mask():
# m = larray(5, shape=(4,3)) # 4 rows, 3 columns
#    mask = numpy.array([True, False, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert_equal(cols, [5, 5])
#
# def test_columnwise_iteration_with_structured_array_and_mask():
#    input = numpy.arange(12).reshape((4,3))
# m = larray(input, shape=(4,3)) # 4 rows, 3 columns
#    mask = numpy.array([False, True, True])
#    cols = [col for col in m.by_column(mask=mask)]
#    assert_array_equal(cols[0], input[:,1])
#    assert_array_equal(cols[1], input[:,2])


def test_size_related_properties():
    m1 = larray(1, shape=(9, 7))
    m2 = larray(1, shape=(13,))
    m3 = larray(1)
    assert_equal(m1.nrows, 9)
    assert_equal(m1.ncols, 7)
    assert_equal(m1.size, 63)
    assert_equal(m2.nrows, 13)
    assert_equal(m2.ncols, 1)
    assert_equal(m2.size, 13)
    assert_raises(ValueError, lambda: m3.nrows)
    assert_raises(ValueError, lambda: m3.ncols)
    assert_raises(ValueError, lambda: m3.size)


def test_evaluate_with_flat_array():
    m = larray(5, shape=(4, 3))
    assert_array_equal(m.evaluate(), 5 * numpy.ones((4, 3)))


def test_evaluate_with_structured_array():
    input = numpy.arange(12).reshape((4, 3))
    m = larray(input, shape=(4, 3))
    assert_array_equal(m.evaluate(), input)


def test_evaluate_with_functional_array():
    input = lambda i, j: 2 * i + j
    m = larray(input, shape=(4, 3))
    assert_array_equal(m.evaluate(),
                       numpy.array([[0, 1, 2],
                                    [2, 3, 4],
                                    [4, 5, 6],
                                    [6, 7, 8]]))


def test_evaluate_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7, 3))
    assert_array_equal(m.evaluate(),
                       numpy.arange(21).reshape((7, 3)))


def test_evaluate_twice_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m1 = larray(input, shape=(7, 3)) + 3
    m2 = larray(input, shape=(7, 3)) + 17
    assert_array_equal(m1.evaluate(),
                       numpy.arange(3, 24).reshape((7, 3)))
    assert_array_equal(m2.evaluate(),
                       numpy.arange(38, 59).reshape((7, 3)))


def test_evaluate_structured_array_size_1_simplify():
    m = larray([5.0], shape=(1,))
    assert_equal(m.evaluate(simplify=True), 5.0)
    n = larray([2.0], shape=(1,))
    assert_equal((m/n).evaluate(simplify=True), 2.5)


def test_iadd_with_flat_array():
    m = larray(5, shape=(4, 3))
    m += 2
    assert_array_equal(m.evaluate(), 7 * numpy.ones((4, 3)))
    assert_equal(m.base_value, 5)
    assert_equal(m.evaluate(simplify=True), 7)


def test_add_with_flat_array():
    m0 = larray(5, shape=(4, 3))
    m1 = m0 + 2
    assert_equal(m1.evaluate(simplify=True), 7)
    assert_equal(m0.evaluate(simplify=True), 5)


def test_lt_with_flat_array():
    m0 = larray(5, shape=(4, 3))
    m1 = m0 < 10
    assert_equal(m1.evaluate(simplify=True), True)
    assert_equal(m0.evaluate(simplify=True), 5)


def test_lt_with_structured_array():
    input = numpy.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = m0 < 5
    assert_array_equal(m1.evaluate(simplify=True), input < 5)


def test_structured_array_lt_array():
    input = numpy.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    comparison = 5 * numpy.ones((4, 3))
    m1 = m0 < comparison
    assert_array_equal(m1.evaluate(simplify=True), input < comparison)


def test_rsub_with_structured_array():
    m = larray(numpy.arange(12).reshape((4, 3)))
    assert_array_equal((11 - m).evaluate(),
                       numpy.arange(11, -1, -1).reshape((4, 3)))


def test_inplace_mul_with_structured_array():
    m = larray((3 * x for x in range(4)), shape=(4,))
    m *= 7
    assert_array_equal(m.evaluate(),
                       numpy.arange(0, 84, 21))


def test_abs_with_structured_array():
    m = larray(lambda i, j: i - j, shape=(3, 4))
    assert_array_equal(abs(m).evaluate(),
                       numpy.array([[0, 1, 2, 3],
                                    [1, 0, 1, 2],
                                    [2, 1, 0, 1]]))


def test_multiple_operations_with_structured_array():
    input = numpy.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = (m0 + 2) < 5
    m2 = (m0 < 5) + 2
    assert_array_equal(m1.evaluate(simplify=True), (input + 2) < 5)
    assert_array_equal(m2.evaluate(simplify=True), (input < 5) + 2)
    assert_array_equal(m0.evaluate(simplify=True), input)


def test_multiple_operations_with_functional_array():
    m = larray(lambda i: i, shape=(5,))
    m0 = m / 100.0
    m1 = 0.2 + m0
    assert_array_almost_equal(m0.evaluate(), numpy.array([0.0, 0.01, 0.02, 0.03, 0.04]), decimal=12)
    assert_array_almost_equal(m1.evaluate(), numpy.array([0.20, 0.21, 0.22, 0.23, 0.24]), decimal=12)
    assert_equal(m1[0], 0.2)


def test_operations_combining_constant_and_structured_arrays():
    m0 = larray(10, shape=(5,))
    m1 = larray(numpy.arange(5))
    m2 = m0 + m1
    assert_array_almost_equal(m2.evaluate(), numpy.arange(10, 15))


def test_apply_function_to_constant_array():
    f = lambda m: 2 * m + 3
    m0 = larray(5, shape=(4, 3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_equal(m1.evaluate(simplify=True), 13)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m1.operations, [(operator.mul, 2), (operator.add, 3)])


def test_apply_function_to_structured_array():
    f = lambda m: 2 * m + 3
    input = numpy.arange(12).reshape((4, 3))
    m0 = larray(input, shape=(4, 3))
    m1 = f(m0)
    assert isinstance(m1, larray)
    assert_array_equal(m1.evaluate(simplify=True), input * 2 + 3)


def test_apply_function_to_functional_array():
    input = lambda i, j: 2 * i + j
    m0 = larray(input, shape=(4, 3))
    f = lambda m: 2 * m + 3
    m1 = f(m0)
    assert_array_equal(m1.evaluate(),
                       numpy.array([[3, 5, 7],
                                    [7, 9, 11],
                                    [11, 13, 15],
                                    [15, 17, 19]]))


def test_add_two_constant_arrays():
    m0 = larray(5, shape=(4, 3))
    m1 = larray(7, shape=(4, 3))
    m2 = m0 + m1
    assert_equal(m2.evaluate(simplify=True), 12)
    # the following tests the internals, not the behaviour
    # it is just to check I understand what's going on
    assert_equal(m2.base_value, m0.base_value)
    assert_equal(m2.operations, [(operator.add, m1)])


def test_add_incommensurate_arrays():
    m0 = larray(5, shape=(4, 3))
    m1 = larray(7, shape=(5, 3))
    assert_raises(ValueError, m0.__add__, m1)


def test_getitem_from_2D_constant_array():
    m = larray(3, shape=(4, 3))
    assert m[0, 0] == m[3, 2] == m[-1, 2] == m[-4, 2] == m[2, -3] == 3
    assert_raises(IndexError, m.__getitem__, (4, 0))
    assert_raises(IndexError, m.__getitem__, (2, -4))


def test_getitem_from_1D_constant_array():
    m = larray(3, shape=(43,))
    assert m[0] == m[42] == 3


def test_getitem__with_slice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:3, 0],
                       numpy.array([3, 3, 3]))


def test_getitem__with_thinslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_equal(m[2:3, 0:1], 3)


def test_getitem__with_mask_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[1, (0, 2)],
                       numpy.array([3, 3]))


def test_getitem_with_numpy_integers_from_2D_constant_array():
    if not hasattr(numpy, "int64"):
        raise SkipTest("test requires a 64-bit system")
    m = larray(3, shape=(4, 3))
    assert m[numpy.int64(0), numpy.int32(0)] == 3


def test_getslice_from_constant_array():
    m = larray(3, shape=(4, 3))
    assert_array_equal(m[:2],
                       numpy.array([[3, 3, 3],
                                    [3, 3, 3]]))


def test_getslice_past_bounds_from_constant_array():
    m = larray(3, shape=(5,))
    assert_array_equal(m[2:10],
                       numpy.array([3, 3, 3]))


def test_getitem_from_structured_array():
    m = larray(3 * numpy.ones((4, 3)), shape=(4, 3))
    assert m[0, 0] == m[3, 2] == m[-1, 2] == m[-4, 2] == m[2, -3] == 3
    assert_raises(IndexError, m.__getitem__, (4, 0))
    assert_raises(IndexError, m.__getitem__, (2, -4))


def test_getitem_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert_equal(m[5, 4], 14)


def test_getitem_from_1D_functional_array():
    m = larray(lambda i: i ** 3, shape=(6,))
    assert_equal(m[5], 125)


def test_getitem_from_3D_functional_array():
    m = larray(lambda i, j, k: i + j + k, shape=(2, 3, 4))
    assert_raises(NotImplementedError, m.__getitem__, (0, 1, 2))


def test_getitem_from_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7,))
    m3 = m[3]
    assert isinstance(m3, (int, numpy.integer))
    assert_equal(m3, 0)
    assert_equal(m[0], 1)


def test_getitem_with_slice_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert_array_equal(m[2:5, 3:],
                       numpy.array([[7, 8],
                                    [9, 10],
                                    [11, 12]]))


def test_getitem_with_slice_from_2D_functional_array_2():
    def test_function(i, j):
        return i * i + 2 * i * j + 3
    m = larray(test_function, shape=(3, 15))
    assert_array_equal(m[:, 3:14:3],
                       numpy.fromfunction(test_function, shape=(3, 15))[:, 3:14:3])


def test_getitem_with_mask_from_2D_functional_array():
    a = numpy.arange(30).reshape((6, 5))
    m = larray(lambda i, j: 5 * i + j, shape=(6, 5))
    assert_array_equal(a[[2, 3], [3, 4]],
                       numpy.array([13, 19]))
    assert_array_equal(m[[2, 3], [3, 4]],
                       numpy.array([13, 19]))


def test_getitem_with_mask_from_1D_functional_array():
    m = larray(lambda i: numpy.sqrt(i), shape=(10,))
    assert_array_equal(m[[0, 1, 4, 9]],
                       numpy.array([0, 1, 2, 3]))


def test_getitem_with_boolean_mask_from_1D_functional_array():
    m = larray(lambda i: numpy.sqrt(i), shape=(10,))
    assert_array_equal(m[numpy.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)],
                       numpy.array([0, 1, 2, 3]))


def test_getslice_from_2D_functional_array():
    m = larray(lambda i, j: 2 * i + j, shape=(6, 5))
    assert_array_equal(m[1:3],
                       numpy.array([[2, 3, 4, 5, 6],
                                    [4, 5, 6, 7, 8]]))


def test_getitem_from_iterator_array():
    m = larray(iter([1, 2, 3]), shape=(3,))
    assert_raises(NotImplementedError, m.__getitem__, 2)


def test_getitem_from_array_with_operations():
    a1 = numpy.array([[1, 3, 5], [7, 9, 11]])
    m1 = larray(a1)
    f = lambda i, j: numpy.sqrt(i * i + j * j)
    a2 = numpy.fromfunction(f, shape=(2, 3))
    m2 = larray(f, shape=(2, 3))
    a3 = 3 * a1 + a2
    m3 = 3 * m1 + m2
    assert_array_equal(a3[:, (0, 2)],
                       m3[:, (0, 2)])


def test_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    assert_raises(ValueError, m.evaluate)


def test_partially_evaluate_with_invalid_base_value():
    m = larray(range(5))
    m.base_value = "foo"
    assert_raises(ValueError, m._partially_evaluate, 3)


def test_check_bounds_with_invalid_address():
    m = larray([[1, 3, 5], [7, 9, 11]])
    assert_raises(TypeError, m.check_bounds, (object(), 1))


def test_check_bounds_with_invalid_address2():
    m = larray([[1, 3, 5], [7, 9, 11]])
    assert_raises(ValueError, m.check_bounds, ([], 1))


def test_partially_evaluate_constant_array_with_one_element():
    m = larray(3, shape=(1,))
    a = 3 * numpy.ones((1,))
    m1 = larray(3, shape=(1, 1))
    a1 = 3 * numpy.ones((1, 1))
    m2 = larray(3, shape=(1, 1, 1))
    a2 = 3 * numpy.ones((1, 1, 1))
    assert_equal(a[0], m[0])
    assert_equal(a.shape, m.shape)
    assert_equal(a[:].shape, m[:].shape)
    assert_equal(a, m.evaluate())
    assert_equal(a1.shape, m1.shape)
    assert_equal(a1[0,:].shape, m1[0,:].shape)
    assert_equal(a1[:].shape, m1[:].shape)
    assert_equal(a1, m1.evaluate())
    assert_equal(a2.shape, m2.shape)
    assert_equal(a2[:, 0,:].shape, m2[:, 0,:].shape)
    assert_equal(a2[:].shape, m2[:].shape)
    assert_equal(a2, m2.evaluate())


def test_partially_evaluate_constant_array_with_boolean_index():
    m = larray(3, shape=(4, 5))
    a = 3 * numpy.ones((4, 5))
    addr_bool = numpy.array([True, True, False, False, True])
    addr_int = numpy.array([0, 1, 4])
    assert_equal(a[::2, addr_bool].shape, a[::2, addr_int].shape)
    assert_equal(a[::2, addr_int].shape, m[::2, addr_int].shape)
    assert_equal(a[::2, addr_bool].shape, m[::2, addr_bool].shape)


def test_partially_evaluate_constant_array_with_all_boolean_indices_false():
    m = larray(3, shape=(3,))
    a = 3 * numpy.ones((3,))
    addr_bool = numpy.array([False, False, False])
    assert_equal(a[addr_bool].shape, m[addr_bool].shape)


def test_partially_evaluate_constant_array_with_only_one_boolean_indice_true():
    m = larray(3, shape=(3,))
    a = 3 * numpy.ones((3,))
    addr_bool = numpy.array([False, True, False])
    assert_equal(a[addr_bool].shape, m[addr_bool].shape)
    assert_equal(m[addr_bool][0], a[0])


def test_partially_evaluate_constant_array_with_boolean_indice_as_random_valid_ndarray():
    m = larray(3, shape=(3,))
    a = 3 * numpy.ones((3,))
    addr_bool = numpy.random.rand(3) > 0.5
    while not addr_bool.any():
        # random array, but not [False, False, False]
        addr_bool = numpy.random.rand(3) > 0.5
    assert_equal(a[addr_bool].shape, m[addr_bool].shape)
    assert_equal(m[addr_bool][0], a[addr_bool][0])


def test_partially_evaluate_constant_array_size_one_with_boolean_index_true():
    m = larray(3, shape=(1,))
    a = numpy.array([3])
    addr_bool = numpy.array([True])
    m1 = larray(3, shape=(1, 1))
    a1 = 3 * numpy.ones((1, 1))
    addr_bool1 = numpy.array([[True]], ndmin=2)
    assert_equal(m[addr_bool][0], a[0])
    assert_equal(m[addr_bool], a[addr_bool])
    assert_equal(m[addr_bool].shape, a[addr_bool].shape)
    assert_equal(m1[addr_bool1][0], a1[addr_bool1][0])
    assert_equal(m1[addr_bool1].shape, a1[addr_bool1].shape)


def test_partially_evaluate_constant_array_size_two_with_boolean_index_true():
    m2 = larray(3, shape=(1, 2))
    a2 = 3 * numpy.ones((1, 2))
    addr_bool2 = numpy.ones((1, 2), dtype=bool)
    assert_equal(m2[addr_bool2][0], a2[addr_bool2][0])
    assert_equal(m2[addr_bool2].shape, a2[addr_bool2].shape)


def test_partially_evaluate_constant_array_size_one_with_boolean_index_false():
    m = larray(3, shape=(1,))
    m1 = larray(3, shape=(1, 1))
    a = numpy.array([3])
    a1 = numpy.array([[3]], ndmin=2)
    addr_bool = numpy.array([False])
    addr_bool1 = numpy.array([[False]], ndmin=2)
    addr_bool2 = numpy.array([False])
    assert_equal(m[addr_bool].shape, a[addr_bool].shape)
    assert_equal(m1[addr_bool1].shape, a1[addr_bool1].shape)


def test_partially_evaluate_constant_array_size_with_empty_boolean_index():
    m = larray(3, shape=(1,))
    a = numpy.array([3])
    addr_bool = numpy.array([], dtype='bool')
    assert_equal(m[addr_bool].shape, a[addr_bool].shape)
    assert_equal(m[addr_bool].shape, (0,))


def test_partially_evaluate_functional_array_with_boolean_index():
    m = larray(lambda i, j: 5 * i + j, shape=(4, 5))
    a = numpy.arange(20.0).reshape((4, 5))
    addr_bool = numpy.array([True, True, False, False, True])
    addr_int = numpy.array([0, 1, 4])
    assert_equal(a[::2, addr_bool].shape, a[::2, addr_int].shape)
    assert_equal(a[::2, addr_int].shape, m[::2, addr_int].shape)
    assert_equal(a[::2, addr_bool].shape, m[::2, addr_bool].shape)


def test_getslice_with_vectorized_iterable():
    input = MockRNG(0, 1)
    m = larray(input, shape=(7, 3))
    assert_array_equal(m[::2, (0, 2)],
                       numpy.arange(8).reshape((4, 2)))


def test_equality():
    m1 = larray(42.0, shape=(4, 5)) / 23.0 + 2.0
    m2 = larray(42.0, shape=(4, 5)) / 23.0 + 2.0
    assert_equal(m1, m2)


def test_deepcopy():
    m1 = 3 * larray(lambda i, j: 5 * i + j, shape=(4, 5)) + 2
    m2 = deepcopy(m1)
    m1.shape = (3, 4)
    m3 = deepcopy(m1)
    assert_equal(m1.shape, m3.shape, (3, 4))
    assert_equal(m2.shape, (4, 5))
    assert_array_equal(m1.evaluate(), m3.evaluate())


def test_deepcopy_with_ufunc():
    m1 = sqrt(larray([x ** 2 for x in range(5)]))
    m2 = deepcopy(m1)
    m1.base_value[0] = 49
    assert_array_equal(m1.evaluate(), numpy.array([7, 1, 2, 3, 4]))
    assert_array_equal(m2.evaluate(), numpy.array([0, 1, 2, 3, 4]))


def test_set_shape():
    m = larray(42) + larray(lambda i: 3 * i)
    assert_equal(m.shape, None)
    m.shape = (5,)
    assert_array_equal(m.evaluate(), numpy.array([42, 45, 48, 51, 54]))


def test_call():
    A = larray(numpy.array([1, 2, 3]), shape=(3,)) - 1
    B = 0.5 * larray(lambda i: 2 * i, shape=(3,))
    C = B(A)
    assert_array_equal(C.evaluate(), numpy.array([0, 1, 2]))
    assert_array_equal(A.evaluate(), numpy.array([0, 1, 2]))  # A should be unchanged


def test_call2():
    positions = numpy.array(
        [[0.,  2.,  4.,  6.,  8.],
         [0.,  0.,  0.,  0.,  0.],
         [0.,  0.,  0.,  0.,  0.]])

    def position_generator(i):
        return positions.T[i]

    def distances(A, B):
        d = A - B
        d **= 2
        d = numpy.sum(d, axis=-1)
        numpy.sqrt(d, d)
        return d

    def distance_generator(f, g):
        def distance_map(i, j):
            return distances(f(i), g(j))
        return distance_map
    distance_map = larray(distance_generator(position_generator, position_generator),
                          shape=(4, 5))
    f_delay = 1000 * larray(lambda d: 0.1 * (1 + d), shape=(4, 5))
    assert_array_almost_equal(
        f_delay(distance_map).evaluate(),
        numpy.array([[100, 300, 500, 700, 900],
                     [300, 100, 300, 500, 700],
                     [500, 300, 100, 300, 500],
                     [700, 500, 300, 100, 300]], dtype=float),
        decimal=12)
    # repeat, should be idempotent
    assert_array_almost_equal(
        f_delay(distance_map).evaluate(),
        numpy.array([[100, 300, 500, 700, 900],
                     [300, 100, 300, 500, 700],
                     [500, 300, 100, 300, 500],
                     [700, 500, 300, 100, 300]], dtype=float),
        decimal=12)


def test__issue4():
    m = larray(numpy.arange(12).reshape((4, 3)))
   # mask1 = (slice(None), True)
    mask1 = (slice(None), numpy.array([True]))
    mask2 = (slice(None), numpy.array([True]))
    assert_equal(m[mask1].shape, partial_shape(mask1, m.shape), (4,))
    assert_equal(m[mask2].shape, partial_shape(mask2, m.shape), (4, 1))


def test__issue3():
    a = numpy.arange(12).reshape((4, 3))
    b = larray(a)
    c = larray(lambda i, j: 3*i + j, shape=(4, 3))
    assert_array_equal(a[(1, 3), :][:, (0, 2)], b[(1, 3), :][:, (0, 2)])
    assert_array_equal(b[(1, 3), :][:, (0, 2)], c[(1, 3), :][:, (0, 2)])
    assert_array_equal(a[(1, 3), (0, 2)], b[(1, 3), (0, 2)])
    assert_array_equal(b[(1, 3), (0, 2)], c[(1, 3), (0, 2)])


def test_partial_shape():
    a = numpy.arange(12).reshape((4, 3))
    test_cases = [
        (slice(None), (4, 3)),
        ((slice(None), slice(None)), (4, 3)),
        (slice(1, None, 2), (2, 3)),
        (1, (3,)),
        ((1, slice(None)), (3,)),
        ([0, 2, 3], (3, 3)),
        (numpy.array([0, 2, 3]), (3, 3)),
        ((numpy.array([0, 2, 3]), slice(None)), (3, 3)),
        (numpy.array([True, False, True, True]), (3, 3)),
        (numpy.array([True, False]), (1, 3)),
        (numpy.array([[True, False, False], [False, False, False], [True, True, False], [False, True, False]]), (4,)),
        (numpy.array([[True, False, False], [False, False, False], [True, True, False]]), (3,)),
        ((3, 1), tuple()),
        ((slice(None), 1), (4,)),
        ((slice(None), slice(1, None, 3)), (4, 1)),
        ((numpy.array([0, 3]), 2), (2,)),
        ((numpy.array([0, 3]), numpy.array([1, 2])), (2,)),
        ((slice(None), numpy.array([2])), (4, 1)),
        (((1, 3), (0, 2)), (2,)),
        (numpy.array([], bool), (0, 3)),
    ]
    for mask, expected_shape in test_cases:
        assert_equal(partial_shape(mask, a.shape), a[mask].shape)
        assert_equal(partial_shape(mask, a.shape), expected_shape)
    b = numpy.arange(5)
    test_cases = [
        (numpy.arange(5), (5,))
    ]
    for mask, expected_shape in test_cases:
        assert_equal(partial_shape(mask, b.shape), b[mask].shape)
        assert_equal(partial_shape(mask, b.shape), expected_shape)

def test_is_homogeneous():
    m0 = larray(10, shape=(5,))
    m1 = larray(numpy.arange(1, 6))
    m2 = m0 + m1
    m3 = 9 + m0 / m1
    assert m0.is_homogeneous
    assert not m1.is_homogeneous
    assert not m2.is_homogeneous
    assert not m3.is_homogeneous
