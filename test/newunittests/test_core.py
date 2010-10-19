from pyNN.core import is_listlike, LazyArray
import numpy
from nose.tools import assert_raises
from tools import assert_arrays_equal


def test_is_list_like_with_tuple():
    assert is_listlike((1,2,3))    
    
def test_is_list_like_with_list():
    assert is_listlike([1,2,3]) 

def test_is_list_like_with_iterator():
    assert not is_listlike(iter((1,2,3))) 

def test_is_list_like_with_set():
    assert is_listlike(set((1,2,3))) 

def test_is_list_like_with_numpy_array():
    assert is_listlike(numpy.arange(10))

def test_is_list_like_with_string():
    assert not is_listlike("abcdefg")

#def test_is_list_like_with_file():
#    f = file()
#    assert not is_listlike(f)

# test LazyArray
def test_create_with_int():
    A = LazyArray(5, 3)
    assert A.size == 5
    assert A.value == 3

def test_create_with_float():
    A = LazyArray(5, 3.0)
    assert A.size == 5
    assert A.value == 3.0

def test_create_with_list():
    A = LazyArray(3, [1,2,3])
    assert A.size == 3
    assert A.value == [1,2,3]

def test_create_with_array():
    A = LazyArray(3, numpy.array([1,2,3]))
    assert A.size == 3
    assert_arrays_equal(A.value, numpy.array([1,2,3]))

def test_create_inconsistent():
    assert_raises(AssertionError, LazyArray, 4, [1,2,3])

def test_create_with_string():
    assert_raises(AssertionError, LazyArray, 3, "123")

def test_getitem_nonexpanded():
    A = LazyArray(5, 3)
    assert A[0] == A[4] == A[-1] == A[-5] == 3
    assert_raises(IndexError, A.__getitem__, 5)
    assert_raises(IndexError, A.__getitem__, -6)

def test_getitem_expanded():
    A = LazyArray(5, numpy.array((3,3,3,3,3)))
    assert A[0] == A[4] == A[-1] == A[-5] == 3
    assert_raises(IndexError, A.__getitem__, 5)
    assert_raises(IndexError, A.__getitem__, -6)
    
def test_setitem_nonexpanded_same_value():
    A = LazyArray(5, 3)
    assert A.value == 3
    A[0] = 3
    assert A.value == 3

def test_setitem_invalid_value():
    A = LazyArray(5, 3)
    assert_raises(TypeError, A, "abc")

def test_setitem_nonexpanded_different_value():
    A = LazyArray(5, 3)
    assert A.value == 3
    A[0] = 4; A[4] = 5
    assert_arrays_equal(A.value, numpy.array([4, 3, 3, 3, 5]))
