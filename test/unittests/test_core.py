from pyNN.core import is_listlike
import numpy


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
