from pyNN.recording import files
from textwrap import dedent
from mock import Mock
from nose.tools import assert_equal
import numpy
import os

builtin_open = open

def assert_arrays_equal(a, b):
    assert isinstance(a, numpy.ndarray), "a is a %s" % type(a)
    assert isinstance(b, numpy.ndarray), "b is a %s" % type(b)
    assert all(a==b), "%s != %s" % (a,b)

def test__savetxt():
    files.open = Mock()
    files._savetxt(filename="dummy_file",
                   data=[(0, 2.3),(1, 3.4),(2, 4.3)],
                   format="%f", 
                   delimiter=" ")
    target_output = dedent("""
                        0 2.3
                        1 3.4
                        2 4.3
                    """)
    files.open.write.assert_called_with_args(target_output)
    
def test_create_BaseFile():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    files.open.assert_called_with_args("filename", "r", files.DEFAULT_BUFFER_SIZE)
    files.open = builtin_open    

def test_del():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    close_mock = Mock()
    bf.close = close_mock
    del bf
    close_mock.assert_called_with_args()
    files.open = builtin_open
    
def test_close():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    bf.close()
    bf.fileobj.close.assert_called_with_args()
    files.open = builtin_open
    
def test_StandardTextFile_write():
    files.open = Mock()
    stf = files.StandardTextFile("filename", "w")
    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    target = [(('# a = 1\n# b = 9.99\n',), {}),
              (('0.0\t2.3\n',), {}),
              (('1.0\t3.4\n',), {}),
              (('2.0\t4.3\n',), {})]
    stf.write(data, metadata)
    assert_equal(stf.fileobj.write.call_args_list,
                 target)
    files.open = builtin_open
    
def test_StandardTextFile_read():
    files.open = Mock()
    stf = files.StandardTextFile("filename", "w")
    orig_loadtxt = numpy.loadtxt
    numpy.loadtxt = Mock()
    stf.read()
    numpy.loadtxt.assert_called_with_args(stf.fileobj)
    numpy.loadtxt = orig_loadtxt
    files.open = builtin_open
    
def test_PickleFile():
    pf = files.PickleFile("tmp.pickle", "w")
    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    pf.write(data, metadata)
    pf.close()
    
    pf = files.PickleFile("tmp.pickle", "r")
    assert_equal(pf.get_metadata(), metadata)
    assert_equal(pf.read(), data)
    pf.close()
    
    os.remove("tmp.pickle")
    
def test_NumpyBinaryFile():
    nbf = files.NumpyBinaryFile("tmp.npz", "w")
    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    nbf.write(data, metadata)
    nbf.close()
    
    nbf = files.NumpyBinaryFile("tmp.npz", "r")
    assert_equal(nbf.get_metadata(), metadata)
    assert_arrays_equal(nbf.read().flatten(), numpy.array(data).flatten())
    nbf.close()

    os.remove("tmp.npz")
    
def test_HDF5ArrayFile():
    h5f = files.HDF5ArrayFile("tmp.h5", "w")
    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    h5f.write(data, metadata)
    h5f.close()
    
    h5f = files.HDF5ArrayFile("tmp.h5", "r")
    assert_equal(h5f.get_metadata(), metadata)
    assert_arrays_equal(numpy.array(h5f.read()).flatten(),
                        numpy.array(data).flatten())
    h5f.close()

    os.remove("tmp.h5")
