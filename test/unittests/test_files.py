from pyNN.recording import files
from textwrap import dedent
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
from nose.tools import assert_equal
import numpy
import os
from pyNN.utility import assert_arrays_equal

builtin_open = open

def test__savetxt():
    mock_file = Mock()
    files.open = Mock(return_value=mock_file)
    files._savetxt(filename="dummy_file",
                   data=[(0, 2.3),(1, 3.4),(2, 4.3)],
                   format="%f", 
                   delimiter=" ")
    target = [(('0.000000 2.300000\n',), {}),
              (('1.000000 3.400000\n',), {}),
              (('2.000000 4.300000\n',), {})]
    assert_equal(mock_file.write.call_args_list, target)
    files.open = builtin_open  
    
def test_create_BaseFile():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    files.open.assert_called_with("filename", "r", files.DEFAULT_BUFFER_SIZE)
    files.open = builtin_open    

def test_del():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    close_mock = Mock()
    bf.close = close_mock
    del bf
    close_mock.assert_called_with()
    files.open = builtin_open
    
def test_close():
    files.open = Mock()
    bf = files.BaseFile("filename", 'r')
    bf.close()
    bf.fileobj.close.assert_called_with()
    files.open = builtin_open
    
#def test_StandardTextFile_write():
#    files.open = Mock()
#    stf = files.StandardTextFile("filename", "w")
#    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
#    metadata = {'a': 1, 'b': 9.99}
#    target = [(('# a = 1\n# b = 9.99\n',), {}),
#              (('0.0\t2.3\n',), {}),
#              (('1.0\t3.4\n',), {}),
#              (('2.0\t4.3\n',), {})]
#    stf.write(data, metadata)
#    assert_equal(stf.fileobj.write.call_args_list,
#                 target)
#    files.open = builtin_open
    
def test_StandardTextFile_read():
    files.open = Mock()
    stf = files.StandardTextFile("filename", "w")
    orig_loadtxt = numpy.loadtxt
    numpy.loadtxt = Mock()
    stf.read()
    numpy.loadtxt.assert_called_with(stf.fileobj)
    numpy.loadtxt = orig_loadtxt
    files.open = builtin_open
    
def test_PickleFile():
    pf = files.PickleFile("tmp.pickle", "wb")
    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    pf.write(data, metadata)
    pf.close()
    
    pf = files.PickleFile("tmp.pickle", "rb")
    assert_equal(pf.get_metadata(), metadata)
    assert_equal(pf.read(), data)
    pf.close()
    
    os.remove("tmp.pickle")
    
#def test_NumpyBinaryFile():
#    nbf = files.NumpyBinaryFile("tmp.npz", "w")
#    data=[(0, 2.3), (1, 3.4), (2, 4.3)]
#    metadata = {'a': 1, 'b': 9.99}
#    nbf.write(data, metadata)
#    nbf.close()
#    
#    nbf = files.NumpyBinaryFile("tmp.npz", "r")
#    assert_equal(nbf.get_metadata(), metadata)
#    assert_arrays_equal(nbf.read().flatten(), numpy.array(data).flatten())
#    nbf.close()
#
#    os.remove("tmp.npz")
    
def test_HDF5ArrayFile():
    if files.have_hdf5:
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
