import os
from unittest.mock import Mock
from textwrap import dedent

import numpy as np
from numpy.testing import assert_array_equal

from pyNN.recording import files


builtin_open = open


def test__savetxt():
    mock_file = Mock()
    files.open = Mock(return_value=mock_file)
    files._savetxt(filename="dummy_file",
                   data=[(0, 2.3), (1, 3.4), (2, 4.3)],
                   format="%f",
                   delimiter=" ")
    target = [(('0.000000 2.300000\n',), {}),
              (('1.000000 3.400000\n',), {}),
              (('2.000000 4.300000\n',), {})]
    assert mock_file.write.call_args_list == target
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

# def test_StandardTextFile_write():
#    files.open = Mock()
#    stf = files.StandardTextFile("filename", "w")
#    data=[(0, 2.3),(1, 3.4),(2, 4.3)]
#    metadata = {'a': 1, 'b': 9.99}
#    target = [(('# a = 1\n# b = 9.99\n',), {}),
#              (('0.0\t2.3\n',), {}),
#              (('1.0\t3.4\n',), {}),
#              (('2.0\t4.3\n',), {})]
#    stf.write(data, metadata)
#    assert stf.fileobj.write.call_args_list == target
#    files.open = builtin_open


def test_StandardTextFile_read():
    files.open = Mock()
    stf = files.StandardTextFile("filename", "w")
    orig_loadtxt = np.loadtxt
    np.loadtxt = Mock()
    stf.read()
    np.loadtxt.assert_called_with(stf.fileobj)
    np.loadtxt = orig_loadtxt
    files.open = builtin_open


def test_PickleFile():
    pf = files.PickleFile("tmp.pickle", "wb")
    data = [(0, 2.3), (1, 3.4), (2, 4.3)]
    metadata = {'a': 1, 'b': 9.99}
    pf.write(data, metadata)
    pf.close()

    pf = files.PickleFile("tmp.pickle", "rb")
    assert pf.get_metadata() == metadata
    assert pf.read() == data
    pf.close()

    os.remove("tmp.pickle")

# def test_NumpyBinaryFile():
#    nbf = files.NumpyBinaryFile("tmp.npz", "w")
#    data=[(0, 2.3), (1, 3.4), (2, 4.3)]
#    metadata = {'a': 1, 'b': 9.99}
#    nbf.write(data, metadata)
#    nbf.close()
#
#    nbf = files.NumpyBinaryFile("tmp.npz", "r")
#    assert nbf.get_metadata() == metadata
#    assert_array_equal(nbf.read().flatten(), np.array(data).flatten())
#    nbf.close()
#
#    os.remove("tmp.npz")


def test_HDF5ArrayFile():
    if files.have_hdf5:
        h5f = files.HDF5ArrayFile("tmp.h5", "w")
        data = [(0, 2.3), (1, 3.4), (2, 4.3)]
        metadata = {'a': 1, 'b': 9.99}
        h5f.write(data, metadata)
        h5f.close()

        h5f = files.HDF5ArrayFile("tmp.h5", "r")
        assert h5f.get_metadata() == metadata
        assert_array_equal(np.array(h5f.read()).flatten(),
                            np.array(data).flatten())
        h5f.close()

        os.remove("tmp.h5")
