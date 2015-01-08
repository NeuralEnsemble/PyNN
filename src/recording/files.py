"""
Provides standard interfaces to various text and binary file formats for saving
position and connectivity data.  Note that saving spikes, membrane potential
and synaptic conductances is now done via Neo.

Classes:
    StandardTextFile
    PickleFile
    NumpyBinaryFile
    HDF5ArrayFile - requires PyTables

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""


import numpy, os, shutil
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import tables
    have_hdf5 = True
except ImportError:
    have_hdf5 = False
from pyNN.core import iteritems


DEFAULT_BUFFER_SIZE = 10000

def _savetxt(filename, data, format, delimiter):
    """
    Due to the lack of savetxt in older versions of numpy
    we provide a cut-down version of that function.
    """
    f = open(filename, 'w')
    for row in data:
        f.write(delimiter.join([format%val for val in row]) + '\n')
    f.close()


def savez(file, *args, **kwds):

    __doc__ = numpy.savez.__doc__
    import zipfile
    from numpy.lib import format

    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            raise ValueError("Cannot use un-named variables and keyword %s" % key)
        namedict[key] = val

    zip = zipfile.ZipFile(file, mode="wb")

    # Place to write temporary .npy files
    #  before storing them in the zip. We need to path this to have a working
    # function in parallel !
    import tempfile
    direc = tempfile.mkdtemp()
    for key, val in iteritems(namedict):
        fname = key + '.npy'
        filename = os.path.join(direc, fname)
        fid = open(filename, 'wb')
        format.write_array(fid, numpy.asanyarray(val))
        fid.close()
        zip.write(filename, arcname=fname)
    zip.close()
    shutil.rmtree(direc)


class BaseFile(object):
    """
    Base class for PyNN File classes.
    """

    def __init__(self, filename, mode='rb'):
        """
        Open a file with the given filename and mode.
        """
        self.name = filename
        self.mode = mode
        dir = os.path.dirname(filename)
        if dir and not os.path.exists(dir):
            try:  # wrapping in try...except block for MPI
                os.makedirs(dir)
            except IOError:
                pass
        try: ## Need this because in parallel, file names are changed
            self.fileobj = open(self.name, mode, DEFAULT_BUFFER_SIZE)
        except Exception as err:
            self.open_error = err

    def __del__(self):
        self.close()

    def _check_open(self):
        if not hasattr(self, 'fileobj'):
            raise self.open_error

    def rename(self, filename):
        self.close()
        try: ## Need this because in parallel, only one node will delete the file with NFS
            os.remove(self.name)
        except Exception:
            pass
        self.name = filename
        self.fileobj = open(self.name, self.mode, DEFAULT_BUFFER_SIZE)

    def write(self, data, metadata):
        """
        Write data and metadata to file. `data` should be a NumPy array,
        `metadata` should be a dictionary.
        """
        raise NotImplementedError

    def read(self):
        """
        Read data from the file and return a NumPy array.
        """
        raise NotImplementedError

    def get_metadata(self):
        """
        Read metadata from the file and return a dict.
        """
        raise NotImplementedError

    def close(self):
        """Close the file."""
        if hasattr(self, 'fileobj'):
            self.fileobj.close()


class StandardTextFile(BaseFile):
    """
    Data and metadata is written as text. Metadata is written at the top of the
    file, with each line preceded by "#". Data is written with one data point per line.
    """

    def write(self, data, metadata):
        __doc__ = BaseFile.write.__doc__
        self._check_open()
        # can we write to the file more than once? In this case, should use seek,tell
        # to always put the header information at the top?
        # write header
        header_lines = ["# %s = %s" % item for item in metadata.items()]
        header = "\n".join(header_lines) + '\n'
        self.fileobj.write(header.encode('utf-8'))
        # write data
        savetxt = getattr(numpy, 'savetxt', _savetxt)
        savetxt(self.fileobj, data, fmt='%r', delimiter='\t')
        self.fileobj.close()

    def read(self):
        self._check_open()
        return numpy.loadtxt(self.fileobj)

    def get_metadata(self):
        self._check_open()
        D = {}
        for line in self.fileobj:
            if line:
                if line[0] != "#":
                    break
                name, value = line[1:].split("=")
                name = name.strip()
                try:
                    D[name] = eval(value)
                except Exception:
                    D[name] = value.strip()
            else:
                break
        self.fileobj.seek(0)
        return D


class PickleFile(BaseFile):
    """
    Data and metadata are pickled and saved to file.
    """

    def write(self, data, metadata):
        __doc__ = BaseFile.write.__doc__
        self._check_open()
        pickle.dump((data, metadata), self.fileobj)

    def read(self):
        __doc__ = BaseFile.read.__doc__
        self._check_open()
        data = pickle.load(self.fileobj)[0]
        self.fileobj.seek(0)
        return data

    def get_metadata(self):
        __doc__ = BaseFile.get_metadata.__doc__
        self._check_open()
        metadata = pickle.load(self.fileobj)[1]
        self.fileobj.seek(0)
        return metadata


class NumpyBinaryFile(BaseFile):
    """
    Data and metadata are saved in .npz format, which is a zipped archive of
    arrays.
    """

    def write(self, data, metadata):
        __doc__ = BaseFile.write.__doc__
        self._check_open()
        metadata_array = numpy.array(metadata.items(), dtype=(str, float))
        savez(self.fileobj, data=data, metadata=metadata_array)

    def read(self):
        __doc__ = BaseFile.read.__doc__
        self._check_open()
        data = numpy.load(self.fileobj)['data']
        self.fileobj.seek(0)
        return data

    def get_metadata(self):
        __doc__ = BaseFile.get_metadata.__doc__
        self._check_open()
        D = {}
        for name,value in numpy.load(self.fileobj)['metadata']:
            try:
                D[name] = eval(value)
            except Exception:
                D[name] = value
        self.fileobj.seek(0)
        return D


if have_hdf5:
    class HDF5ArrayFile(BaseFile):
        """
        Data are saved as an array within a node named "data". Metadata are
        saved as attributes of this node.
        """

        def __init__(self, filename, mode='r', title="PyNN data file"):
            """
            Open an HDF5 file with the given filename, mode and title.
            """
            self.name = filename
            self.mode = mode
            self.fileobj = tables.openFile(filename, mode=mode, title=title)

        # may not work with old versions of PyTables < 1.3, since they only support numarray, not numpy
        def write(self, data, metadata):
            __doc__ = BaseFile.write.__doc__
            if len(data) > 0:
                try:
                    node = self.fileobj.createArray(self.fileobj.root, "data", data)
                except tables.HDF5ExtError as e:
                    raise tables.HDF5ExtError("%s. data.shape=%s, metadata=%s" % (e, data.shape, metadata))
                for name, value in metadata.items():
                    setattr(node.attrs, name, value)
                self.fileobj.close()

        def read(self):
            __doc__ = BaseFile.read.__doc__
            return self.fileobj.root.data.read()

        def get_metadata(self):
            __doc__ = BaseFile.get_metadata.__doc__
            D = {}
            node = self.fileobj.root.data
            for name in node._v_attrs._f_list():
                D[name] = node.attrs.__getattr__(name)
            return D
