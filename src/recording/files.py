"""
Provides standard interfaces to various text and binary file formats for saving
spikes, membrane potential and synaptic conductances.

To record data in a given format, pass an instance of any of the File classes to
the Population.printSpikes(), print_v() or print_gsyn() methods.

Classes:
    StandardTextFile
    PickleFile
    NumpyBinaryFile
    HDF5ArrayFile - requires PyTables

$Id$
"""


import numpy
import cPickle as pickle
try:
    import tables
    have_hdf5 = True
except ImportError:
    have_hdf5 = False
    

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


class BaseFile(object):
    """
    Base class for PyNN File classes.
    """
    
    def __init__(self, filename, mode='r'):
        """
        Open a file with the given filename and mode.
        """
        self.name = filename
        self.mode = mode
        self.fileobj = open(self.name, mode, DEFAULT_BUFFER_SIZE)

    def __del__(self):
        self.close()

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
        # can we write to the file more than once? In this case, should use seek,tell
        # to always put the header information at the top?
        # write header
        header_lines = ["# %s = %s" % item for item in metadata.items()]
        self.fileobj.write("\n".join(header_lines) + '\n')
        # write data
        savetxt = getattr(numpy, 'savetxt', _savetxt)
        savetxt(self.fileobj, data, fmt='%s', delimiter='\t')
        self.fileobj.close()


class PickleFile(BaseFile):
    """
    Data and metadata are pickled and saved to file.
    """
    
    def write(self, data, metadata):
        __doc__ = BaseFile.write.__doc__
        pickle.dump((data, metadata), self.fileobj)
        
    def read(self):
        __doc__ = BaseFile.read.__doc__
        data = pickle.load(self.fileobj)[0]
        self.fileobj.seek(0)
        return data
    
    def get_metadata(self):
        __doc__ = BaseFile.get_metadata.__doc__
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
        metadata_array = numpy.array(metadata.items())
        numpy.savez(self.fileobj, data=data, metadata=metadata_array)
        
    def read(self):
        __doc__ = BaseFile.read.__doc__
        data = numpy.load(self.fileobj)['data']
        self.fileobj.seek(0)
        return data
    
    def get_metadata(self):
        __doc__ = BaseFile.get_metadata.__doc__
        D = {}
        for name,value in numpy.load(self.fileobj)['metadata']:
            D[name] = eval(value)
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
                except tables.HDF5ExtError, e:
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
    