
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
    
    def __init__(self, filename, mode='r'):
        self.name = filename
        self.mode = mode
        self.fileobj = open(self.name, mode, DEFAULT_BUFFER_SIZE)

    def __del__(self):
        self.close()

    def close(self):
        self.fileobj.close()


class StandardTextFile(BaseFile):
    
    def write(self, data, metadata):
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
    
    def write(self, data, metadata):
        pickle.dump((data, metadata), self.fileobj)
        
    def read(self):
        data = pickle.load(self.fileobj)[0]
        self.fileobj.seek(0)
        return data
    
    def get_metadata(self):
        metadata = pickle.load(self.fileobj)[1]
        self.fileobj.seek(0)
        return metadata
        
        
class NumpyBinaryFile(BaseFile):
    
    def write(self, data, metadata):
        metadata_array = numpy.array(metadata.items())
        numpy.savez(self.fileobj, data=data, metadata=metadata_array)
        
    def read(self):
        data = numpy.load(self.fileobj)['data']
        self.fileobj.seek(0)
        return data
    
    def get_metadata(self):
        D = {}
        for name,value in numpy.load(self.fileobj)['metadata']:
            D[name] = eval(value)
        self.fileobj.seek(0)
        return D
    
    
if have_hdf5:    
    class HDF5ArrayFile(BaseFile):
        
        def __init__(self, filename, mode='r'):
            self.name = filename
            self.mode = mode
            self.fileobj = tables.openFile(filename, mode=mode, title="PyNN data file")
            
        # may not work with old versions of PyTables < 1.3, since they only support numarray, not numpy
        def write(self, data, metadata):
            if len(data) > 0:
                try:
                    node = self.fileobj.createArray(self.fileobj.root, "data", data)
                except tables.HDF5ExtError, e:
                    raise tables.HDF5ExtError("%s. data.shape=%s, metadata=%s" % (e, data.shape, metadata))
                for name, value in metadata.items():
                    setattr(node.attrs, name, value)
                self.fileobj.close()
    
        def read(self):
            return self.fileobj.root.data.read()
    
        def get_metadata(self):
            D = {}
            node = self.fileobj.root.data
            for name in node._v_attrs._f_list():
                D[name] = node.attrs.__getattr__(name)
            return D
    