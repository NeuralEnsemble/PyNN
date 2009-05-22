"""
Defines classes and functions for managing recordings (spikes, membrane
potential etc).
$Id$
"""

import tempfile
import logging
import os.path
import numpy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

DEFAULT_BUFFER_SIZE = 10000

if MPI:
    mpi_comm = MPI.COMM_WORLD
MPI_ROOT = 0

#class RecordingManager(object):
#    
#    def __init__(self, setup_function, get_function):
#        """
#        `setup_function` should take a variable, a source list, and an optional filename
#        and return an identifier.
#        `get_function` should take the identifier returned by `setup_function` and
#        return the recorded spikes, Vm trace, etc.
#        
#        Example:
#        rm = RecordingManager(_nest_record, _nest_get)
#        """
#        # create temporary directory
#        tempdir = tempfile.mkdtemp()
#        # initialise mapping from recording identifiers to temporary filenames
#        self.recorder_list = []
#        # for parallel simulations, determine if this is the master node
#        self._setup = setup_function
#        self._get = get_function
#    
#    def add_recording(self, sources, variable, filename=None):
#        recorder = self._setup(variable, sources, filename)
#        self.recorder_list.append(recorder)
#    
#    def get_recording(self, recording_id):
#        return self._get(recording_id)
#    
#    def write(self, recording_id, filename_or_obj, format="compatible", gather=True):
#        pass

def rename_existing(filename):
    if os.path.exists(filename):
        os.system('mv %s %s_old' % (filename, filename))
        logging.warning("File %s already exists. Renaming the original file to %s_old" % (filename, filename))


    
def write_compatible_output(sim_filename, user_filename, variable, input_format, population, dt, scale_factor=1):
    """
    Rewrite simulation data in a standard format:
        spiketime (in ms) cell_id-min(cell_id)
    """
    logging.info("Writing %s in compatible format (was %s)" % (user_filename, sim_filename))
                    
    # Writing spiketimes, cell_id-min(cell_id)                    
    # open file
    
    try: 
        N = os.path.getsize(sim_filename)
    except Exception:
        N = 0
    
    # First read the data (in case sim_filename and user_filename are identical)
    if N > 0:
        data = readArray(sim_filename, sepchar=None)
        
    # Write header info (e.g., dimensions of the population)
    result = open(user_filename,'w',DEFAULT_BUFFER_SIZE)
    if population is not None:
        result.write("# dimensions = %s\n" %list(population.dim))
        result.write("# first_id = %d\n" % 0)
        result.write("# last_id = %d\n" % (len(population)-1,))
        padding = population.first_id
    else:
        padding = 0
    result.write("# dt = %g\n" % dt)
        
    if N > 0:
        ##this is currently done in the individual modules# data[:,0] = data[:,0] - padding
        
        # sort
        ##indx = data.argsort(axis=0, kind='mergesort')[:,0] # will quicksort (not stable) work?
        ##data = data[indx]
        
        input_format = input_format.split()
        time_column = input_format.index('t')
        id_column = input_format.index('id')
        
        if data.shape[1] == 4: # conductance files
            ge_column = input_format.index('ge')
            gi_column = input_format.index('gi')
            result.write("# n = %d\n" % len(data))
            if scale_factor != 1:
                data[:,ge_column] *= scale_factor
                data[:,gi_column] *= scale_factor
            for idx in xrange(len(data)):
                result.write("%g\t%g\t%d\n" % (data[idx][ge_column], data[idx][gi_column], data[idx][id_column])) # ge gi id
        elif data.shape[1] == 3: # voltage files
            v_column = input_format.index('v')
            #result.write("# n = %d\n" % int(nest.GetStatus([0], "time")[0]/dt))
            result.write("# n = %d\n" % len(data))
            if scale_factor != 1:
                data[:,v_column] *= scale_factor
            for idx in xrange(len(data)):
                result.write("%g\t%g\n" % (data[idx][v_column], int(data[idx][id_column]))) # v id
        elif data.shape[1] == 2: # spike files
            for idx in xrange(len(data)):
                result.write("%g\t%g\n" % (data[idx][time_column], data[idx][id_column])) # time id
        else:
            raise Exception("Data file should have 2,3 or 4 columns, actually has %d" % data.shape[1])
    else:
        logging.info("%s is empty" % sim_filename)
        result.write("# n = 0\n")
    result.close()

def write_compatible_output1(data_source, user_filename, variable, input_format, population, dt):
    """
    Rewrite simulation data in a standard format:
        spiketime (in ms) cell_id-min(cell_id)
    """
    if isinstance(data_source, numpy.ndarray):
        logging.info("Writing %s in compatible format (from memory)" % user_filename)
        N = len(data_source)
    else: # assume data is a filename or open file object
        logging.info("Writing %s in compatible format (was %s)" % (user_filename, data_source))
        try: 
            N = os.path.getsize(data_source)
        except Exception:
            N = 0
    
    if N > 0:
        user_file = StandardTextFile(user_filename)    
        # Write header info (e.g., dimensions of the population)
        metadata = {}
        if population is not None:
            metadata.update({
                'dimensions': str(list(population.dim)),
                'first_id': 0,
                'last_id': len(population)-1
            })
            id_offset = population.first_id
        else:
            id_offset = 0
        metadata['dt'] = dt
        
        input_format = input_format.split()
        time_column = input_format.index('t')
        id_column = input_format.index('id')
        
        if variable == 'conductance':
            ge_column = input_format.index('ge')
            gi_column = input_format.index('gi')
            column_map = [ge_column, gi_column, id_column]
        elif variable == 'v': # voltage files
            v_column = input_format.index('v')
            column_map = [v_column, id_column]
        elif variable == 'spikes': # spike files
            column_map = [time_column, id_column]
        else:
            raise Exception("Invalid variable")
        
        # Read/select data
        if isinstance(data_source, numpy.ndarray):
            data_array = data_source[:, column_map]
        else: # assume data is a filename or open file object
            #data_array = numpy.loadtxt(data_source, usecols=column_map)
            data_array = readArray(data_source, sepchar=None)
        data_array[:,-1] -= id_offset # replies on fact that id is always last column
        metadata['n'] = data_array.shape[0]
        
        user_file.write(data_array, metadata)
    else:
        logging.warning("%s is empty or does not exist" % data_source)

def readArray(filename, sepchar=None, skipchar='#'):
    """
    (Pylab has a great load() function, but it is not necessary to import
    it into pyNN. The fromfile() function of numpy has trouble on several
    machines with Python 2.5, so that's why a dedicated readArray function
    has been created to load from file the spike raster or the membrane potentials
    saved by the simulators).
    """
    myfile = open(filename, "r", DEFAULT_BUFFER_SIZE)
    contents = myfile.readlines()
    myfile.close()
    data = []
    for line in contents:
        stripped_line = line.lstrip()
        if (len(stripped_line) != 0):
            if (stripped_line[0] != skipchar):
                items = stripped_line.split(sepchar)
                data.append(map(float, items))
    a = numpy.array(data)
    if a.size > 0:
        (Nrow, Ncol) = a.shape
        #logging.debug(str(a.shape))
        #if ((Nrow == 1) or (Ncol == 1)): a = numpy.ravel(a)
    return a

def gather(data):
    # gather 1D or 2D numpy arrays
    assert isinstance(data, numpy.ndarray)
    assert len(data.shape) < 3
    # first we pass the data size
    size = data.size
    sizes = mpi_comm.gather(size, root=MPI_ROOT) or []
    # now we pass the data
    displacements = [sum(sizes[:i]) for i in range(len(sizes))]
    #print mpi_comm.rank, sizes, displacements, data
    gdata = numpy.empty(sum(sizes))
    mpi_comm.Gatherv([data.flatten(), size, MPI.DOUBLE], [gdata, (sizes,displacements), MPI.DOUBLE], root=MPI_ROOT)
    if len(data.shape) == 1:
        return gdata
    else:
        num_columns = data.shape[1]
        return gdata.reshape((gdata.size/num_columns, num_columns))
  
def gather_dict(D):
    # Note that if the same key exists on multiple nodes, the value from the
    # node with the highest rank will appear in the final dict.
    Ds = mpi_comm.gather(D, root=MPI_ROOT)
    if Ds:
        for otherD in Ds:
            D.update(otherD)
    return D

def mpi_sum(x):
    return mpi_comm.allreduce(x, op=MPI.SUM)


class StandardTextFile(object):
    
    
    def __init__(self, filename, gather=False):
        self.name = filename
        self.gather = gather
        self.fileobj = open(self.name, 'w', DEFAULT_BUFFER_SIZE)
        
    def write(self, data, metadata):
        # can we write to the file more than once? In this case, should use seek,tell
        # to always put the header information at the top?
        # write header
        header_lines = ["# %s = %s" % item for item in metadata.items()]
        self.fileobj.write("\n".join(header_lines) + '\n')
        # write data
        numpy.savetxt(self.fileobj, data, fmt = '%s', delimiter='\t')
        self.fileobj.close()
