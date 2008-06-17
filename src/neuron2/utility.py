from pyNN import __path__ as pyNN_path
import platform
import logging
import numpy
import os.path
import neuron
h = neuron.h

# Global variables
nrn_dll_loaded = []

def load_mechanisms(path=pyNN_path[0]):
    global nrn_dll_loaded
    if path not in nrn_dll_loaded:
        arch_list = [platform.machine(), 'i686', 'x86_64', 'powerpc']
        # in case NEURON is assuming a different architecture to Python, we try multiple possibilities
        for arch in arch_list:
            lib_path = os.path.join(path, 'hoc', arch, '.libs', 'libnrnmech.so')
            if os.path.exists(lib_path):
                h.nrn_load_dll(lib_path)
                nrn_dll_loaded.append(path)
                return
        raise Exception("NEURON mechanisms not found in %s." % os.path.join(path, 'hoc'))

class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    formats = {'spikes': "%g\t%d",
               'v': "%g\t%g\t%d"}
    
    def __init__(self, variable, population=None, file=None):
        """
        `file` should be one of:
            a file-name,
            `None` (write to a temporary file)
            `False` (write to memory).
        """
        self.variable = variable
        self.filename = file or None
        self.population = population # needed for writing header information
        self.recorded = set([])        

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        logging.debug('Recorder.record(%s)', str(ids))
        if self.population:
            ids = set([id for id in ids if id in self.population._local_ids])
        else:
            ids = set(ids) # how to decide if the cell is local?
        new_ids = list( ids.difference(self.recorded) )
        
        self.recorded = self.recorded.union(ids)
        logging.debug('Recorder.recorded = %s' % self.recorded)
        if self.variable == 'spikes':
            for id in new_ids:
                id._cell.record(1)
        elif self.variable == 'v':
            for id in new_ids:
                id._cell.record_v(1)
        
    def get(self, gather=False):
        """Returns the recorded data."""
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.recorded:
                spikes = id._cell.spiketimes.toarray()
                if len(spikes) > 0:
                    new_data = numpy.array([spikes, numpy.ones(spikes.shape)*id]).T
                    data = numpy.concatenate((data, new_data))
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.recorded:
                v = id._cell.vtrace.toarray()
                t = id._cell.record_times.toarray()
                new_data = numpy.array([t, v, numpy.ones(v.shape)*id]).T
                data = numpy.concatenate((data, new_data))
        return data
    
    def write(self, file=None, gather=False, compatible_output=True):
        data = self.get(gather)
        numpy.savetxt(file or self.filename, data, Recorder.formats[self.variable])
        
class Initializer(object):
    
    def __init__(self):
        self.cell_list = []
        self.population_list = []
        neuron.h('objref initializer')
        neuron.h('initializer = PythonObject(self)')
    
    def initialize(self):
        for cell in self.cell_list:
            cell.memb_init()


load_mechanisms()