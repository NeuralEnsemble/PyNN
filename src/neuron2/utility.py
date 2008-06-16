from pyNN import __path__ as pyNN_path
import platform
import logging
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
        ids = set([id for id in ids if id in self.population._local_ids])
        new_ids = list( ids.difference(self.recorded) )
        logging.info("%s.record('%s', %s)", self.population.label, self.variable, new_ids[:5])
        self.recorded = self.recorded.union(ids)
        if self.variable == 'spikes':
            for cell in new_ids:
                cell.record(1)
        elif self.variable == 'v':
            for cell in new_ids:
                cell.record_v(1)
        
    def get(self, gather=False):
        """Returns the recorded data."""
        pass
    
    def write(self, file=None, gather=False, compatible_output=True):
        pass

load_mechanisms()