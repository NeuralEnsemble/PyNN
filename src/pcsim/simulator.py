
import logging
import pypcsim
import types
import numpy
from pyNN import recording

recorder_list = []

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.t = 0.0
        self.dt = None
        self.min_delay = None
        self.max_delay = None
        self.constructRNGSeed = None
        self.simulationRNGSeed = None
    
    @property
    def num_processes(self):
        return net.mpi_size()
    
    @property
    def mpi_rank(self):
        return net.mpi_rank()


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    fieldnames = {'v': 'Vm',
                  'gsyn': None}
    numpy1_0_formats = {'spikes': "%g", # only later versions of numpy support different
                        'v': "%g",      # formats for different columns
                        'gsyn': "%g"}
    formats = {'spikes': 'id t',
               'v': 'id t v',
               'gsyn': 'id t ge gi'}
    
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
        self.recorders = {}
    
    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        logging.debug('Recorder.record(%s)', str(ids))
        if self.population:
            ids = set([id for id in ids if id in self.population.local_cells])
        else:
            ids = set(ids) # how to decide if the cell is local?
        new_ids = list( ids.difference(self.recorded) )
        
        self.recorded = self.recorded.union(ids)
        logging.debug('Recorder.recorded = %s' % self.recorded)
        if self.variable == 'spikes':        
            for id in new_ids:
                if self.population:
                    pcsim_id = self.population.pcsim_population[int(id)]
                else:
                    pcsim_id = int(id)
                src_id = pypcsim.SimObject.ID(pcsim_id)    
                rec = net.create(pypcsim.SpikeTimeRecorder(),
                                 pypcsim.SimEngine.ID(src_id.node, src_id.eng))            
                net.connect(pcsim_id, rec, pypcsim.Time.sec(0))
                self.recorders[id] = rec
        elif self.variable in ('v', 'gsyn'):
            for id in new_ids:
                if self.population:
                    pcsim_id = self.population.pcsim_population[int(id)]
                else:
                    pcsim_id = int(id)
                src_id = pypcsim.SimObject.ID(pcsim_id)
                rec = net.create(pypcsim.AnalogRecorder(),
                                 pypcsim.SimEngine.ID(src_id.node, src_id.eng))
                net.connect(pcsim_id, Recorder.fieldnames[self.variable], rec, 0, pypcsim.Time.sec(0))
                self.recorders[id] = rec
        else:
            raise Exception("Recording of %s not implemented." % self.variable)

    def get(self, gather=False, compatible_output=True, offset=None):
        """Returns the recorded data."""
        # compatible_output is not used, but is needed for compatibility with the nest2 module.
        # Does nest2 really need it?
        if offset is None:
            if self.population:
                offset = 0 #self.population.first_id
            else:
                offset = 0
                
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.recorded:
                rec = self.recorders[id]
                spikes = 1000.0*numpy.array(net.object(rec).getSpikeTimes())
                spikes = spikes.flatten()
                spikes = spikes[spikes<=state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape)*(id-offset), spikes]).T
                    data = numpy.concatenate((data, new_data))           
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.recorded:
                rec = self.recorders[id]
                v = 1000.0*numpy.array(net.object(rec).getRecordedValues())
                v = v.flatten()
                dt = state.dt
                t = numpy.arange(0, dt*len(v), dt).flatten()              
                new_data = numpy.array([numpy.ones(v.shape)*(id-offset), t, v]).T
                data = numpy.concatenate((data, new_data))
        elif self.variable == 'gsyn':
            raise NotImplementedError
        else:
            raise Exception("Recording of %s not implemented." % self.variable)
        return data

    def write(self, file=None, gather=False, compatible_output=True):
        data = self.get(gather, offset=0)
        filename = file or self.filename
        recording.rename_existing(filename)
        try:
            numpy.savetxt(filename, data, Recorder.numpy1_0_formats[self.variable], delimiter='\t')
        except AttributeError, errmsg:
            # we assume the error is due to the lack of savetxt in older versions of numpy and
            # so provide a cut-down version of that function
            f = open(filename, 'w')
            fmt = Recorder.numpy1_0_formats[self.variable]
            for row in data:
                f.write('\t'.join([fmt%val for val in row]) + '\n')
            f.close()
        if compatible_output:
            recording.write_compatible_output(filename, filename, self.variable,
                                              Recorder.formats[self.variable],
                                              self.population, state.dt)



class Connection(object):
    
    def __init__(self, pcsim_connection, weight_unit_factor):
        self.pcsim_connection = pcsim_connection
        self.weight_unit_factor = weight_unit_factor
        
    @property
    def weight(self):
        return self.weight_unit_factor*self.pcsim_connection.W
    
    @property
    def delay(self):
        return 1000.0*self.pcsim_connection.delay # s --> ms
    

class ConnectionManager(object):
    """docstring needed."""

    def __init__(self, synapse_model='static_synapse', parent=None):
        self.parent = parent

    def __getitem__(self, i):
        """Returns a Connection object."""
        if self.parent.is_conductance:
            A = 1e6 # S --> uS
        else:
            A = 1e9 # A --> nA
        return Connection(self.parent.pcsim_projection.object(i), A)
    
    def __len__(self):
        return self.parent.pcsim_projection.size()
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
            
            
net = None
state = _State()
del _State