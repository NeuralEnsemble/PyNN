# encoding: utf-8
import logging
import pypcsim
import types
import numpy
from pyNN import common, recording

recorder_list = []

def reset():
    net.reset()
    state.t = 0.0

class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.t = 0.0
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

    dt = property(fget=lambda self: net.get_dt().in_ms(),
                  fset=lambda self,x: net.set_dt(pypcsim.Time.ms(x)))


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    fieldnames = {'v': 'Vm',
                  'gsyn': 'psr'}
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
                #if self.population:
                #    pcsim_id = self.population.pcsim_population[int(id)]
                #else:
                pcsim_id = int(id)
                src_id = pypcsim.SimObject.ID(pcsim_id)
                rec = net.create(pypcsim.SpikeTimeRecorder(),
                                 pypcsim.SimEngine.ID(src_id.node, src_id.eng))            
                net.connect(pcsim_id, rec, pypcsim.Time.sec(0))
                assert id not in self.recorders
                self.recorders[id] = rec
        elif self.variable == 'v':
            for id in new_ids:
                #if self.population:
                #    pcsim_id = self.population.pcsim_population[int(id)]
                #else:
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
                offset = self.population.first_id
            else:
                offset = 0
                
        if self.variable == 'spikes':
            data = numpy.empty((0,2))
            for id in self.recorded:
                rec = self.recorders[id]
                if isinstance(net.object(id), pypcsim.SpikingInputNeuron):
                    spikes = 1000.0*numpy.array(net.object(id).getSpikeTimes()) # is this special case really necessary?
                    spikes = spikes[spikes<=state.t]
                else:
                    spikes = 1000.0*numpy.array(net.object(rec).getSpikeTimes())
                spikes = spikes.flatten()
                spikes = spikes[spikes<=state.t+1e-9]
                if len(spikes) > 0:    
                    new_data = numpy.array([numpy.ones(spikes.shape, dtype=int)*(id-offset), spikes]).T
                    data = numpy.concatenate((data, new_data))           
        elif self.variable == 'v':
            data = numpy.empty((0,3))
            for id in self.recorded:
                rec = self.recorders[id]
                v = 1000.0*numpy.array(net.object(rec).getRecordedValues())
                v = v.flatten()
                dt = state.dt
                t = numpy.arange(0, dt*len(v), dt).flatten()              
                new_data = numpy.array([numpy.ones(v.shape, dtype=int)*(id-offset), t, v]).T
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

    def count(self, gather=False):
        N = {}
        if self.variable == 'spikes':
            for id in self.recorded:
                N[id] = net.object(self.recorders[id]).spikeCount()
        else:
            raise Exception("Only implemented for spikes.")
        if gather and state.num_processes > 1:
            N = recording.gather_dict(N)
        return N
              

class ID(long, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object.
    
    """
    
    def __init__(self, n):
        long.__init__(n)
        common.IDMixin.__init__(self)
    
    def _pcsim_cell(self):
        global net
        #if self.parent:
        #    pcsim_cell = self.parent.pcsim_population.object(self)
        #else:
        pcsim_cell = net.object(self)
        return pcsim_cell
    
    def get_native_parameters(self):
        pcsim_cell = self._pcsim_cell()
        pcsim_parameters = {}
        if self.is_standard_cell():
            parameter_names = [D['translated_name'] for D in self.cellclass.translations.values()]
        else:
            parameter_names = [] # for native cells, is there a way to get their list of parameters?
        
        for translated_name in parameter_names:
            if hasattr(self.cellclass, 'getterMethods') and translated_name in self.cellclass.getterMethods:
                getterMethod = self.cellclass.getterMethods[translated_name]
                pcsim_parameters[translated_name] = getattr(pcsim_cell, getterMethod)()    
            else:
                try:
                    pcsim_parameters[translated_name] = getattr(pcsim_cell, translated_name)
                except AttributeError, e:
                    raise AttributeError("%s. Possible attributes are: %s" % (e, dir(pcsim_cell)))
        for k,v in pcsim_parameters.items():
            if isinstance(v, pypcsim.StdVectorDouble):
                pcsim_parameters[k] = list(v)
        return pcsim_parameters
    
    def set_native_parameters(self, parameters):
        simobj = self._pcsim_cell()
        for name, value in parameters.items():
            if hasattr(self.cellclass, 'setterMethods') and name in self.cellclass.setterMethods:
                setterMethod = self.cellclass.setterMethods[name]
                getattr(simobj, setterMethod)(value)
            else:               
                setattr(simobj, name, value)

def is_local(id):
    return pypcsim.SimObject.ID(id).node == net.mpi_rank()

def create_cells(cellclass, cellparams, n, parent=None):
    """
    Function used by both `create()` and `Population.__init__()`
    """
    global net
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        if issubclass(cellclass, common.StandardCellType):
            cellfactory = cellclass(cellparams).simObjFactory
        elif issubclass(cellclass, pypcsim.SimObject):
            #cellfactory = apply(cellclass, (), cellparams)
            cellfactory = cellclass(**cellparams)
        else:
            raise common.InvalidModelError("Trying to create non-existent cellclass %s" % cellclass.__name__)
    else:
        raise common.InvalidModelError("Trying to create non-existent cellclass %s" % cellclass)

    all_ids = numpy.array([i for i in net.add(cellfactory, n)], ID)
    first_id = all_ids[0]
    last_id = all_ids[-1]
    # mask_local is used to extract those elements from arrays that apply to the cells on the current node
    mask_local = numpy.array([is_local(id) for id in all_ids])
    for i,(id,local) in enumerate(zip(all_ids, mask_local)):
        #if local:
        all_ids[i] = ID(id)
        all_ids[i].parent = parent
        all_ids[i].local = local
    return all_ids, mask_local, first_id, last_id


class Connection(object):
    
    def __init__(self, source, target, pcsim_connection, weight_unit_factor):
        self.source = source
        self.target = target
        self.pcsim_connection = pcsim_connection
        self.weight_unit_factor = weight_unit_factor
        
    def _get_weight(self):
        return self.weight_unit_factor*self.pcsim_connection.W
    def _set_weight(self, w):
        self.pcsim_connection.W = w/self.weight_unit_factor
    weight = property(fget=_get_weight, fset=_set_weight)
    
    def _get_delay(self):
        return 1000.0*self.pcsim_connection.delay # s --> ms
    def _set_delay(self, d):
        self.pcsim_connection.delay = 0.001*d
    delay = property(fget=_get_delay, fset=_set_delay)
    

class ConnectionManager(object):
    """docstring needed."""

    synapse_target_ids = { 'excitatory': 1, 'inhibitory': 2 }

    def __init__(self, synapse_model='static_synapse', parent=None):
        self.parent = parent
        #if parent is None:
        self.connections = []

    def __getitem__(self, i):
        """Returns a Connection object."""
        #if self.parent:
        #    if self.parent.is_conductance:
        #        A = 1e6 # S --> uS
        #    else:
        #        A = 1e9 # A --> nA
        #    return Connection(self.parent.pcsim_projection.object(i), A)
        #else:
        return self.connections[i]
    
    def __len__(self):
        #if self.parent:
        #    return self.parent.pcsim_projection.size()
        #else:
        return len(self.connections)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def connect(self, source, targets, weights, delays, synapse_type):
        """
        Connect a neuron to one or more other neurons.
        """
        if not isinstance(source, (int, long)) or source < 0:
            errmsg = "Invalid source ID: %s" % source
            raise common.ConnectionError(errmsg)
        if not common.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise common.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        if common.is_conductance(targets[0]):
            weight_scale_factor = 1e-6 # Convert from µS to S  
        else:
            weight_scale_factor = 1e-9 # Convert from nA to A
        
        synapse_type = synapse_type or "excitatory"
        syn_target_id = ConnectionManager.synapse_target_ids[synapse_type]
                
        if isinstance(synapse_type, basestring):
            syn_factory = pypcsim.SimpleScalingSpikingSynapse(
                              syn_target_id, weights[0], delays[0])
        elif isinstance(synapse_type, pypcsim.SimObject):
            syn_factory = synapse_type
        else:
            raise Exception("synapse_type must be a string or a PCSIM synapse factory.")
        for target, weight, delay in zip(targets, weights, delays):
            syn_factory.W = weight*weight_scale_factor
            syn_factory.delay = delay*0.001 # ms --> s
            try:
                c = net.connect(source, target, syn_factory)
            except RuntimeError, e:
                raise common.ConnectionError(e)
            if target.local:
                self.connections.append(Connection(source, target, net.object(c), 1.0/weight_scale_factor))
            
    def get(self, parameter_name, format, offset=(0,0)):
        if format == 'list':
            values = [getattr(c, parameter_name) for c in self]
        elif format == 'array':
            values = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            for c in self:
                if self.parent:
                    addr = (self.parent.pre.id_to_index(c.source), self.parent.post.id_to_index(c.target))
                else:
                    addr = (c.source-offset[0], c.target-offset[1])
                values[addr] = getattr(c, parameter_name)
        else:
            raise Exception("format must be 'list' or 'array'")
        return values        
    
    def set(self, name, value):
        if common.is_number(value):
            for c in self:
                setattr(c, name, value)
        elif common.is_listlike(value):
            for c,val in zip(self.connections, value):
                setattr(c, name, val)
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")

            
net = None
state = _State()
del _State