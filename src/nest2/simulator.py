import nest
from pyNN import common, recording, random
import logging
import numpy
import os
import tempfile

RECORDING_DEVICE_NAMES = {'spikes': 'spike_detector',
                          'v':      'voltmeter',
                          'gsyn':   'conductancemeter'}
CHECK_CONNECTIONS = False
recorder_list = []

class ID(int, common.IDMixin):

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)
        self._v_init = None

    def get_native_parameters(self):
        parameters = nest.GetStatus([int(self)])[0]
        if self._v_init:
            parameters['v_init'] = self._v_init
        return parameters

    def set_native_parameters(self, parameters):
        if 'v_init' in parameters:
            self._v_init = parameters.pop('v_init')
            parameters['V_m'] = self._v_init # not correct, since could set v_init in the middle of a simulation, but until we add a proper reset mechanism, this will do.
        nest.SetStatus([self], [parameters])



class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    formats = {'spikes': 'id t',
               'v':      'id t v',
               'gsyn':   'id t ge gi'}
    numpy1_1_formats = {'spikes': "%g\t%d",
                        'v': "%g\t%g\t%d",
                        'gsyn': "%g\t%g\t%g\t%d"}
    numpy1_0_formats = {'spikes': "%g", # only later versions of numpy support different
                        'v': "%g",      # formats for different columns
                        'gsyn': "%g"}
    scale_factors = {'spikes': 1,
                     'v': 1,
                     'gsyn': 0.001} # units conversion
    
    def __init__(self, variable, population=None, file=None):
        """
        `file` should be one of:
            a file-name,
            `None` (write to a temporary file)
            `False` (write to memory).
        """
        assert variable in RECORDING_DEVICE_NAMES
        self.variable = variable
        self.file = file
        self.population = population # needed for writing header information
        self.recorded = set([])        
        # we defer creating the actual device until it is needed.
        self._device = None
        if self.population is not None:
            self._id_offset = self.population.first_id
        else:
            self._id_offset = 0
        self._local_files_merged = False
        self._gathered = False
        
    def in_memory(self):
        return nest.GetStatus(self._device, 'to_memory')[0]

    def _create_device(self):
        device_name = RECORDING_DEVICE_NAMES[self.variable]
        self._device = nest.Create(device_name)
        device_parameters = {"withgid": True, "withtime": True}
        if self.variable != 'spikes':
            device_parameters["interval"] = common.get_time_step()
        if self.file is False:
            device_parameters.update(to_file=False, to_memory=True)
        else: # (includes self.file is None)
            device_parameters.update(to_file=True, to_memory=False)
        nest.SetStatus(self._device, device_parameters)

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""            
        if self._device is None:
            self._create_device()
        if self.population:
            ids = set([id for id in ids if id in self.population.local_cells])
        else:
            ids = set([id for id in ids if id.local])
        new_ids = list( ids.difference(self.recorded) )
        
        self.recorded = self.recorded.union(ids)
        
        device_name = nest.GetStatus(self._device, "model")[0]
        if device_name == "spike_detector":
            nest.ConvergentConnect(new_ids, self._device, model='static_synapse')
        elif device_name in ('voltmeter', 'conductancemeter'):
            nest.DivergentConnect(self._device, new_ids, model='static_synapse')
        else:
            raise Exception("%s is not a valid recording device" % device_name)
    
    def _make_compatible(self, data):
        # add initial values
        # NEST does not record the values at t=0 or t=simtime, so we add them now
        if self.variable == 'v':
            initial = [[id, 0.0, id.v_init] for id in self.recorded]
        elif self.variable == 'gsyn':
            initial = [[id, 0.0, 0.0, 0.0] for id in self.recorded]
        if self.recorded:
            data = numpy.concatenate((initial, data))
        # scale data
        scale_factor = Recorder.scale_factors[self.variable]
        if scale_factor != 1:
            data *= scale_factor
         # apply id offset
        data[:,0] = data[:,0] - self._id_offset
        return data
    
    def _events_to_array(self, events):
        ids = events['senders']
        times = events['times']
        if self.variable == 'spikes':
            data = numpy.array((ids, times)).T
        elif self.variable == 'v':
            data = numpy.array((ids, times, events['potentials'])).T
        elif self.variable == 'gsyn':
            data = numpy.array((ids, times, events['exc_conductance'], events['inh_conductance'])).T
        return data
                   
    def _read_data_from_memory(self, gather, compatible_output):
        data = nest.GetStatus(self._device,'events')[0] # only for spikes?
        if compatible_output:
            data = self._events_to_array(data)
            data = self._make_compatible(data)
        if gather:
            data = recording.gather(data)
        return data
                     
    def _read_data(self, gather, compatible_output):
        if gather:
            if self._gathered:
                self._gathered_file.seek(0)
                data = numpy.load(self._gathered_file)
            else:
                local_data = self._read_local_data(compatible_output)
                if self.population and hasattr(self.population.celltype, 'always_local') and self.population.celltype.always_local:
                    data = local_data # for always_local cells, no need to gather
                else:
                    data = recording.gather(local_data) 
                self._gathered_file = tempfile.TemporaryFile()
                numpy.save(self._gathered_file, data)
                self._gathered = True
            return data
        else:
            return self._read_local_data(compatible_output)
                        
    def _read_local_data(self, compatible_output):
        if self._local_files_merged:
            self._merged_file.seek(0)
            data = numpy.load(self._merged_file)
        else:
            if state.num_threads > 1:
                nest_files = []
                for nest_thread in range(1, state.num_threads):
                    addr = nest.GetStatus(self._device, "address")[0]
                    addr.append(nest_thread)
                    nest_files.append(nest.GetStatus([addr], "filename")[0])
            else:
                nest_files = [nest.GetStatus(self._device, "filename")[0]]
            # possibly we can just keep on saving to the end of self._merged_file, instead of concatenating everything in memory
            data = numpy.concatenate([recording.readArray(nest_file, sepchar=None) for nest_file in nest_files])
            if data.size == 0:
                ncol = len(Recorder.formats[self.variable].split())
                data = numpy.empty([0, ncol])
            if compatible_output:
                data = self._make_compatible(data)
            self._merged_file = tempfile.TemporaryFile()
            numpy.save(self._merged_file, data)
            self._local_files_merged = True
        return data
    
    def get(self, gather=False, compatible_output=True):
        """Returns the recorded data."""
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to return")
        
        if self.in_memory():
            data = self._read_data_from_memory(gather, compatible_output)
        else: # in file
            data = self._read_data(gather, compatible_output)
        return data
    
    def write(self, file=None, gather=False, compatible_output=True):
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to write to file.")
        user_file = file or self.file
        if isinstance(user_file, basestring):
            if gather==False and state.num_processes > 1:
                user_file += '.%d' % state.mpi_rank
            #recording.rename_existing(user_file)
        logging.debug("Recorder is writing '%s' to file '%s' with gather=%s and compatible_output=%s" % (self.variable,
                                                                                                         user_file,
                                                                                                         gather,
                                                                                                         compatible_output))
        data = self.get(gather, compatible_output)
        dt = state.dt
        if state.mpi_rank == 0 or gather==False:
            numpy.savetxt(user_file, data, Recorder.numpy1_0_formats[self.variable], delimiter='\t')
            if compatible_output:
                recording.write_compatible_output(user_file, user_file, self.variable,
                                                  Recorder.formats[self.variable],
                                                  self.population, dt)

    def count(self, gather=False):
        """
        Return the number of data points for each cell, as a dict. This is mainly
        useful for spike counts or for variable-time-step integration methods.
        """
        N = {}
        if self.variable == 'spikes':
            if nest.GetStatus(self._device,'to_memory')[0]:
                events = nest.GetStatus(self._device, 'events')
                for id in self.recorded:
                    mask = events['senders'] == int(id)
                    N[id] = events['times'][mask].count()
            else:
                raise Exception("count() not yet supported for writing to file.")
        else:
            raise Exception("Only implemented for spikes.")
        if gather and state.num_processes > 1:
            N = recording.gather_dict(N)
        return N


class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False
        self.running = False

    @property
    def t(self):
        return nest.GetKernelStatus()['time']
    
    dt = property(fget=lambda self: nest.GetKernelStatus()['resolution'],
                  fset=lambda self, timestep: nest.SetKernelStatus({'resolution': timestep}))
    
    @property
    def min_delay(self):
        return nest.GetDefaults('static_synapse')['min_delay']
    
    @property
    def max_delay(self):
        # any reason why not nest.GetKernelStatus()['min_delay']?
        return nest.GetDefaults('static_synapse')['max_delay']
    
    @property
    def num_processes(self):
        return nest.GetKernelStatus()['num_processes']
    
    @property
    def mpi_rank(self):
        return nest.Rank()
    
    @property
    def num_threads(self):
        return nest.GetKernelStatus()['local_num_threads']


def run(simtime):
    if not state.running:
        simtime += state.dt # we simulate past the real time by one time step, otherwise NEST doesn't give us all the recorded data
        state.running = True
    nest.Simulate(simtime)

def create_cells(cellclass, cellparams=None, n=1, parent=None):
    """
    Function used by both `create()` and `Population.__init__()`
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, basestring):  # celltype is not a standard cell
        nest_model = cellclass
        cell_parameters = cellparams or {}
    elif isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
        celltype = cellclass(cellparams)
        nest_model = celltype.nest_name
        cell_parameters = celltype.parameters
    else:
        raise Exception("Invalid cell type: %s" % type(cellclass))
    try:
        cell_gids = nest.Create(nest_model, n)
    except nest.NESTError, errmsg:
        raise common.InvalidModelError(errmsg)
    if cell_parameters:
        try:
            v_init = cell_parameters.pop('v_init', None)
            if v_init:
                cell_parameters['V_m'] = v_init
            nest.SetStatus(cell_gids, [cell_parameters])
        except nest.NESTError:
            print "NEST error when trying to set the following dictionary: %s" % cell_parameters
            raise
    first_id = cell_gids[0]
    last_id = cell_gids[-1]
    mask_local = numpy.array(nest.GetStatus(cell_gids, 'local'))
    cell_gids = numpy.array([ID(gid) for gid in cell_gids], ID)
    for gid, local in zip(cell_gids, mask_local):
        gid.local = local
    if cell_parameters and v_init:
        for cell in cell_gids:
            cell._v_init = v_init
    return cell_gids, mask_local, first_id, last_id

class Connection(object):

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def id(self):
        src = self.parent.sources[self.index]
        port = self.parent.ports[self.index]
        synapse_model = self.parent.synapse_model
        return [[src], synapse_model, port]

    @property
    def source(self):
        return self.parent.sources[self.index]
    
    @property
    def target(self):
        return self.parent.targets[self.index]

    @property
    def port(self):
        return self.parent.ports[self.index]

    def _set_weight(self, w):
        args = self.id() + [{'weight': w*1000.0}]
        nest.SetConnection(*args)

    def _get_weight(self):
        return 0.001*nest.GetConnection(*self.id())['weight']

    def _set_delay(self, d):
        args = self.id() + [{'delay': d}]
        nest.SetConnection(*args)

    def _get_delay(self):
        return nest.GetConnection(*self.id())['delay']

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)
    

class ConnectionManager:
    """docstring needed."""

    def __init__(self, synapse_model='static_synapse', parent=None):
        self.sources = []
        self.targets = []
        self.ports = []
        self.synapse_model = synapse_model
        self.parent = parent
        if parent is not None:
            assert parent.plasticity_name == self.synapse_model

    def __getitem__(self, i):
        """Returns a Connection object."""
        if i < len(self):
            return Connection(self, i)
        else:
            raise IndexError("%d > %d" % (i, len(self)-1))
    
    def __len__(self):
        return len(self.sources)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def connect(self, source, targets, weights, delays, synapse_type):
        """
        Connect a neuron to one or more other neurons.
        """
        # are we sure the targets are all on the current node?
        if common.is_listlike(source):
            assert len(source) == 1
            source = source[0]
        if not common.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0
        weights = weights*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if synapse_type == 'inhibitory' and common.is_conductance(targets[0]):
            weights = -1*weights # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]
        
        initial_ports = {}
        for tgt in targets:
            try:
                initial_ports[tgt] = nest.FindConnections([source], [tgt], self.synapse_model)['ports']
            except nest.NESTError, e:
                raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, source, targets, weights, delays, self.synapse_model))
            
        try:
            nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)
        except nest.NESTError, e:
            raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, source, targets, weights, delays, self.synapse_model))
        
        final_ports = {}
        for tgt in targets:
            final_ports[tgt] = nest.FindConnections([source], [tgt], self.synapse_model)['ports']
        
        #print "\n", state.mpi_rank, source, "initial:", initial_ports, "final:", final_ports
        
        #all_new_ports = []
        local_targets = final_ports.keys()
        local_targets.sort()
        for tgt in local_targets:
            #new_ports = final_ports[tgt].difference(initial_ports[tgt])
            new_ports = final_ports[tgt][len(initial_ports[tgt]):]
            #if state.mpi_rank == 0:
            #    print "-", state.mpi_rank, tgt, initial_ports[tgt], final_ports[tgt], new_ports
            n = len(new_ports)
            if n > 0:   
                self.sources.extend([source]*n)
                self.targets.extend([tgt*n])     
                self.ports.extend(new_ports)
                #all_new_ports.extend(new_ports)
                
        #print state.mpi_rank, source, targets, all_new_ports, nest.GetConnections([source], self.synapse_model)[0]['targets']
    
    def get(self, parameter_name, format, offset=(0,0)):
        # this is a slow implementation, going through each connection one at a time
        # better to use GetConnections, which means we should probably store
        # connections in a dict, with source as keys and a list of ports as values
        if format == 'list':
            values = []
            for src, port in zip(self.sources, self.ports):
                value = nest.GetConnection([src], self.synapse_model, port)[parameter_name]
                if parameter_name == "weight":
                    value *= 0.001
                values.append(value)
        elif format == 'array':
            values = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            for src, tgt, port in zip(self.sources, self.targets, self.ports):
                # could instead get tgt from the 'target' entry with GetConnection
                value = nest.GetConnection([src], self.synapse_model, port)[parameter_name]
                # don't need to pass offset as arg, now we store the parent projection
                # (offset is always 0,0 for connections created with connect())
                addr = (src-offset[0], tgt-offset[1])
                if numpy.isnan(values[addr]):
                    values[addr] = value
                else:
                    values[addr] += value
            if parameter_name == 'weight':
                values *= 0.001
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values
    
    def set(self, name, value):
        if common.is_number(value):
            if name == 'weight':
                value *= 1000.0
            for src, port in zip(self.sources, self.ports):
                nest.SetConnection([src], self.synapse_model, port, {name: float(value)})
        elif common.is_listlike(value):
            if name == 'weight':
                value = 1000.0*numpy.array(value)
            for src, port, val in zip(self.sources, self.ports, value):
                #if state.mpi_rank == 0:
                #    print state.mpi_rank, "setting:", src, port, val
                nest.SetConnection([src], self.synapse_model, port, {name: val})
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")


state = _State()  # a Singleton, so only a single instance ever exists
del _State