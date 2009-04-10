import nest
from pyNN import common, recording, random
import logging
import numpy
import os

RECORDING_DEVICE_NAMES = {'spikes': 'spike_detector',
                          'v': 'voltmeter',
                          'conductance': 'conductancemeter'}
CHECK_CONNECTIONS = False
recorder_list = []

class ID(int, common.IDMixin):

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)

    def get_native_parameters(self):
        return nest.GetStatus([int(self)])[0]

    def set_native_parameters(self, parameters):
        nest.SetStatus([self], [parameters])

def _merge_files(recorder, gather):
    """
    Combine data from multiple files (one per thread and per process) into a single file.
    Returns the filename of the merged file.
    """
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])

    # Combine data from different threads to the zeroeth thread
    merged_filename = nest.GetStatus(recorder, "filename")[0]

    if local_num_threads > 1:
        for nest_thread in range(1, local_num_threads):
            addr = nest.GetStatus(recorder, "address")[0]
            addr.append(nest_thread)
            nest_filename = nest.GetStatus([addr], "filename")[0]
            system_line = 'cat %s >> %s' % (nest_filename, merged_filename)
            os.system(system_line)
            os.remove(nest_filename)
    if gather and len(node_list) > 1:
        if rank() == 0:
            raise Exception("gather not yet implemented")
    return merged_filename


class Recorder(object):
    """Encapsulates data and functions related to recording model variables."""
    
    formats = {'spikes': 'id t',
               'v': 'id t v',
               'conductance':'id t ge gi'}
    
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
        # line needed for old version of nest 2.0
        #device_parameters.pop('to_memory')
        nest.SetStatus(self._device, device_parameters)

    def record(self, ids):
        """Add the cells in `ids` to the set of recorded cells."""
        if self._device is None:
            self._create_device()
        ids = set(ids)
        new_ids = list( ids.difference(self.recorded) )
        self.recorded = self.recorded.union(ids)
        
        device_name = nest.GetStatus(self._device, "model")[0]
        if device_name == "spike_detector":
            nest.ConvergentConnect(new_ids, self._device, model='static_synapse')
        elif device_name in ('voltmeter', 'conductancemeter'):
            nest.DivergentConnect(self._device, new_ids, model='static_synapse')
        else:
            raise Exception("%s is not a valid recording device" % device_name)
    
    def get(self, gather=False, compatible_output=True):
        """Returns the recorded data."""
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to return")
        if nest.GetStatus(self._device, 'to_file')[0]:
            if 'filename' in nest.GetStatus(self._device)[0]:
                nest_filename = _merge_files(self._device, gather)
                data = recording.readArray(nest_filename, sepchar=None)
            else:
                data = numpy.array([])
            #os.remove(nest_filename)
            if data.size > 0:
                # the following returns indices, not IDs. I'm not sure this is what we want.
                if self.population is not None:
                    padding = self.population.cell.flatten()[0]
                else:
                    padding = 0
                data[:,0] = data[:,0] - padding
            else:
                ncol = len(Recorder.formats[self.variable].split())
                data = numpy.empty([0, ncol])
        elif nest.GetStatus(self._device,'to_memory')[0]:
            data = nest.GetStatus(self._device,'events')[0]
            data = recording.convert_compatible_output(data, self.population, self.variable,compatible_output)
        return data
    
    def _get_header(self, file_name):
        header = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for line in f:
                if line[0] == '#':
                    key, value = line[1:].split("=")
                    header[key.strip()] = value.strip()
        else:
            logging.warning("File %s does not exist, so could not get header." % file_name)
        return header
    
    def _strip_header(self, input_file_name, output_file_name):
        if os.path.exists(input_file_name):
            fin = open(input_file_name, 'r')
            fout = open(output_file_name, 'a')
            for line in fin:
                if line[0] != '#':
                    fout.write(line)
            fin.close()
            fout.close()
    
    def write(self, file=None, gather=False, compatible_output=True):
        if self._device is None:
            raise common.NothingToWriteError("No cells recorded, so no data to write to file.")
        user_file = file or self.file
        if isinstance(user_file, basestring):
            if common.num_processes() > 1:
                user_file += '.%d' % rank()
            recording.rename_existing(user_file)
        logging.debug("Recorder is writing '%s' to file '%s' with gather=%s and compatible_output=%s" % (self.variable,
                                                                                                         user_file,
                                                                                                         gather,
                                                                                                         compatible_output))
        # what if the data was only saved to memory?
        if self.file is not False:
            nest_filename = _merge_files(self._device, gather)
            if compatible_output:
                # We should do the post processing (i.e the compatible output) in a distributed
                # manner to speed up the thing. The only problem that will arise is the headers, 
                # that should be taken into account to be really clean. Otherwise, if the # symbol
                # is escaped while reading the file, there is no problem
               recording.write_compatible_output(nest_filename, user_file,
                                                 self.variable,
                                                 Recorder.formats[self.variable],
                                                 self.population, common.get_time_step())
            else:
                if isinstance(user_file, basestring):
                    os.system('cat %s > %s' % (nest_filename, user_file))
                elif hasattr(user_file, 'write'):
                    nest_file = open(nest_filename)
                    user_file.write(nest_file.read())
                    nest_file.close()
                else:
                    raise Exception('Must provide a filename or an open file')
            np = common.num_processes()
            logging.debug("num_processes() = %s" % np)
            if gather == True and common.rank() == 0 and np > 1:
                root_file = file or self.filename
                logging.debug("Gathering files generated by different nodes into %s" % root_file)
                n_cells = 0
                recording.rename_existing(root_file)
                for node in xrange(np):
                    node_file = root_file + '.%d' % node
                    logging.debug("node_file = %s" % node_file)
                    if os.path.exists(node_file):
                        # merge headers
                        header = self._get_header(node_file)
                        logging.debug(str(header))
                        if header.has_key('n'):
                            n_cells += int(header['n'])
                        #os.system('cat %s >> %s' % (node_file, root_file))
                        self._strip_header(node_file, root_file)
                        os.system('rm %s' % node_file)
                # write header for gathered file
                f = open("tmp_header", 'w')
                header['n'] = n_cells
                for k,v in header.items():
                    f.write("# %s = %s\n" % (k,v))
                f.close()
                os.system('cat %s >> tmp_header' % root_file)
                os.system('mv tmp_header %s' % root_file)
            # don't want to remove nest_filename at this point in case the user wants to access the data
            # a second time (e.g. with both getSpikes() and printSpikes()), but we should
            # maintain a list of temporary files to be deleted at the end of the simulation
        else:
            raise Exception("Writing to file not yet supported for data recorded only to memory.")


class _State(object):
    """Represent the simulator state."""
    
    def __init__(self):
        self.initialized = False

    @property
    def t(self):
        return nest.GetKernelStatus()['time']
    
    #@property
    #def dt(self):
    #    return nest.GetKernelStatus()['resolution']
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
    

def run(simtime):
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
    cell_gids = nest.Create(nest_model, n)
    if cell_parameters:
        try:
            nest.SetStatus(cell_gids, [cell_parameters])
        except nest.NESTError:
            print "NEST error when trying to set the following dictionary: %s" % self.cellparams
            raise
    first_id = cell_gids[0]
    last_id = cell_gids[-1]
    mask_local = numpy.array(nest.GetStatus(cell_gids, 'local'))
    cell_gids = numpy.array([ID(gid) for gid in cell_gids], ID)
    return cell_gids, mask_local, first_id, last_id

class Connection(object):
    """Not part of the API as of 0.4."""

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
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
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
        weights = weights*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if common.is_listlike(source):
            assert len(source) == 1
            source = source[0]
        if not common.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]
        try:
            nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)
        except nest.NESTError, e:
            raise common.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, source, targets, weights, delays, self.synapse_model))
        self.sources.extend([source]*len(targets))
        self.targets.extend(targets)
        # get ports
        connections = nest.GetConnections([source], self.synapse_model)[0]
        n = len(connections['targets'])
        ports = range(n-len(targets), n)
        self.ports.extend(ports)
    
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
                values[src-offset[0], tgt-offset[1]] = value
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
                nest.SetConnection([src], self.synapse_model, port, {name: value})
        elif common.is_listlike(value):
            if name == 'weight':
                value = 1000.0*numpy.array(value)
            for src, port, val in zip(self.sources, self.ports, value):
                nest.SetConnection([src], self.synapse_model, port, {name: val})
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")


def probabilistic_connect(connector, projection, p):
    if projection.rng:
        if isinstance(projection.rng, random.NativeRNG):
            logging.warning("Warning: use of NativeRNG not implemented. Using NumpyRNG")
            rng = random.NumpyRNG()
        else:
            rng = projection.rng
    else:
        rng = random.NumpyRNG()
        
    local = projection.post._mask_local.flatten()
    is_conductance = common.is_conductance(projection.post.index(0))
    for src in projection.pre.all():
        # ( the following two lines are a nice idea, but this needs some thought for
        #   the parallel case, to ensure reproducibility when varying the number
        #   of processors
        #      N = rng.binomial(npost,self.p_connect,1)[0]
        #      targets = sample(postsynaptic_neurons, N)   # )
        N = projection.post.size
        # if running in parallel, rng.next(N) will not return N values, but only
        # as many as are needed on this node, as determined by mask_local.
        # Over the simulation as a whole (all nodes), N values will indeed be
        # returned.
        rarr = rng.next(N, 'uniform', (0, 1), mask_local=local)
        create = rarr<p
        targets = projection.post.local_cells[create].tolist()
        
        weights = connector.get_weights(N, local)[create]
        weights = common.check_weight(weights, projection.synapse_type, is_conductance)
        delays  = connector.get_delays(N, local)[create]
        
        if not connector.allow_self_connections and src in targets:
            assert len(targets) == len(weights) == len(delays)
            i = targets.index(src)
            weights = numpy.delete(weights, i)
            delays = numpy.delete(delays, i)
            targets.remove(src)
        
        if len(targets) > 0:
            projection.connection_manager.connect(src, targets, weights, delays, projection.synapse_type)
        if CHECK_CONNECTIONS:
            check_connections(projection, src, targets)


state = _State()  # a Singleton, so only a single instance ever exists
del _State