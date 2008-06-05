# -*- coding: utf-8 -*-
"""
PyNEST v2 implementation of the PyNN API.
$Id:__init__.py 188 2008-01-29 10:03:59Z apdavison $
"""
__version__ = "$Rev:188 $"

import nest
from pyNN import common
from pyNN.random import *
from pyNN import recording
import numpy, types, sys, shutil, os, logging, copy, tempfile
from math import *
from pyNN.nest2.cells import *
from pyNN.nest2.connectors import *
from pyNN.nest2.synapses import *
Set = set

recorder_dict  = {}
tempdirs       = []
recording_device_names = {'spikes': 'spike_detector',
                          'v': 'voltmeter',
                          'conductance': 'conductancemeter'}
DEFAULT_BUFFER_SIZE = 10000
NEST_SYNAPSE_TYPES = ["cont_delay_synapse" ,"static_synapse", "stdp_pl_synapse_hom",
                      "stdp_synapse", "stdp_synapse_hom", "tsodyks_synapse"]

# ==============================================================================
#   Utility classes and functions
# ==============================================================================

class ID(int, common.IDMixin):
    """
    Instead of storing ids as integers, we store them as ID objects,
    which allows a syntax like:
        p[3,4].tau_m = 20.0
    where p is a Population object. The question is, how big a memory/performance
    hit is it to replace integers with ID objects?
    """

    def __init__(self, n):
        int.__init__(n)
        common.IDMixin.__init__(self)

    def get_native_parameters(self):
        return nest.GetStatus([int(self)])[0]

    def set_native_parameters(self, parameters):
        nest.SetStatus([self], [parameters])


class Connection(object):
    """Not part of the API as of 0.4."""

    def __init__(self, pre, post, synapse_model):
        self.pre = pre
        self.post = post
        self.synapse_model = synapse_model
        try:
            conn_dict = nest.GetConnections([pre], self.synapse_model)[0]
        except Exception:
            raise common.ConnectionError
        if (len(conn_dict['targets']) == 0):
            raise common.ConnectionError
        if conn_dict:
            self.port = len(conn_dict['targets'])-1
        else:
            raise Exception("Could not get port number for connection between %s and %s" % (pre, post))

    def _set_weight(self, w):
        pass

    def _get_weight(self):
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
        conn_dict = nest.GetConnections([self.pre], self.synapse_model)[0]
        if conn_dict:
            return conn_dict['weights'][self.port]
        else:
            return None

    def _set_delay(self, d):
        pass

    def _get_delay(self):
        # this needs to be modified to take account of threads
        # also see nest.GetConnection (was nest.GetSynapseStatus)
        conn_dict = nest.GetConnections([self.pre],'static_synapse')[0]
        if conn_dict:
            return conn_dict['delays'][self.port]
        else:
            return None

    weight = property(_get_weight, _set_weight)
    delay = property(_get_delay, _set_delay)

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, common.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return standard_cell_types

def _discrepancy_due_to_rounding(parameters, output_values):
    """NEST rounds delays to the time step."""
    if 'delay' not in parameters:
        return False
    else:
        # the logic here is not the clearest, the aim was to keep
        # _set_connection() as simple as possible, but it might be better to
        # refactor the whole thing.
        input_delay = parameters['delay']
        if hasattr(output_values, "__len__"):
            output_delay = output_values[parameters.keys().index('delay')]
        else:
            output_delay = output_values
        return abs(input_delay - output_delay) < get_time_step()

def _set_connection(source_id, target_id, synapse_type, **parameters):
    """target_id is a port."""
    nest.SetConnection([source_id], synapse_type, target_id, parameters)
    # check, since NEST ignores connection errors, rather than raising an Exception
    input_values = [v for v in parameters.values()]
    if len(input_values)==1:
        input_values = input_values[0]
    output_values = _get_connection(source_id, target_id, synapse_type, *parameters.keys())
    if input_values != output_values:
        # The problem must be with parameter values, otherwise _get_connection()
        # would have raised an exception
        # There is one special case: delays are rounded to the time step precision in NEST
        if _discrepancy_due_to_rounding(parameters, output_values):
            raise common.RoundingWarning("delays rounded to the precision of the timestep.")
        else:
            raise common.ConnectionError("Invalid parameter value(s): %(parameters)s [%(input_values)s != %(output_values)s]" % locals())

def _set_connections(source_id, synapse_type, **parameters):
    n = len( nest.GetConnections([source_id], synapse_type)[0]['targets'] )
    for name, value in parameters.items():
        name += 's'
        if is_number(value):
            value = [value]*n
        nest.SetConnections([source_id], synapse_type, [{name:value}])
    # need to check that the value has been set    

def _get_connection(source_id, target_id, synapse_type, *parameter_names):
    try:
        conn_dict = nest.GetConnection([source_id], synapse_type, target_id)
    except Exception, e:
        err_msg = str(e) + "\n  Problem getting connection from %s to %s with synapse type '%s'." % (source_id, target_id, synapse_type)
        err_msg += "\n  Valid connections for source %s: %s" % (source_id, nest.GetConnections([source_id], synapse_type)[0])
        raise common.ConnectionError(err_msg)
    if isinstance(conn_dict, dict):
        if len(parameter_names) == 1:
            return conn_dict[parameter_names[0]]
        else:
            return [conn_dict[p] for p in parameter_names]
    else:
        assert synapse_type in NEST_SYNAPSE_TYPES, "Invalid synapse type: '%s'" % synapse_type
        conn_dict = nest.GetConnections([source_id], synapse_type)[0]
        if isinstance(conn_dict, dict): # valid dict returned, so target_id must be the problem
            raise common.ConnectionError("Invalid target_id (%s). Valid target_ids for source_id=%s are: %s" % (
                target_id, source_id, range(len(conn_dict['weights']))))
        elif isinstance(conn_dict, basestring):
            raise common.ConnectionError("Invalid source_id (%s) or target_id (port) (%s)" % (source_id, target_id))
        else:
            raise Exception("Internal error: type(conn_dict) == %s" % type(conn_dict))

def is_number(n):
    return type(n) == types.FloatType or type(n) == types.IntType or type(n) == numpy.float64

def _convertWeight(w, synapse_type):
    weight = w*1000.0
    if isinstance(w, numpy.ndarray):
        all_negative = (weight<=0).all()
        all_positive = (weight>=0).all()
        assert all_negative or all_positive, "Weights must be either all positive or all negative"
        if synapse_type == 'inhibitory':
            if all_positive:
                weights *= -1
    elif is_number(weight):
        if synapse_type == 'inhibitory' and weight > 0:
            weight *= -1
    else:
        raise TypeError("w must be either a number or a numpy array")
    return weight

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    global tempdir
    assert min_delay >= timestep, "min_delay (%g) must be greater than timestep (%g)" % (min_delay, timestep)

    # reset the simulation kernel
    nest.ResetKernel()
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')

    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list

    # set tempdir
    nest.SetStatus([0], {'device_prefix':tempdir,})
    # set resolution
    nest.SetStatus([0], {'resolution': timestep})

    # set kernel RNG seeds
    num_threads = extra_params.get('threads') or 1
    if 'rng_seeds' in extra_params:
        rng_seeds = extra_params['rng_seeds']
    else:
        rng = NumpyRNG(42)
        rng_seeds = (rng.rng.uniform(size=num_threads*num_processes())*100).astype('int').tolist()
    logging.debug("rng_seeds = %s" % rng_seeds)

    nest.SetStatus([0],[{'local_num_threads': num_threads,
                         'rng_seeds'        : rng_seeds}])

    # Set min_delay and max_delay for all synapse models
    for synapse_model in NEST_SYNAPSE_TYPES:
        # this is done in two steps, because otherwise NEST sometimes complains
        #   "max_delay is not compatible with default delay"
        nest.SetSynapseDefaults(synapse_model, {'delay': min_delay})
        nest.SetSynapseDefaults(synapse_model, {'min_delay': min_delay,
                                                'max_delay': max_delay})

    # Initialisation of the log module. To write in the logfile, simply enter
    # logging.critical(), logging.debug(), logging.info(), logging.warning()
    if debug:
        if isinstance(debug, basestring):
            filename = debug
        else:
            filename = "nest.log"
        logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=filename,
                    filemode='w')
    else:
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='nest.log',
                    filemode='w')

    logging.info("Initialization of Nest")
    return nest.Rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    # We close the high level files opened by populations objects
    # that may have not been written.

    # NEST will soon close all its output files after the simulate function is over, therefore this step is not necessary
    global tempdirs

    # And we postprocess the low level files opened by record()
    # and record_v() method
    print "Saving the following files:", recorder_dict.keys()
    for filename in recorder_dict.keys():
        _print(filename, gather=False, compatible_output=compatible_output)

    for tempdir in tempdirs:
        os.system("rm -rf %s" %tempdir)

def run(simtime):
    """Run the simulation for simtime ms."""
    nest.Simulate(simtime)
    return get_current_time()

def get_current_time():
    """Return the current time in the simulation."""
    return nest.GetStatus([0])[0]['time']

def get_time_step():
    return nest.GetStatus([0])[0]['resolution']
common.get_time_step = get_time_step

def get_min_delay():
    return nest.GetSynapseDefaults('static_synapse')['min_delay']
common.get_min_delay = get_min_delay

def get_max_delay():
    return nest.GetSynapseDefaults('static_synapse')['max_delay']
common.get_max_delay = get_max_delay

def num_processes():
    return nest.GetStatus([0])[0]['num_processes']

def rank():
    """Return the MPI rank."""
    return nest.Rank()

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def create(cellclass, param_dict=None, n=1):
    """
    Create n cells all of the same type.
    If n > 1, return a list of cell ids/references.
    If n==1, return just the single id.
    """
    assert n > 0, 'n must be a positive integer'
    if isinstance(cellclass, type):
        celltype = cellclass(param_dict)
        cell_gids = nest.Create(celltype.nest_name, n)
        cell_gids = [ID(gid) for gid in cell_gids]
        nest.SetStatus(cell_gids, [celltype.parameters])
    elif isinstance(cellclass, str):  # celltype is not a standard cell
        cell_gids = nest.Create(cellclass, n)
        cell_gids = [ID(gid) for gid in cell_gids]
        if param_dict:
            nest.SetStatus(cell_gids, [param_dict])
    else:
        raise Exception("Invalid cell type")
    for id in cell_gids:
    #    #id.setCellClass(cellclass)
        id.cellclass = cellclass
    if n == 1:
        return cell_gids[0]
    else:
        return cell_gids

def connect(source, target, weight=None, delay=None, synapse_type=None, p=1, rng=None):
    """Connect a source of spikes to a synaptic target. source and target can
    both be individual cells or lists of cells, in which case all possible
    connections are made with probability p, using either the random number
    generator supplied, or the default rng otherwise.
    Weights should be in nA or µS."""
    if weight is None:
        weight = 0.0
    if delay is None:
        delay = get_min_delay()
    # If the delay is too small , we have to throw an error
    if delay < get_min_delay() or delay > get_max_delay():
        raise common.ConnectionError("delay (%s) is out of range [%s,%s]" % (delay, get_min_delay(), get_max_delay()))
    weight = weight*1000 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                         # Using convention in this way is not ideal. We should
                         # be able to look up the units used by each model somewhere.
    if synapse_type == 'inhibitory' and weight > 0:
        weight *= -1
    try:
        if type(source) != types.ListType and type(target) != types.ListType:
            nest.ConnectWD([source], [target], [weight], [delay])
            connect_id = Connection(source, target, 'static_synapse')
        else:
            connect_id = []
            if type(source) != types.ListType:
                source = [source]
            if type(target) != types.ListType:
                target = [target]
            for src in source:
                if p < 1:
                    if rng: # use the supplied RNG
                        rarr = rng.rng.uniform(0, 1, len(target))
                    else:   # use the default RNG
                        rarr = numpy.random.uniform(0, 1, len(target))
                for j,tgt in enumerate(target):
                    if p >= 1 or rarr[j] < p:
                        nest.ConnectWD([src], [tgt], [weight], [delay])
                        connect_id += [Connection(src, tgt, 'static_synapse')]
    #except nest.SLIError:
    except Exception: # unfortunately, SLIError seems to have disappeared.Hopefully it will be reinstated.
        raise common.ConnectionError
    return connect_id

def set(cells, param, val=None):
    """Set one or more parameters of an individual cell or list of cells.
    param can be a dict, in which case val should not be supplied, or a string
    giving the parameter name, in which case val is the parameter value.
    """
    if val:
        param = {param:val}
    if not hasattr(cells, '__len__'):
        cells = [cells]
    # see comment in Population.set() below about the efficiency of the
    # following
    for cell in cells:
        cell.set_parameters(**param)

def _connect_recording_device(recorder, record_from=None):
    #print "Connecting recorder %s to cell(s) %s" % (recorder, record_from)
    device = nest.GetStatus(recorder, "model")[0]
    if device == "spike_detector":
        nest.ConvergentConnect(record_from, recorder)
    elif device in ('voltmeter', 'conductancemeter'):
        nest.DivergentConnect(recorder, record_from)
    else:
        raise Exception("Not a valid recording device")

def _record(variable, source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    device_name = recording_device_names[variable]

    recording_device = nest.Create(device_name)
    
    ss_dict = {"to_file" : True, "withgid" : True, "withtime" : True}
    
    # check for older nest2 with need for interval
    if len(nest.GetStatus(recording_device, 'interval')) == 1: # returns a list of length 1 if 'interval' is present, otherwise returns the whole dict 
        print "PyNN Warning: nest2 recording_device.interval detected."
        print "Please upgrade to a more recent version of nest 2"
        print "Transition code only temporarily supported."
        ss_dict['interval'] = nest.GetStatus([0],"resolution")[0]
        
    nest.SetStatus(recording_device,ss_dict)

    print "Trying to record %s from cell %s using %s %s (filename=%s)" % (variable, source, device_name,
                                                                          recording_device, filename)

    if type(source) != types.ListType:
        source = [source]
    _connect_recording_device(recording_device, record_from=source)
    if filename is not None:
        recorder_dict[filename] = recording_device

def record(source, filename):
    """Record spikes to a file. source can be an individual cell or a list of
    cells."""
    # would actually like to be able to record to an array and choose later
    # whether to write to a file.
    _record('spikes', source, filename)

def record_v(source, filename):
    """
    Record membrane potential to a file. source can be an individual cell or
    a list of cells."""
    # would actually like to be able to record to an array and
    # choose later whether to write to a file.
    _record('v', source, filename)

def _merge_files(recorder, gather):
    """
    Combine data from multiple files (one per thread and per process) into a single file.
    Returns the filename of the merged file.
    """
    nest.FlushDevice(recorder)
    status = nest.GetStatus([0])[0]
    local_num_threads = status['local_num_threads']
    node_list = range(nest.GetStatus([0], "num_processes")[0])

    # Combine data from different threads to the zeroeth thread
    nest.sps(recorder[0])
    nest.sr("%i GetAddress %i append" % (recorder[0], 0))
    nest.sr("GetStatus /filename get")
    merged_filename = nest.spp() #nest.GetStatus(recorder, "filename")

    if local_num_threads > 1:
        for nest_thread in range(1, local_num_threads):
            nest.sps(recorder[0])
            nest.sr("%i GetAddress %i append" % (recorder[0], nest_thread))
            nest.sr("GetStatus /filename get")
            nest_filename = nest.spp() #nest.GetStatus(recorder, "filename")
            system_line = 'cat %s >> %s' % (nest_filename, merged_filename)
            os.system(system_line)
            os.remove(nest_filename)
    if gather and len(node_list) > 1:
        raise Exception("gather not yet implemented")
    return merged_filename

def _print(user_filename, gather=True, compatible_output=True, population=None,
           variable=None):
    global recorder_dict

    if population is None:
        recorder = recorder_dict[user_filename]
    else:
        assert variable in ['spikes', 'v', 'conductance']
        recorder = population.recorders[variable]

    logging.info("Printing to %s from recorder %s (compatible_output=%s)" % (user_filename, recorder,
                                                                             compatible_output))
    nest_filename = _merge_files(recorder, gather)

    if compatible_output:
        if gather == False and num_processes() > 1:
            user_filename += '.%d' % rank()
        if gather == False or rank() == 0: # if we gather, only do this on the master node
            recording.write_compatible_output(nest_filename, user_filename,
                                              population, get_time_step())
    else:
        system_line = 'cat %s >> %s' % (nest_filename, user_filename)
        os.system(system_line)

    os.remove(nest_filename)
    if population is None:
        recorder_dict.pop(user_filename)

def _get_recorded_data(population, variable=None):
    global recorder_dict

    assert variable in ['spikes', 'v', 'conductance']
    recorder = population.recorders[variable]

    nest_filename = _merge_files(recorder, gather=True)
    data = recording.readArray(nest_filename, sepchar=None)
    os.remove(nest_filename)

    if data.size > 0:
        if population is not None:
            padding = population.cell.flatten()[0]
        else:
            padding = 0
        data[:,0] = data[:,0] - padding

    return data


# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    nPop = 0

    def __init__(self, dims, cellclass, cellparams=None, label=None):
        """
        dims should be a tuple containing the population dimensions, or a single
          integer, for a one-dimensional population.
          e.g., (10,10) will create a two-dimensional population of size 10x10.
        cellclass should either be a standardized cell class (a class inheriting
        from common.StandardCellType) or a string giving the name of the
        simulator-specific model that makes up the population.
        cellparams should be a dict which is passed to the neuron model
          constructor
        label is an optional name for the population.
        """

        common.Population.__init__(self, dims, cellclass, cellparams, label)

        # Should perhaps use "LayoutNetwork"?

        if isinstance(cellclass, type):
            self.celltype = cellclass(cellparams)
            self.cell = nest.Create(self.celltype.nest_name, self.size)
            self.cellparams = self.celltype.parameters
        elif isinstance(cellclass, str):
            self.cell = nest.Create(cellclass, self.size)

        self.cell = numpy.array([ ID(GID) for GID in self.cell ], ID)
        self.cell_local = self.cell[numpy.array(nest.GetStatus(self.cell.tolist(),'local'))]
        self.id_start = self.cell.reshape(self.size,)[0]

        for id in self.cell:
            id.parent = self
            #id.setCellClass(cellclass)
            #id.setPosition(self.locate(id))

        if self.cellparams:
            nest.SetStatus(self.cell_local, [self.cellparams])

        self.cell = numpy.reshape(self.cell, self.dim)

        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {'spikes': None, 'v': None, 'conductance': None}
        self.recorded = {'spikes': Set(), 'v': Set(), 'conductance': Set()}
        Population.nPop += 1

    def __getitem__(self, addr):
        """Return a representation of the cell with coordinates given by addr,
           suitable for being passed to other methods that require a cell id.
           Note that __getitem__ is called when using [] access, e.g.
             p = Population(...)
             p[2,3] is equivalent to p.__getitem__((2,3)).
        """
        if isinstance(addr, int):
            addr = (addr,)
        if len(addr) == self.ndim:
            id = self.cell[addr]
        else:
            raise common.InvalidDimensionsError, "Population has %d dimensions. Address was %s" % (self.ndim, str(addr))
        if addr != self.locate(id):
            raise IndexError, 'Invalid cell address %s' % str(addr)
        return id

    def __iter__(self):
        """Iterator over cell ids."""
        return self.cell.flat

    def __address_gen(self):
        """
        Generator to produce an iterator over all cells on this node,
        returning addresses.
        """
        for i in self.__iter__():
            yield self.locate(i)

    def addresses(self):
        """Iterator over cell addresses."""
        return self.__address_gen()

    def ids(self):
        """Iterator over cell ids."""
        return self.__iter__()

    def locate(self, id):
        """Given an element id in a Population, return the coordinates.
               e.g. for  4 6  , element 2 has coordinates (1,0) and value 7
                         7 9
        """
        # The top two lines (commented out) are the original implementation,
        # which does not scale well when the population size gets large.
        # The next lines are the neuron implementation of the same method. This
        # assumes that the id values in self.cell are consecutive. This should
        # always be the case, I think? A unit test is needed to check this.

        ###assert isinstance(id,int)
        ###return tuple([a.tolist()[0] for a in numpy.where(self.cell == id)])

        id -= self.id_start
        if self.ndim == 3:
            rows = self.dim[1]; cols = self.dim[2]
            i = id/(rows*cols); remainder = id%(rows*cols)
            j = remainder/cols; k = remainder%cols
            coords = (i,j,k)
        elif self.ndim == 2:
            cols = self.dim[1]
            i = id/cols; j = id%cols
            coords = (i,j)
        elif self.ndim == 1:
            coords = (id,)
        else:
            raise common.InvalidDimensionsError
        return coords

    def index(self, n):
        """Return the nth cell in the population (Indexing starts at 0)."""
        if hasattr(n, '__len__'):
            n = numpy.array(n)
        return self.cell.flatten()[n]

    def get(self, parameter_name, as_array=False):
        """
        Get the values of a parameter for every cell in the population.
        """
        values = [getattr(cell, parameter_name) for cell in self.cell_local]
        if as_array:
            values = numpy.array(values)
        return values

    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population. param
        can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param, str):
            if isinstance(val, (str, float, int)):
                param_dict = {param: float(val)}
            else:
                raise common.InvalidParameterValueError
        elif isinstance(param,dict):
            param_dict = param
        else:
            raise common.InvalidParameterValueError
        # This is not very efficient for simple and scaled parameters.
        # Should call nest.SetStatus(self.cell_local,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.
        for cell in self.cell_local:
            cell.set_parameters(**param_dict)

    def tset(self, parametername, value_array):
        """
        'Topographic' set. Set the value of parametername to the values in
        value_array, which must have the same dimensions as the Population.
        """
        # Convert everything to 1D arrays
        cells = numpy.reshape(self.cell, self.cell.size)
        if self.cell.shape == value_array.shape: # the values are numbers or non-array objects
            values = numpy.reshape(value_array, self.cell.size)
        elif len(value_array.shape) == len(self.cell.shape)+1: # the values are themselves 1D arrays
            values = numpy.reshape(value_array, (self.cell.size, value_array.size/self.cell.size))
        else:
            raise common.InvalidDimensionsError, "Population: %s, value_array: %s" % (str(cells.shape),
                                                                                      str(value_array.shape))
        # Set the values for each cell
        if len(cells) == len(values):
            for cell,val in zip(cells, values):
                if not isinstance(val, str) and hasattr(val, "__len__"):
                    # tuples, arrays are all converted to lists, since this is
                    # what SpikeSourceArray expects. This is not very robust
                    # though - we might want to add things that do accept arrays.
                    val = list(val)
                if cell in self.cell_local:
                    setattr(cell, parametername, val)
        else:
            raise common.InvalidDimensionsError

    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(rand_distr.rng, NativeRNG):
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            #rarr = rand_distr.next(n=len(self.cell_local))
            rarr = rand_distr.next(n=self.size)
            print rank(), self.cell_local[:5], self.cell_local[-5:], len(rarr), len(self.cell_local)
            assert len(rarr) >= len(self.cell_local), "The length of rarr (%d) must be greater than that of cell_local (%d)" % (len(rarr), len(self.cell_local))
            rarr = rarr[:len(self.cell_local)]
            for cell,val in zip(self.cell_local, rarr):
                setattr(cell, parametername, val)

    def _record(self, variable, record_from=None, rng=None):
        assert variable in ('spikes', 'v', 'conductance')

        # create device
        device_name = recording_device_names[variable]
        if self.recorders[variable] is None:
            self.recorders[variable] = nest.Create(device_name)

            ss_dict = {"to_file" : True, "withgid" : True, "withtime" : True}
    
            # check for older nest2 with need for interval
            if len(nest.GetStatus(self.recorders[variable], 'interval')) == 1:
                print "PyNN Warning: nest2 recording_device.interval detected."
                print "Please upgrade to a more recent version of nest 2"
                print "Transition code only temporarily supported."
                ss_dict['interval'] = nest.GetStatus([0],"resolution")[0]

            nest.SetStatus(self.recorders[variable],ss_dict)

        # create list of neurons
        fixed_list = False
        if record_from:
            if type(record_from) == types.ListType:
                fixed_list = True
                n_rec = len(record_from)
            elif type(record_from) == types.IntType:
                n_rec = record_from
            else:
                raise Exception("record_from must be a list or an integer")
        else:
            n_rec = self.size

        if variable == 'spikes':
            self.n_rec = n_rec

        tmp_list = []
        if (fixed_list == True):
            #for neuron in record_from:
            #    tmp_list = [neuron for neuron in record_from]
            tmp_list = record_from
        else:
            rng = rng or numpy.random
            for neuron in rng.permutation(numpy.reshape(self.cell, (self.cell.size,)))[0:n_rec]:
                tmp_list.append(neuron)

        tmp_set = Set(tmp_list)
        tmp_list = list( tmp_set.difference(self.recorded[variable]) )
        self.recorded[variable] = self.recorded[variable].union(tmp_set)
        # connect device to neurons
        _connect_recording_device(self.recorders[variable], record_from=tmp_list)

    def record(self, record_from=None, rng=None):
        """
        If record_from is not given, record spikes from all cells in the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids
        of the cells to record.
        """
        self._record('spikes', record_from, rng)

    def record_v(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('v', record_from, rng)

    def record_c(self, record_from=None, rng=None):
        """
        If record_from is not given, record the membrane potential for all cells in
        the Population.
        record_from can be an integer - the number of cells to record from, chosen
        at random (in this case a random number generator can also be supplied)
        - or a list containing the ids of the cells to record.
        """
        self._record('conductance', record_from, rng)

    def printSpikes(self, filename, gather=True, compatible_output=True):
        """
        Write spike times to file.

        If compatible_output is True, the format is "spiketime cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        This allows easy plotting of a `raster' plot of spiketimes, with one
        line for each cell.
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.

        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        spike files.

        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="spikes")

    def getSpikes(self, gather=True):
        """
        Return a 2-column numpy array containing cell ids and spike times for
        recorded cells.

        Useful for small populations, for example for single neuron Monte-Carlo.

        NOTE: getSpikes or printSpikes should be called only once per run,
        because they mangle simulator recorder files.
        """
        return _get_recorded_data(population=self, variable="spikes")

    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        # gather is not relevant, but is needed for API consistency
        n_spikes = nest.GetStatus(self.recorders['spikes'], "events")[0]
        return float(n_spikes)/self.n_rec

    def randomInit(self, rand_distr):
        """
        Set initial membrane potentials for all the cells in the population to
        random values.
        """
        self.rset('v_init', rand_distr)

    def print_v(self, filename, gather=True, compatible_output=True):
        """
        Write membrane potential traces to file.

        If compatible_output is True, the format is "v cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.

        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.

        For parallel simulators, if gather is True, all data will be gathered
        to the master node and a single output file created there. Otherwise, a
        file will be written on each node, containing only the cells simulated
        on that node.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="v")

    def print_c(self, filename, gather=True, compatible_output=True):
        """
        Write conductance traces to file.
        If compatible_output is True, the format is "t g cell_id",
        where cell_id is the index of the cell counting along rows and down
        columns (and the extension of that for 3-D).
        The timestep, first id, last id, and number of data points per cell are
        written in a header, indicated by a '#' at the beginning of the line.

        If compatible_output is False, the raw format produced by the simulator
        is used. This may be faster, since it avoids any post-processing of the
        voltage files.
        """
        _print(filename, gather=gather, compatible_output=compatible_output,
               population=self, variable="conductance")

    def describe(self):
        """
        Return a human readable description of the population"
        """
        print "\n------- Population description -------"
        print "Population called %s is made of %d cells [%d being local]" %(self.label, len(self.cell), len(self.cell_local))
        print "-> Cells are aranged on a %dD grid of size %s" %(len(self.dim), self.dim)
        print "-> Celltype is %s" %self.celltype
        print "-> Cell Parameters used for cell[0] (during initialization and now) are: "
        for key, value in self.cellparams.items():
          print "\t|", key, "\t: ", "init->", value, "\t now->", nest.GetStatus([self.cell[0]])[0][key]
        print "--- End of Population description ----"


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    class ConnectionDict:
        """docstring needed."""

        def __init__(self, parent):
            self.parent = parent

        def __getitem__(self, id):
            """Returns a (source address, target port number) tuple."""
            assert isinstance(id, int)
            return (self.parent._sources[id], self.parent._target_ports[id])

    def __init__(self, presynaptic_population, postsynaptic_population,
                 method='allToAll', method_parameters=None, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentials

        target - string specifying which synapse on the postsynaptic cell to
                 connect to

        If source and/or target are not given, default values are used.

        method - string indicating which algorithm to use in determining
                 connections.
        Allowed methods are 'allToAll', 'oneToOne', 'fixedProbability',
        'distanceDependentProbability', 'fixedNumberPre', 'fixedNumberPost',
        'fromFile', 'fromList'.

        method_parameters - dict containing parameters needed by the connection
        method, although we should allow this to be a number or string if there
        is only one parameter.

        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.

        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, method_parameters, source, target,
                                   synapse_dynamics, label, rng)

        self._target_ports = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self.synapse_type = target
        self._method = method

        if synapse_dynamics and synapse_dynamics.fast and synapse_dynamics.slow:
                raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
        self._plasticity_model = self.short_term_plasticity_mechanism or \
                                 self.long_term_plasticity_mechanism or \
                                 "static_synapse"
        assert self._plasticity_model in NEST_SYNAPSE_TYPES, self._plasticity_model

        # Set synaptic plasticity parameters
        original_synapse_context = nest.GetSynapseContext()
        nest.SetSynapseContext(self._plasticity_model)

        if hasattr(self, '_short_term_plasticity_parameters') and self._short_term_plasticity_parameters:
            synapse_defaults = nest.GetSynapseDefaults(self._plasticity_model)
            synapse_defaults.pop('num_connections') # otherwise NEST tells you to check your spelling!
            synapse_defaults.update(self._short_term_plasticity_parameters)
            nest.SetSynapseDefaults(self._plasticity_model, synapse_defaults)

        if hasattr(self, '_stdp_parameters') and self._stdp_parameters:
            # NEST does not support w_min != 0
            self._stdp_parameters.pop("w_min_always_zero_in_NEST")
            # Tau_minus is a parameter of the post-synaptic cell, not of the connection
            tau_minus = self._stdp_parameters.pop("tau_minus")
            # The following is a temporary workaround until the NEST guys stop renaming parameters!
            if 'tau_minus' in nest.GetStatus([self.post.cell_local[0]])[0]:
                nest.SetStatus(self.post.cell_local, [{'tau_minus': tau_minus}])
            elif 'Tau_minus' in nest.GetStatus([self.post.cell_local[0]])[0]:
                nest.SetStatus(self.post.cell_local, [{'Tau_minus': tau_minus}])
            else:
                raise Exception("Postsynaptic cell model does not support STDP.")

            synapse_defaults = nest.GetSynapseDefaults(self._plasticity_model)
            synapse_defaults.pop('num_connections') # otherwise NEST tells you to check your spelling!
            synapse_defaults.update(self._stdp_parameters)
            nest.SetSynapseDefaults(self._plasticity_model, synapse_defaults)

        # Create connections
        if isinstance(method, str):
            connection_method = getattr(self, '_%s' % method)
            self.nconn = connection_method(method_parameters)
        elif isinstance(method, common.Connector):
            self.nconn = method.connect(self)

        # Reset synapse context.
        # This is needed because low-level API does not support synapse dynamics
        # for now. We don't just reset to 'static_synapse' in case the user has
        # made a direct call to nest.SetSynapseContext()
        nest.SetSynapseContext(original_synapse_context)

        # Define a method to access individual connections
        self.connection = Projection.ConnectionDict(self)

    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)

    def connections(self):
        """for conn in prj.connections()..."""
        for i in xrange(len(self)):
            yield self.connection[i]

    # --- Connection methods ---------------------------------------------------

    def _allToAll(self, parameters=None):
        """
        Connect all cells in the presynaptic population to all cells in the postsynaptic population.
        """
        allow_self_connections = True # when pre- and post- are the same population,
                                      # is a cell allowed to connect to itself?
        if parameters and parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = AllToAllConnector(allow_self_connections)
        return c.connect(self)

    def _oneToOne(self, parameters=None):
        """
        Where the pre- and postsynaptic populations have the same size, connect
        cell i in the presynaptic population to cell i in the postsynaptic
        population for all i.
        In fact, despite the name, this should probably be generalised to the
        case where the pre and post populations have different dimensions, e.g.,
        cell i in a 1D pre population of size n should connect to all cells
        in row i of a 2D post population of size (n,m).
        """
        c = OneToOneConnector()
        return c.connect(self)

    def _fixedProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability is constant.
        """
        allow_self_connections = True
        try:
            p_connect = float(parameters)
        except TypeError:
            p_connect = parameters['p_connect']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        c = FixedProbabilityConnector(p_connect, allow_self_connections)
        return c.connect(self)

    def _distanceDependentProbability(self, parameters):
        """
        For each pair of pre-post cells, the connection probability depends on distance.
        d_expression should be the right-hand side of a valid python expression
        for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
        """
        allow_self_connections = True
        if type(parameters) == types.StringType:
            d_expression = parameters
        else:
            d_expression = parameters['d_expression']
            if parameters.has_key('allow_self_connections'):
                allow_self_connections = parameters['allow_self_connections']
        c = DistanceDependentProbabilityConnector(d_expression,
                                                  allow_self_connections=allow_self_connections)
        return c.connect(self)

    def _fixedNumberPre(self, parameters):
        """Each presynaptic cell makes a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPreConnector(n, allow_self_connections)
        return c.connect(self)

    def _fixedNumberPost(self, parameters):
        """Each postsynaptic cell receives a fixed number of connections."""
        n = parameters['n']
        if parameters.has_key('allow_self_connections'):
            allow_self_connections = parameters['allow_self_connections']
        c = FixedNumberPostConnector(n, allow_self_connections)
        return c.connect(self)

    def _fromFile(self, parameters):
        """
        Load connections from a file.
        """
        if type(parameters) == types.FileType:
            fileobj = parameters
            # should check here that fileobj is already open for reading
            lines = fileobj.readlines()
        elif type(parameters) == types.StringType:
            filename = parameters
            # now open the file...
            f = open(filename, 'r', DEFAULT_BUFFER_SIZE)
            lines = f.readlines()
        elif type(parameters) == types.DictType:
            # dict could have 'filename' key or 'file' key
            # implement this...
            raise Exception("Argument type not yet implemented")

        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[", 1)[1]
            tgt = "[%s" % tgt.split("[", 1)[1]
            src = eval(src)
            tgt = eval(tgt)
            input_tuples.append((src, tgt, float(w), float(d)))
        f.close()

        self._fromList(input_tuples)

    def _fromList(self, conn_list):
        """
        Read connections from a list of tuples,
        containing [pre_addr, post_addr, weight, delay]
        where pre_addr and post_addr are both neuron addresses, i.e. tuples or
        lists containing the neuron array coordinates.
        """
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = eval("self.pre%s" %src)
            tgt = eval("self.post%s" %tgt)
            pre_addr = nest.GetAddress([src])
            post_addr = nest.GetAddress([tgt])
            nest.ConnectWD(pre_addr, post_addr, [1000*weight], [delay])
            self._sources.append(src)
            self._targets.append(tgt)
            self._target_ports.append(tgt)


    # --- Methods for setting connection parameters ----------------------------

    def _set_connection_values(self, name, value, optimized=False):
        if is_number(value):
            if optimized:
                for src in self.pre.cell.flat:
                    _set_connections(src, self._plasticity_model, **{name: value})
            else:
                for src, port in self.connections():
                    _set_connection(src, port, self._plasticity_model, **{name: value})
        elif isinstance(value, (list, numpy.ndarray)):
            # this is probably not the most efficient way - should sort by src and then use SetConnections?
            assert len(value) == len(self)
            for (src,port),v in zip(self.connections(), value):
                try:
                    _set_connection(src, port, self._plasticity_model, **{name: v})
                except common.RoundingWarning:
                    logging.warning("Rounding occurred when setting %s to %s" % (name, v))
        else:
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")

    def setWeights(self, w):
        """
        w can be a single number, in which case all weights are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        Weights should be in nA for current-based and µS for conductance-based
        synapses.
        """
        w = _convertWeight(w, self.synapse_type)
        self._set_connection_values('weight', w, optimized=True)

    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        self.setWeights(rand_distr.next(len(self)))

    def setDelays(self, d):
        """
        d can be a single number, in which case all delays are set to this
        value, or a list/1D array of length equal to the number of connections
        in the population.
        """
        self._set_connection_values('delay', d, optimized=True)

    def randomizeDelays(self, rand_distr):
        """
        Set delays to random values taken from rand_distr.
        """
        self.setDelays(rand_distr.next(len(self)))

    def setSynapseDynamics(self, param, value):
        """
        Set parameters of the synapse dynamics linked with the projection
        """
        self._set_connection_values(param, value)

    def randomizeSynapseDynamics(self, param, rand_distr):
        """
        Set parameters of the synapse dynamics to values taken from rand_distr
        """
        self.setSynapseDynamics(param,rand_distr.next(len(self)))


    # --- Methods for writing/reading information to/from file. ----------------

    def _dump_connections(self):
        """For debugging."""
        print "Connections for Projection %s, connected with %s" % (self.label or '(un-labelled)',
                                                                    self._method)
        print "\tsource\ttarget\tport"
        for conn in zip(self._sources, self._targets, self._target_ports):
            print "\t%d\t%d\t%d" % conn
        print "Connection data for the presynaptic population (%s)" % self.pre.label
        for src in self.pre.cell.flat:
            print src, nest.GetConnections([src], self._plasticity_model)

    def _get_connection_values(self, format, parameter_name, gather):
        assert format in ('list', 'array'), "`format` is '%s', should be one of 'list', 'array'" % format
        if format == 'list':
            values = []
            for src, port in self.connections():
                values.append(_get_connection(src, port, self._plasticity_model, parameter_name))
        elif format == 'array':
            values = numpy.zeros((self.pre.size, self.post.size))
            for src, port in self.connections():
                v, tgt = _get_connection(src, port, self._plasticity_model,
                                         parameter_name, 'target')
                # note that we assume that Population ids are consecutive, which is the case, but we should
                # perhaps make an assert in __init__() to really make sure
                values[src-self.pre.id_start, tgt-self.post.id_start] = v
        return values

    def getWeights(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D weight array (with zero or None for non-existent
        connections).
        """
        weights = self._get_connection_values(format, 'weight', gather)
        # change of units
        if format == 'list':
            weights = [0.001*w for w in weights]
        elif format == 'array':
            weights *= 0.001
        return weights

    def getDelays(self, format='list', gather=True):
        """
        Possible formats are: a list of length equal to the number of connections
        in the projection, a 2D delay array (with None or 1e12 for non-existent
        connections).
        """
        return self._get_connection_values(format, 'delay', gather)

    def saveConnections(self, filename, gather=False):
        """Save connections to file in a format suitable for reading in with the
        'fromFile' method."""
        if gather == True:
            raise Exception("saveConnections(gather=True) not yet supported")
        elif num_processes() > 1:
            filename += '.%d' % rank()
        f = open(filename, 'w', DEFAULT_BUFFER_SIZE)
        weights = []; delays = []
        for src, port in self.connections():
            weight, delay = _get_connection(src, port, self._plasticity_model,
                                            'weight', 'delay')
            # Note unit change from pA to nA or nS to uS, depending on synapse type
            weights.append(0.001*weight)
            delays.append(delay)
        fmt = "%s%s\t%s%s\t%s\t%s\n" % (self.pre.label, "%s", self.post.label,
                                        "%s", "%g", "%g")
        for i in xrange(len(self)):
            line = fmt  % (self.pre.locate(self._sources[i]),
                           self.post.locate(self._targets[i]),
                           weights[i],
                           delays[i])
            line = line.replace('(','[').replace(')',']')
            f.write(line)
        f.close()

    def printWeights(self, filename, format='list', gather=True):
        """Print synaptic weights to file."""
        weights = self.getWeights(format=format, gather=gather)
        f = open(filename, 'w', DEFAULT_BUFFER_SIZE)
        if format == 'list':
            f.write("\n".join([str(w) for w in weights]))
        elif format == 'array':
            fmt = "%g "*len(self.post) + "\n"
            for row in weights:
                f.write(fmt % tuple(row))
        f.close()


    def weightHistogram(self, min=None, max=None, nbins=10):
        """
        Return a histogram of synaptic weights.
        If min and max are not given, the minimum and maximum weights are
        calculated automatically.
        """
        # it is arguable whether functions operating on the set of weights
        # should be put here or in an external module.
        bins = numpy.arange(min, max, (max-min)/nbins)
        return numpy.histogram(self.getWeights(format='list', gather=True), bins) # returns n, bins

    def describe(self):
        """
        Return a human readable description of the projection
        """
        print "\n------- Projection description -------"
        print "Projection %s from %s [%d cells] to %s [%d cells]" %(self.label, self.pre.label, len(self.pre.cell),self.post.label, len(self.post.cell))
        print "Connector used is %s : " %self._method
        if isinstance(self._method.weights,RandomDistribution):
          print "\t| Weights are drawn from %s distribution with parameters %s "%(self._method.weights.name, self._method.weights.parameters)
        else:
          print "\t| Weights: ", self._method.weights
        if isinstance(self._method.delays,RandomDistribution):
          print "\t| Delays are drawn from %s distribution with parameters %s " %(self._method.delays.name, self._method.delays.parameters)
        else:
          print "\t| Delays: ", self._method.delays
        print "\t| Plasticity: ", self._plasticity_model
        print "\t --> %d connections have been created for this projection" %len(self)
        print "To check, here are the parameters of one connection from this projection"
        print "\tsource\ttarget\tport"
        print "\t%d\t%d\t%d" %(self._sources[0], self._targets[0], self._target_ports[0])
        dict = nest.GetConnections([self.pre.cell.flat[0]], self._plasticity_model)[0]
        idx = numpy.where(numpy.array(dict['targets']) == self._targets[0])[0]
        for key, value in dict.items():
          print "\t| ", key, ": ", value[idx]

        print "---- End of Projection description -----"

# ==============================================================================
#   Utility classes
# ==============================================================================

Timer = common.Timer

# ==============================================================================
