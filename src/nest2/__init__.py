# -*- coding: utf-8 -*-
"""
PyNEST v2 implementation of the PyNN API.
$Id$
"""

import nest
from pyNN import common, utility
from pyNN.random import *
from pyNN import recording
import numpy, types, sys, shutil, os, logging, copy, tempfile, re
from math import *
from pyNN.nest2.cells import *
from pyNN.nest2.connectors import *
from pyNN.nest2.synapses import *
from pyNN.nest2.electrodes import *
from pyNN.nest2 import simulator
Set = set

tempdirs       = []

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
    except nest.NESTError, e:
        err_msg = str(e) + "\n  Problem getting connection from %s to port %s with synapse type '%s'." % (source_id, target_id, synapse_type)
        err_msg += "\n  Valid connections for source %s: %s" % (source_id, nest.GetConnections([source_id], synapse_type)[0])
        raise common.ConnectionError(err_msg)
    assert isinstance(conn_dict, dict)
    if len(parameter_names) == 1:
        return conn_dict[parameter_names[0]]
    else:
        return [conn_dict[p] for p in parameter_names]

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
    global tempdir
    log_file = "nest2.log"
    if debug:
        if isinstance(debug, basestring):
            log_file = debug
    utility.init_logging(log_file, debug, num_processes(), rank())
    logging.info("Initialization of Nest")
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    assert min_delay >= timestep, "min_delay (%g) must be greater than timestep (%g)" % (min_delay, timestep)

    if 'verbosity' in extra_params:
        nest_verbosity = extra_params['verbosity'].upper()
    else:
        nest_verbosity = "WARNING"
    nest.sli_run("M_%s setverbosity" % nest_verbosity)
        

    # reset the simulation kernel
    nest.ResetKernel()
    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')
    Projection.nProj = 0
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list

    # set tempdir
    try:
        nest.SetKernelStatus({'data_path': tempdir,})
    except nest.NESTError:    
        nest.SetStatus([0], {'device_prefix': tempdir,})


    # set kernel RNG seeds
    num_threads = extra_params.get('threads') or 1
    if 'rng_seeds' in extra_params:
        rng_seeds = extra_params['rng_seeds']
    else:
        rng_seeds_seed = extra_params.get('rng_seeds_seed') or 42
        rng = NumpyRNG(rng_seeds_seed)
        rng_seeds = (rng.rng.uniform(size=num_threads*num_processes())*100000).astype('int').tolist() 
    logging.debug("rng_seeds = %s" % rng_seeds)
    nest.SetStatus([0],[{'local_num_threads': num_threads,
                         'rng_seeds'        : rng_seeds}])

    # Set min_delay and max_delay for all synapse models
    for synapse_model in NEST_SYNAPSE_TYPES:
        # this is done in two steps, because otherwise NEST sometimes complains
        #   "max_delay is not compatible with default delay"
        nest.SetDefaults(synapse_model, {'delay' : min_delay, 'min_delay': min_delay, 'max_delay': max_delay})

    # set resolution
    nest.SetStatus([0], {'resolution': timestep})
    return nest.Rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    # We close the high level files opened by populations objects
    # that may have not been written.

    # NEST will soon close all its output files after the simulate function is over, therefore this step is not necessary
    global tempdirs

    # And we postprocess the low level files opened by record()
    # and record_v() method
    for recorder in simulator.recorder_list:
        recorder.write(gather=False, compatible_output=compatible_output)

    for tempdir in tempdirs:
        os.system("rm -rf %s" %tempdir)
        tempdirs.remove(tempdir)

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
    return nest.GetDefaults('static_synapse')['min_delay']
common.get_min_delay = get_min_delay

def get_max_delay():
    return nest.GetDefaults('static_synapse')['max_delay']
common.get_max_delay = get_max_delay

def num_processes():
    return nest.GetStatus([0])[0]['num_processes']
common.num_processes = num_processes

def rank():
    """Return the MPI rank."""
    return nest.Rank()
common.rank = rank

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

def _create(cellclass, cellparams=None, n=1, parent=None):
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

create = common.build_create(_create)

connect = common.build_connect(simulator)

set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

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
        if isinstance(cellclass, type) and issubclass(cellclass, common.StandardCellType):
            self.celltype = cellclass(cellparams)
        
        self.all_cells, self._mask_local, self.first_id, self.last_id = _create(cellclass, cellparams, self.size, parent=self)
        self.local_cells = self.all_cells[self._mask_local]
        self.all_cells = self.all_cells.reshape(self.dim)
        self._mask_local = self._mask_local.reshape(self.dim)
        
        for id in self.local_cells:
            id.parent = self
        self.cell = self.all_cells # temporary alias, awaiting harmonization
        
        if not self.label:
            self.label = 'population%d' % Population.nPop
        self.recorders = {}
        for variable in simulator.RECORDING_DEVICE_NAMES:
            self.recorders[variable] = simulator.Recorder(variable, population=self)
        Population.nPop += 1

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
        
        # The default implementation in common is is not very efficient for
        # simple and scaled parameters.
        # Should call nest.SetStatus(self.local_cells,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.
        for key, value in param_dict.items():
            if not isinstance(self.celltype, str):
                # Here we check the consistency of the given parameters
                try:
                    self.celltype.default_parameters[key]
                except Exception:
                    raise common.NonExistentParameterError(key, self.celltype.__class__)
                if type(value) != type(self.celltype.default_parameters[key]):
                    raise common.InvalidParameterValueError
                
                # Then we do the call to SetStatus
                if key in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[key]
                    value = eval(translation['forward_transform'], globals(), {key:value})
                    nest.SetStatus(self.local_cells,translation['translated_name'],value)
                elif key in self.celltype.simple_parameters():
                    translation = self.celltype.translations[key]
                    nest.SetStatus(self.local_cells, translation['translated_name'], value)
                else:
                    for cell in self.local_cells:
                        cell.set_parameters(**{key:value})
            else:
                try:
                    nest.SetStatus(self.local_cells, key, value)
                except Exception:
                    raise common.InvalidParameterValueError

    def rset(self, parametername, rand_distr):
        """
        'Random' set. Set the value of parametername to a value taken from
        rand_distr, which should be a RandomDistribution object.
        """
        if isinstance(rand_distr.rng, NativeRNG):
            raise Exception('rset() not yet implemented for NativeRNG')
        else:
            #rarr = rand_distr.next(n=len(self.local_cells))
            rarr = rand_distr.next(n=self.size)
            assert len(rarr) >= len(self.local_cells), "The length of rarr (%d) must be greater than that of local_cells (%d)" % (len(rarr), len(self.local_cells))
            rarr = rarr[:len(self.local_cells)]
            if not isinstance(self.celltype, str):
                try:
                    self.celltype.default_parameters[parametername]
                except Exception:
                    raise common.NonExistentParameterError(parametername, self.celltype.__class__)
                if parametername in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[parametername]
                    rarr = eval(translation['forward_transform'], globals(), {parametername : rarr})
                    nest.SetStatus(self.local_cells,translation['translated_name'],rarr)
                elif parametername in self.celltype.simple_parameters():
                    translation = self.celltype.translations[parametername]
                    nest.SetStatus(self.local_cells, translation['translated_name'], rarr)
                else:
                    for cell,val in zip(self.local_cells, rarr):
                        setattr(cell, parametername, val)
            else:
               nest.SetStatus(self.local_cells, parametername, rarr)

    def _record(self, variable, record_from=None, rng=None,to_file=True):
        if variable not in self.celltype.recordable:
            raise Exception("Cannot record %s from cell type %s" % (variable, self.celltype.__class__.__name__))
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
            if not rng:
                rng = numpy.random
            tmp_list = rng.permutation(numpy.reshape(self.cell, (self.cell.size,)))[0:n_rec]
    
        self.recorders[variable].record(tmp_list)
        nest.SetStatus(self.recorders[variable]._device, {'to_file': to_file, 'to_memory' : not to_file})
    
    def meanSpikeCount(self, gather=True):
        """
        Returns the mean number of spikes per neuron.
        """
        ## Routine to give an average firing rate over all the threads/nodes
        ## This is a rough approximation, because in fact each nodes is only multiplying 
        ## the frequency of the recorders by the number of processes. To do better, we need a MPI
        ## package to send informations to node 0. Nevertheless, it works for threaded mode
        node_list = range(nest.GetStatus([0], "total_num_virtual_procs")[0])
        n_spikes  = 0
        for node in node_list:
            nest.sps(self.recorders['spikes']._device[0])
            nest.sr("%d GetAddress %d append" %(self.recorders['spikes']._device[0], node))
            #nest.sr("GetStatus /events get")
            nest.sr("GetStatus /n_events get")
            n_spikes += nest.spp()
        n_rec = len(self.recorders['spikes'].recorded)
        return float(n_spikes)/n_rec

    def getSubPopulation(self, cell_list, label=None):
        
        # We get the dimensions of the new population
        dims = numpy.array(cell_list).shape
        # We create an empty population
        pop = Population(dims, cellclass=self.celltype, label=label, parent=self)
        # And then copy parameters from its parent
        pop.cellparams  = pop.parent.cellparams
        pop.first_id    = pop.parent.first_id
        idx             = numpy.array(cell_list,int).flatten() - pop.first_id
        pop.cell        = pop.parent.cell.flatten()[idx].reshape(dims)
        pop.local_cells  = pop.parent.local_cells[idx]
        pop.positions   = pop.parent.positions[:,idx]
        return pop


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """

    nProj = 0

    def __init__(self, presynaptic_population, postsynaptic_population,
                 method, source=None,
                 target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentials

        target - string specifying which synapse on the postsynaptic cell to
                 connect to

        If source and/or target are not given, default values are used.

        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.

        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.

        rng - since most of the connection methods need uniform random numbers,
        it is probably more convenient to specify a RNG object here rather
        than within method_parameters, particularly since some methods also use
        random numbers to give variability in the number of connections per cell.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)

        self._target_ports = [] # holds port numbers
        self._targets = []     # holds gids
        self._sources = []     # holds gids
        self.synapse_type = target or 'excitatory'

        if isinstance(self.long_term_plasticity_mechanism, Set):
            logging.warning("Several STDP models are available for these connections:")
            logging.warning(", ".join(model for model in self.long_term_plasticity_mechanism))
            self.long_term_plasticity_mechanism = list(self.long_term_plasticity_mechanism)[0]
            logging.warning("By default, %s is used" % self.long_term_plasticity_mechanism)

        if synapse_dynamics and synapse_dynamics.fast and synapse_dynamics.slow:
                raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
        self._plasticity_model = self.short_term_plasticity_mechanism or \
                                 self.long_term_plasticity_mechanism or \
                                 "static_synapse"
        assert self._plasticity_model in NEST_SYNAPSE_TYPES, self._plasticity_model

        # Set synaptic plasticity parameters 
        # We create a particular synapse context just for this projection, by copying
        # the one which is desired. The name of the synapse context is randomly generated
        # and will be available as projection.plasticity_name
        self.plasticity_name = "projection_%d" % Projection.nProj
        Projection.nProj += 1
        synapse_defaults = nest.GetDefaults(self._plasticity_model)
        synapse_defaults.pop('synapsemodel')
        synapse_defaults.pop('num_connections')
        if 'num_connectors' in synapse_defaults:
            synapse_defaults.pop('num_connectors')
            
        if hasattr(self, '_short_term_plasticity_parameters') and self._short_term_plasticity_parameters:
            synapse_defaults.update(self._short_term_plasticity_parameters)

        if hasattr(self, '_stdp_parameters') and self._stdp_parameters:
            # NEST does not support w_min != 0
            self._stdp_parameters.pop("w_min_always_zero_in_NEST")
            # Tau_minus is a parameter of the post-synaptic cell, not of the connection
            tau_minus = self._stdp_parameters.pop("tau_minus")
            # The following is a temporary workaround until the NEST guys stop renaming parameters!
            if 'tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                nest.SetStatus(self.post.local_cells, [{'tau_minus': tau_minus}])
            elif 'Tau_minus' in nest.GetStatus([self.post.local_cells[0]])[0]:
                nest.SetStatus(self.post.local_cells, [{'Tau_minus': tau_minus}])
            else:
                raise Exception("Postsynaptic cell model does not support STDP.")

            synapse_defaults.update(self._stdp_parameters)

        nest.CopyModel(self._plasticity_model, self.plasticity_name, synapse_defaults)
        
        # Create connections
        self.nconn = method.connect(self)

        # Define a method to access individual connections
        self.connection = simulator.ConnectionDict(self)

    def __len__(self):
        """Return the total number of connections."""
        return len(self._sources)

    def connections(self):
        """for conn in prj.connections()..."""
        self.connection.reset()
        for i in xrange(len(self)):
            yield self.connection[i]

    # --- Methods for setting connection parameters ----------------------------

    def _set_connection_values(self, name, value, optimized=False):
        if is_number(value):
            if optimized:
                for src in self.pre.cell.flat:
                    _set_connections(src, self.plasticity_name, **{name: value})
            else:
                for src, port in self.connections():
                    _set_connection(src, port, self.plasticity_name, **{name: value})
        elif isinstance(value, (list, numpy.ndarray)):
            # this is probably not the most efficient way - should sort by src and then use SetConnections?
            assert len(value) == len(self)
            for (src,port),v in zip(self.connections(), value):
                try:
                    _set_connection(src, port, self.plasticity_name, **{name: v})
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
        for src,tgt in zip(self._sources, self._targets):
            connections = nest.FindConnections([src],[tgt],self.plasticity_name)
            for port in connections['ports']:
                print "\t%d\t%d\t%d" % (src, tgt, port)
        print "Connection data for the presynaptic population (%s)" % self.pre.label
        for src in self.pre.cell.flat:
            print src, nest.GetConnections([src], self.plasticity_name)

    def _get_connection_values(self, format, parameter_name, gather):
        assert format in ('list', 'array'), "`format` is '%s', should be one of 'list', 'array'" % format
        if format == 'list':
            values = []
            for src, port in self.connections():
                values.append(_get_connection(src, port, self.plasticity_name, parameter_name))
        elif format == 'array':
            values = numpy.zeros((self.pre.size, self.post.size))
            for src, port in self.connections():
                v, tgt = _get_connection(src, port, self.plasticity_name,
                                         parameter_name, 'target')
                # note that we assume that Population ids are consecutive, which is the case, but we should
                # perhaps make an assert in __init__() to really make sure
                values[src-self.pre.first_id, tgt-self.post.first_id] = v
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
            weight, delay = _get_connection(src, port, self.plasticity_name,
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

    def describe(self, template='standard'):
        """
        Returns a human readable description of the projection
        """
        description = common.Projection.describe(self, template)
        src = self._sources[0]
        tgt = self._targets[0]
        port = nest.FindConnections([src],[tgt],self.plasticity_name)['ports'][0]
        description += "\n    Parameters of connection from %d to %d [port %d]" % (src, tgt, port)
        dict = nest.GetConnections([self.pre.cell.flat[0]], self.plasticity_name)[0]
        for i in xrange(len(self._targets)):
            idx  = numpy.where(numpy.array(dict['targets']) == self._targets[i])[0]
            if len(idx) > 0: 
                for key, value in dict.items():
                    description += "\n    | %s: %s" % (key, value[idx[0]])
                break
        description += "\n---- End of NEST-specific Projection description -----"
        return description

# ==============================================================================



