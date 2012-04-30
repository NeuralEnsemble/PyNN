# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""
import nest
from pyNN.nest import simulator
from pyNN import common, recording, errors, space, __doc__

if recording.MPI and (nest.Rank() != recording.mpi_comm.rank):
    raise Exception("MPI not working properly. Please make sure you import pyNN.nest before pyNN.random.")

import numpy
import os
import shutil
import logging
import tempfile
from pyNN.nest.cells import NativeCellType, native_cell_type
from pyNN.nest.synapses import NativeSynapseDynamics, NativeSynapseMechanism
from pyNN.nest.standardmodels.cells import *
from pyNN.nest.connectors import *
from pyNN.nest.standardmodels.synapses import *
from pyNN.nest.standardmodels.electrodes import *
from pyNN.nest.recording import *
from pyNN.random import RandomDistribution
from pyNN import standardmodels

Set = set
tempdirs       = []
NEST_SYNAPSE_TYPES = nest.Models(mtype='synapses')

STATE_VARIABLE_MAP = {"v": "V_m", "w": "w", "gsyn_exc": "g_ex",
                      "gsyn_inh": "g_in"}
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(obj, type) and issubclass(obj, standardmodels.StandardCellType)]
    for cell_class in standard_cell_types:
        try:
            create(cell_class)
        except Exception, e:
            print "Warning: %s is defined, but produces the following error: %s" % (cell_class.__name__, e)
            standard_cell_types.remove(cell_class)
    return [obj.__name__ for obj in standard_cell_types]

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

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    global tempdir

    common.setup(timestep, min_delay, max_delay, **extra_params)

    if 'verbosity' in extra_params:
        nest_verbosity = extra_params['verbosity'].upper()
    else:
        nest_verbosity = "WARNING"
    nest.sli_run("M_%s setverbosity" % nest_verbosity)

    if "spike_precision" in extra_params:
        simulator.state.spike_precision = extra_params["spike_precision"]
        if extra_params["spike_precision"] == 'off_grid':
            simulator.state.default_recording_precision = 15
    nest.SetKernelStatus({'off_grid_spiking': simulator.state.spike_precision=='off_grid'})
    if "recording_precision" in extra_params:
        simulator.state.default_recording_precision = extra_params["recording_precision"]


    # clear the sli stack, if this is not done --> memory leak cause the stack increases
    nest.sr('clear')

    # reset the simulation kernel
    nest.ResetKernel()

    # all NEST to erase previously written files (defaut with all the other simulators)
    nest.SetKernelStatus({'overwrite_files' : True})

    # set tempdir
    tempdir = tempfile.mkdtemp()
    tempdirs.append(tempdir) # append tempdir to tempdirs list
    nest.SetKernelStatus({'data_path': tempdir,})

    # set kernel RNG seeds
    num_threads = extra_params.get('threads') or 1
    if 'rng_seeds' in extra_params:
        rng_seeds = extra_params['rng_seeds']
    else:
        rng_seeds_seed = extra_params.get('rng_seeds_seed') or 42
        rng = NumpyRNG(rng_seeds_seed)
        rng_seeds = (rng.rng.uniform(size=num_threads*num_processes())*100000).astype('int').tolist()
    logger.debug("rng_seeds = %s" % rng_seeds)
    nest.SetKernelStatus({'local_num_threads': num_threads,
                          'rng_seeds'        : rng_seeds})

    # set resolution
    nest.SetKernelStatus({'resolution': float(timestep)})

    # Set min_delay and max_delay for all synapse models
    for synapse_model in NEST_SYNAPSE_TYPES:
        nest.SetDefaults(synapse_model, {'delay'    : float(min_delay),
                                         'min_delay': float(min_delay),
                                         'max_delay': float(max_delay)})
    simulator.reset()

    return rank()

def end():
    """Do any necessary cleaning up before exiting."""
    global tempdirs
    for (population, variables, filename) in simulator.write_on_end:
        io = recording.get_io(filename)
        population.write_data(io, variables)
    for tempdir in tempdirs:
        shutil.rmtree(tempdir)
    tempdirs = []
    simulator.write_on_end = []

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.run(simtime)
    return get_current_time()

reset = common.build_reset(simulator)

initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
            num_processes, rank = common.build_state_queries(simulator)


# ==============================================================================
#   High-level API for creating, connecting and recording from populations of
#   neurons.
# ==============================================================================

class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _simulator = simulator
    assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    recorder_class = Recorder
    assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        super(Population, self).__init__(size, cellclass, cellparams, structure, initial_values, label)
        self._simulator.populations.append(self)

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create cells in NEST.

        `cellclass`  -- a PyNN standard cell or the name of a native NEST model.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        # this method should never be called more than once
        # perhaps should check for that
        assert n > 0, 'n must be a positive integer'
        n = int(n)

        celltype = cellclass(cellparams)
        nest_model = celltype.nest_name[simulator.state.spike_precision]
        try:
            self.all_cells = nest.Create(nest_model, n, params=celltype.parameters)
        except nest.NESTError, err:
            if "UnknownModelName" in err.message and "cond" in err.message:
                raise errors.InvalidModelError("%s Have you compiled NEST with the GSL (Gnu Scientific Library)?" % err)
            raise errors.InvalidModelError(err)
        # create parrot neurons if necessary
        if hasattr(celltype, "uses_parrot") and celltype.uses_parrot:
            self.all_cells_source = numpy.array(self.all_cells)  # we put the parrots into all_cells, since this will
            self.all_cells = nest.Create("parrot_neuron", n)     # be used for connections and recording. all_cells_source
            nest.Connect(self.all_cells_source, self.all_cells)  # should be used for setting parameters
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self
        if hasattr(celltype, "uses_parrot") and celltype.uses_parrot:
            for gid, source in zip(self.all_cells, self.all_cells_source):
                gid.source = source


    def set(self, param, val=None):
        """
        Set one or more parameters for every cell in the population.

        param can be a dict, in which case val should not be supplied, or a string
        giving the parameter name, in which case val is the parameter value.
        val can be a numeric value, or list of such (e.g. for setting spike times).
        e.g. p.set("tau_m",20.0).
             p.set({'tau_m':20,'v_rest':-65})
        """
        if isinstance(param, str):
            if isinstance(val, (str, float, int)):
                param_dict = {param: float(val)}
            elif isinstance(val, (list, numpy.ndarray)):
                param_dict = {param: [val]*len(self)}
            else:
                raise errors.InvalidParameterValueError
        elif isinstance(param, dict):
            param_dict = param
        else:
            raise errors.InvalidParameterValueError
        param_dict = self.celltype.check_parameters(param_dict, with_defaults=False)
        # The default implementation in common is is not very efficient for
        # simple and scaled parameters.
        # Should call nest.SetStatus(self.local_cells,...) for the parameters in
        # self.celltype.__class__.simple_parameters() and .scaled_parameters()
        # and keep the loop below just for the computed parameters. Even in this
        # case, it may be quicker to test whether the parameters participating
        # in the computation vary between cells, since if this is not the case
        # we can do the computation here and use nest.SetStatus.

        if isinstance(self.celltype, standardmodels.StandardCellType):
            to_be_set = {}
            if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
                gids = self.all_cells_source[self._mask_local]
            else:
                gids = self.local_cells
            for key, value in param_dict.items():
                if key in self.celltype.scaled_parameters():
                    translation = self.celltype.translations[key]
                    value = eval(translation['forward_transform'], globals(), {key:value})
                    to_be_set[translation['translated_name']] = value
                elif key in self.celltype.simple_parameters():
                    translation = self.celltype.translations[key]
                    to_be_set[translation['translated_name']] = value
                else:
                    assert key in self.celltype.computed_parameters()
            logger.debug("Setting the following parameters: %s" % to_be_set)
            nest.SetStatus(gids.tolist(), to_be_set)
            for key, value in param_dict.items():
                if key in self.celltype.computed_parameters():
                    logger.debug("Setting %s = %s" % (key, value))
                    for cell in self:
                        cell.set_parameters(**{key:value})
        else:
            nest.SetStatus(self.local_cells.tolist(), param_dict)

    def _set_initial_value_array(self, variable, value):
        variable = STATE_VARIABLE_MAP.get(variable, variable)
        try:
            nest.SetStatus(self.local_cells.tolist(), variable, value)
        except nest.NESTError, e:
            logger.warning("NEST does not allow setting an initial value for %s" % variable) # assuming this is an "Unused dictionary items" error - should really check


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    _simulator = simulator
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

        rng - specify an RNG object to be used by the Connector.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   method, source, target,
                                   synapse_dynamics, label, rng)
        self.synapse_type = target or 'excitatory'
        if self.synapse_dynamics:
            synapse_dynamics = self.synapse_dynamics
            self.synapse_dynamics._set_tau_minus(self.post.local_cells)
        else:
            synapse_dynamics = NativeSynapseDynamics("static_synapse")
        synapse_model = synapse_dynamics._get_nest_synapse_model("projection_%d" % Projection.nProj)
        if synapse_model is None:
            self.synapse_model = 'static_synapse_%s' % id(self)
            nest.CopyModel('static_synapse', self.synapse_model)
        else:
            self.synapse_model = synapse_model
        self._sources = []
        self._connections = None
        Projection.nProj += 1

        # Create connections
        method.connect(self)

    def __getitem__(self, i):
        """Return the `i`th connection on the local MPI node."""
        if isinstance(i, int):
            if i < len(self):
                return simulator.Connection(self, i)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                return [simulator.Connection(self, j) for j in range(i.start, i.stop, i.step or 1)]
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __len__(self):
        """Return the number of connections on the local MPI node."""
        return nest.GetDefaults(self.synapse_model)['num_connections']

    @property
    def connections(self):
        if self._connections is None:
            self._sources = numpy.unique(self._sources)
            self._connections = nest.FindConnections(self._sources, synapse_type=self.synapse_model)
        return self._connections

    def _set_tsodyks_params(self):
        if 'tsodyks' in self.synapse_model: # there should be a better way to do this. In particular, if the synaptic time constant is changed
                                            # after creating the Projection, tau_psc ought to be changed as well.
            assert self.synapse_type in ('excitatory', 'inhibitory'), "only basic synapse types support Tsodyks-Markram connections"
            logger.debug("setting tau_psc")
            targets = nest.GetStatus(self.connections, 'target')
            if self.synapse_type == 'inhibitory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_I']['translated_name']
            if self.synapse_type == 'excitatory':
                param_name = self.post.local_cells[0].celltype.translations['tau_syn_E']['translated_name']
            tau_syn = nest.GetStatus(targets, (param_name))
            nest.SetStatus(self.connections, 'tau_psc', tau_syn)

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        # are we sure the targets are all on the current node?
        if core.is_listlike(source):
            assert len(source) == 1
            source = source[0]
        if not core.is_listlike(targets):
            targets = [targets]
        assert len(targets) > 0

        if self.synapse_type not in targets[0].celltype.synapse_types:
            raise errors.ConnectionError("User gave synapse_type=%s, synapse_type must be one of: %s" % ( self.synapse_type, "'"+"', '".join(st for st in targets[0].celltype.synapse_types or ['*No connections supported*']))+"'" )
        weights = numpy.array(weights)*1000.0 # weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and common.is_conductance(targets[0]):
            weights *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]

        if targets[0].celltype.standard_receptor_type:
            try:
                nest.DivergentConnect([source], targets, weights, delays, self.synapse_model)
            except nest.NESTError, e:
                raise errors.ConnectionError("%s. source=%s, targets=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                             e, source, targets, weights, delays, self.synapse_model))
        else:
            for target, w, d in zip(targets, weights, delays):
                nest.Connect([source], [target], {'weight': w, 'delay': d, 'receptor_type': target.celltype.get_receptor_type(self.synapse_type)})
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self._sources.append(source)


    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect one or more neurons to a single post-synaptic neuron.
        `sources` -- a list/1D array of pre-synaptic cell IDs, or a single ID.
        `target`  -- the ID of the post-synaptic cell.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        # are we sure the targets are all on the current node?
        if core.is_listlike(target):
            assert len(target) == 1
            target = target[0]
        if not core.is_listlike(sources):
            sources = [sources]
        assert len(sources) > 0, sources
        if self.synapse_type not in ('excitatory', 'inhibitory', None):
            raise errors.ConnectionError("synapse_type must be 'excitatory', 'inhibitory', or None (equivalent to 'excitatory')")
        weights = numpy.array(weights)*1000.0# weights should be in nA or uS, but iaf_neuron uses pA and iaf_cond_neuron uses nS.
                                 # Using convention in this way is not ideal. We should
                                 # be able to look up the units used by each model somewhere.
        if self.synapse_type == 'inhibitory' and common.is_conductance(target):
            weights = -1*weights # NEST wants negative values for inhibitory weights, even if these are conductances
        if isinstance(weights, numpy.ndarray):
            weights = weights.tolist()
        elif isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, numpy.ndarray):
            delays = delays.tolist()
        elif isinstance(delays, float):
            delays = [delays]

        try:
            nest.ConvergentConnect(sources, [target], weights, delays, self.synapse_model)
        except nest.NESTError, e:
            raise errors.ConnectionError("%s. sources=%s, target=%s, weights=%s, delays=%s, synapse model='%s'" % (
                                         e, sources, target, weights, delays, self.synapse_model))
        self._connections = None # reset the caching of the connection list, since this will have to be recalculated
        self._sources.extend(sources)

    def set(self, name, value):
        """
        Set connection attributes for all connections on the local MPI node.

        `name`  -- attribute name

        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        if not (numpy.isscalar(value) or core.is_listlike(value)):
            raise TypeError("Argument should be a numeric type (int, float...), a list, or a numpy array.")

        if isinstance(value, numpy.ndarray) and len(value.shape) == 2:
            value_list = []
            connection_parameters = nest.GetStatus(self.connections, ('source', 'target'))
            for conn in connection_parameters:
                addr = self.pre.id_to_index(conn['source']), self.post.id_to_index(conn['target'])
                try:
                    val = value[addr]
                except IndexError, e:
                    raise IndexError("%s. addr=%s" % (e, addr))
                if numpy.isnan(val):
                    raise Exception("Array contains no value for synapse from %d to %d" % (c.source, c.target))
                else:
                    value_list.append(val)
            value = value_list
        if core.is_listlike(value):
            value = numpy.array(value)
        else:
            value = float(value)

        if name == 'weight':
            value *= 1000.0
            if self.synapse_type == 'inhibitory' and common.is_conductance(self.post[0]):
                value *= -1 # NEST wants negative values for inhibitory weights, even if these are conductances
        elif name == 'delay':
            pass
        else:
            #translation = self.synapse_dynamics.reverse_translate({name: value})
            #name, value = translation.items()[0]
            translated_name = None
            if self.synapse_dynamics.fast:
                if name in self.synapse_dynamics.fast.translations:
                    translated_name = self.synapse_dynamics.fast.translations[name]["translated_name"] # a hack
            if translated_name is None:
                if self.synapse_dynamics.slow:
                    for component_name in "timing_dependence", "weight_dependence", "voltage_dependence":
                        component = getattr(self.synapse_dynamics.slow, component_name)
                        if component and name in component.translations:
                            translated_name = component.translations[name]["translated_name"]
                            break
            if translated_name:
                name = translated_name

        i = 0
        try:
            nest.SetStatus(self.connections, name, value)
        except nest.NESTError, e:
            n = 1
            if hasattr(value, '__len__'):
                n = len(value)
            raise Exception("%s. Trying to set %d values." % (e, n))

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        import operator

        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')

        lines   = nest.GetStatus(self.connections, ('source', 'target', 'weight', 'delay'))

        if gather == True and num_processes() > 1:
            all_lines = { rank(): lines }
            all_lines = recording.gather_dict(all_lines)
            if rank() == 0:
                lines = reduce(operator.add, all_lines.values())
        elif num_processes() > 1:
            file.rename('%s.%d' % (file.name, rank()))
        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)

        if gather == False or rank() == 0:
            lines       = numpy.array(lines, dtype=float)
            lines[:,2] *= 0.001
            if compatible_output:
                lines[:,0] = self.pre.id_to_index(lines[:,0])
                lines[:,1] = self.post.id_to_index(lines[:,1])
            file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
            file.close()

    def randomizeWeights(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        self.setWeights(rand_distr.next(len(self), mask_local=False))

    def randomizeDelays(self, rand_distr):
        """
        Set weights to random values taken from rand_distr.
        """
        # Arguably, we could merge this with set_weights just by detecting the
        # argument type. It could make for easier-to-read simulation code to
        # give it a separate name, though. Comments?
        self.setDelays(rand_distr.next(len(self), mask_local=False))

    def get(self, parameter_name, format, gather=True):
        """
        Get the values of a given attribute (weight or delay) for all
        connections in this Projection.

        `parameter_name` -- name of the attribute whose values are wanted.

        `format` -- "list" or "array". Array format implicitly assumes that all
                    connections belong to a single Projection.

        Return a list or a 2D Numpy array. The array element X_ij contains the
        attribute value for the connection from the ith neuron in the pre-
        synaptic Population to the jth neuron in the post-synaptic Population,
        if a single such connection exists. If there are no such connections,
        X_ij will be NaN. If there are multiple such connections, the summed
        value will be given, which makes some sense for weights, but is
        pretty meaningless for delays.
        """

        if parameter_name not in ('weight', 'delay'):
            translated_name = None
            if self.synapse_dynamics.fast and parameter_name in self.synapse_dynamics.fast.translations:
                translated_name = self.synapse_dynamics.fast.translations[parameter_name]["translated_name"] # this is a hack that works because there are no units conversions
            elif self.synapse_dynamics.slow:
                for component_name in "timing_dependence", "weight_dependence", "voltage_dependence":
                    component = getattr(self.synapse_dynamics.slow, component_name)
                    if component and parameter_name in component.translations:
                        translated_name = component.translations[parameter_name]["translated_name"]
                        break
            if translated_name:
                parameter_name = translated_name
            else:
                raise Exception("synapse type does not have an attribute '%s', or else this attribute is not accessible." % parameter_name)
        if format == 'list':
            values = nest.GetStatus(self.connections, parameter_name)
            if parameter_name == "weight":
                values = [0.001*val for val in values]
        elif format == 'array':
            value_arr = numpy.nan * numpy.ones((self.pre.size, self.post.size))
            connection_parameters = nest.GetStatus(self.connections, ('source', 'target', parameter_name))
            for conn in connection_parameters:
                # (offset is always 0,0 for connections created with connect())
                src, tgt, value = conn
                addr = self.pre.id_to_index(src), self.post.id_to_index(tgt)
                if numpy.isnan(value_arr[addr]):
                    value_arr[addr] = value
                else:
                    value_arr[addr] += value
            if parameter_name == 'weight':
                value_arr *= 0.001
                if self.synapse_type == 'inhibitory' and common.is_conductance(self[0].target):
                    value_arr *= -1 # NEST uses negative values for inhibitory weights, even if these are conductances
            values = value_arr
        else:
            raise Exception("format must be 'list' or 'array', actually '%s'" % format)
        return values

Space = space.Space

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

record_gsyn = common.build_record('gsyn', simulator)

# ==============================================================================
