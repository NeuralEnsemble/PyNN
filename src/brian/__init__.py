# -*- coding: utf-8 -*-
"""
Brian implementation of the PyNN API.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import logging
#import brian_no_units_no_warnings
from pyNN.brian import simulator
from pyNN import common, recording, space, core, __doc__
from pyNN.recording import files
from pyNN.brian.standardmodels.cells import *
from pyNN.brian.standardmodels.electrodes import *
from pyNN.brian.connectors import *
from pyNN.brian.standardmodels.synapses import *
from pyNN.brian import electrodes
from pyNN.brian.recording import *
from pyNN import standardmodels
from pyNN.parameters import ParameterSpace

logger = logging.getLogger("PyNN")

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

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    brian.set_global_preferences(**extra_params)
    simulator.state = simulator._State(timestep, min_delay, max_delay)
    simulator.state.add(update_currents) # from electrodes
    ## We need to reset the clock of the update_currents function, for the electrodes
    simulator.state.network._all_operations[0].clock = brian.Clock(t=0*ms, dt=timestep*ms)
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.dt        = timestep
    recording.simulator = simulator
    reset()
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    simulator.recorder_list = []
    electrodes.current_sources = []
    for item in simulator.state.network.groups + simulator.state.network._all_operations:
        del item
    del simulator.state

def run(simtime):
    """Run the simulation for simtime ms."""
    simulator.state.run(simtime)
    return get_current_time()

reset = simulator.reset

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
    _assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        Return a ParameterSpace containing native parameters
        """
        params = {}
        for name in names:
            params[name] = getattr(self.parent.brian_cells, name)[self.mask]
            assert isinstance(params[name], numpy.ndarray)
        return ParameterSpace(params, size=self.size)


class Population(common.Population):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _create_cells(self):
        """
        Create cells in Brian using the celltype of the current Population.
        """
        # currently, we create a single NeuronGroup for create(), but
        # arguably we should use n NeuronGroups each containing a single cell
        # either that or use the subgroup() method in connect(), etc
        cell_parameters = self.celltype.translated_parameters
        cell_parameters.size = self.size
        cell_parameters.evaluate(simplify=False)
        brian_cells = self.celltype.brian_model(self.size,
                                                self.celltype.eqs,
                                                **cell_parameters)

        # should we globally track the IDs used, so as to ensure each cell gets a unique integer? (need only track the max ID)
        self.all_cells = numpy.array([simulator.ID(simulator.state.next_id)
                                        for cell in xrange(len(brian_cells))],
                                     simulator.ID)
        for cell in self.all_cells:
            cell.parent = self
            cell.parent_group = brian_cells

        self._mask_local = numpy.ones((self.size,), bool) # all cells are local. This doesn't seem very efficient.
        self.first_id    = self.all_cells[0]
        self.last_id     = self.all_cells[-1]
        self.brian_cells = brian_cells
        simulator.state.network.add(brian_cells)

    def _get_parameters(self, *names):
        """
        Return a ParameterSpace containing native parameters
        """
        params = {}
        for name in names:
            params[name] = getattr(self.brian_cells, name)
            assert isinstance(params[name], numpy.ndarray)
        return ParameterSpace(params, size=self.size)

    def _set_parameters(self, parameter_space):
        """
        parameter_space should contain native parameters
        """
        parameter_space.evaluate(simplify=False)
        for name, value in parameter_space.items():
            setattr(self.brian_cells, name, value)

    def _set_initial_value_array(self, variable, value):
        if variable is 'v':
            value = value*mV
        if not hasattr(value, "__len__"):
            value = value*numpy.ones((len(self),))
        self.brian_cells.initial_values[variable] = value
        self.brian_cells.initialize()


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None, synapse_dynamics=None, label=None, rng=None):
        """
        presynaptic_population and postsynaptic_population - Population objects.

        source - string specifying which attribute of the presynaptic cell
                 signals action potentialss

        target - string specifying which synapse on the postsynaptic cell to
                 connect to

        If source and/or target are not given, default values are used.

        method - a Connector object, encapsulating the algorithm to use for
                 connecting the neurons.

        synapse_dynamics - a `SynapseDynamics` object specifying which
        synaptic plasticity mechanisms to use.

        rng - specify an RNG object to be used by the Connector.
        """
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population, method,
                                   source, target, synapse_dynamics, label, rng)

        self._method           = method
        self._connections      = None
        self.synapse_type      = target or 'excitatory'

        #if isinstance(presynaptic_population, common.Assembly) or isinstance(postsynaptic_population, common.Assembly):
            #raise Exception("Projections with Assembly objects are not working yet in Brian")

        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                if self.synapse_dynamics.slow:
                    raise Exception("It is not currently possible to have both short-term and long-term plasticity at the same time with this simulator.")
                else:
                    self._plasticity_model = "tsodyks_markram_synapse"
            elif synapse_dynamics.slow:
                self._plasticity_model = "stdp_synapse"
        else:
            self._plasticity_model = "static_synapse"

        self._n                 = {}
        self._brian_connections = {}
        self._indices           = {}
        self._populations      = [{}, {}]

        method.connect(self)
        self._finalize()

        if self._plasticity_model != "static_synapse":
            for key in self._brian_connections.keys():
                synapses = self._brian_connections[key]
                if self._plasticity_model is "stdp_synapse":
                    parameters   = self.synapse_dynamics.slow.all_parameters
                    if common.is_conductance(self.post[0]):
                        units = uS
                    else:
                        units = nA
                    stdp = simulator.STDP(synapses,
                                        parameters['tau_plus'] * ms,
                                        parameters['tau_minus'] * ms,
                                        parameters['A_plus'],
                                        -parameters['A_minus'],
                                        parameters['mu_plus'],
                                        parameters['mu_minus'],
                                        wmin = parameters['w_min'] * units,
                                        wmax = parameters['w_max'] * units)
                    simulator.state.add(stdp)
                elif self._plasticity_model is "tsodyks_markram_synapse":
                    parameters   = self.synapse_dynamics.fast.parameters
                    stp = brian.STP(synapses, parameters['tau_rec'] * ms,
                                            parameters['tau_facil'] * ms,
                                            parameters['U'])
                    simulator.state.add(stp)

    def __len__(self):
        """Return the total number of connections in this Projection."""
        result = 0
        for key in self._brian_connections.keys():
            result += self._n[key]
        return result

    def _get_indices(self):
        sources = numpy.array([], int)
        targets = numpy.array([], int)
        for key in self._brian_connections.keys():
            paddings = self._populations[0][key[0]], self._populations[1][key[1]]
            sources  = numpy.concatenate((sources, self._indices[key][0] + paddings[0]))
            targets  = numpy.concatenate((targets, self._indices[key][1] + paddings[1]))
        return sources.astype(int), targets.astype(int)

    def __getitem__(self, i):
        """Return the `i`th connection as a Connection object."""
        cumsum_idx     = numpy.cumsum(self._n.values())
        if isinstance(i, slice):
            idx  = numpy.searchsorted(cumsum_idx, numpy.arange(*i.indices(i.stop)), 'left')
            keys = [self._brian_connections.keys()[j] for j in idx]
        else:
            idx  = numpy.searchsorted(cumsum_idx, i, 'left')
            keys = self._brian_connections.keys()[idx]
        global_indices = self._get_indices()
        if isinstance(i, int):
            if i < len(self):
                pad        = i - cumsum_idx[idx]
                local_idx  = self._indices[keys][0][pad], self._indices[keys][1][pad]
                local_addr = global_indices[0][i], global_indices[1][i]
                return Connection(self._brian_connections[keys], local_idx, local_addr)
            else:
                raise IndexError("%d > %d" % (i, len(self)-1))
        elif isinstance(i, slice):
            if i.stop < len(self):
                res = []
                for count, j in enumerate(xrange(*i.indices(i.stop))):
                    key = keys[count]
                    pad = j - cumsum_idx[idx[count]]
                    local_idx  = self._indices[key][0][pad], self._indices[key][1][pad]
                    local_addr = global_indices[0][j], global_indices[1][j]
                    res.append(simulator.Connection(self._brian_connections[key], local_idx, local_addr))
                return res
            else:
                raise IndexError("%d > %d" % (i.stop, len(self)-1))

    def __connection_generator(self):
        """Yield each connection in turn."""
        global_indices = self._get_indices()
        count = 0
        for key in self._brian_connections.keys():
            bc = self._brian_connections[key]
            for i in xrange(bc.W.getnnz()):
                #local_idx  = self._indices[key][0][i], self._indices[key][0][i]
                #local_addr = global_indices[0][count], global_indices[1][count]
                yield simulator.Connection(bc, self._indices[key])
                count += 1

    def __iter__(self):
        """Return an iterator over all connections in this Projection."""
        return self.__connection_generator()

    def _finalize(self):
        for key in self._brian_connections.keys():
            self._indices[key]  = self._brian_connections[key].W.nonzero()
            self._brian_connections[key].compress()

    def _get_brian_connection(self, source_group, target_group, synapse_obj, weight_units, homogeneous=False):
        """
        Return the Brian Connection object that connects two NeuronGroups with a
        given synapse model.

        source_group -- presynaptic Brian NeuronGroup.
        target_group -- postsynaptic Brian NeuronGroup
        synapse_obj  -- name of the variable that will be modified by synaptic
                        input.
        weight_units -- Brian Units object: nA for current-based synapses,
                        uS for conductance-based synapses.
        """
        key = (source_group, target_group, synapse_obj)
        if not self._brian_connections.has_key(key):
            assert isinstance(source_group, brian.NeuronGroup)
            assert isinstance(target_group, brian.NeuronGroup), type(target_group)
            assert isinstance(synapse_obj, basestring), "%s (%s)" % (synapse_obj, type(synapse_obj))
            try:
                max_delay = simulator.state.max_delay*ms
            except Exception:
                raise Exception("Simulation timestep not yet set. Need to call setup()")
            if not homogeneous:
                self._brian_connections[key] = brian.DelayConnection(source_group,
                                                               target_group,
                                                               synapse_obj,
                                                               max_delay=max_delay)
            else:
                self._brian_connections[key] = brian.Connection(source_group,
                                                          target_group,
                                                          synapse_obj,
                                                          max_delay=simulator.state.max_delay*ms)
            self._brian_connections[key].weight_units = weight_units
            simulator.state.add(self._brian_connections[key])
            self._n[key] = 0
        return self._brian_connections[key]

    def _detect_parent_groups(self, cells):
        groups = {}
        for index, cell in enumerate(cells):
            group = cell.parent_group
            if not groups.has_key(group):
                groups[group] = [index]
            else:
                groups[group] += [index]
        return groups

    def _divergent_connect(self, source, targets, weights, delays, homogeneous=False):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        #print "connecting", source, "to", targets, "with weights", weights, "and delays", delays
        if not core.is_listlike(targets):
            targets = [targets]
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        if not isinstance(source, common.IDMixin):
            raise errors.ConnectionError("source should be an ID object, actually %s" % type(source))
        for target in targets:
            if not isinstance(target, common.IDMixin):
                raise errors.ConnectionError("Invalid target ID: %s" % target)
        assert len(targets) == len(weights) == len(delays), "%s %s %s" % (len(targets),len(weights),len(delays))
        if common.is_conductance(targets[0]):
            units = uS
        else:
            units = nA
        synapse_type = self.synapse_type or "excitatory"
        try:
            source_group = source.parent_group
        except AttributeError, errmsg:
            raise errors.ConnectionError("%s. Maybe trying to connect from non-existing cell (ID=%s)." % (errmsg, source))
        groups = self._detect_parent_groups(targets) # we assume here all the targets belong to the same NeuronGroup

        weights = numpy.array(weights) * units
        delays  = numpy.array(delays) * ms
        weights[weights == 0] = simulator.ZERO_WEIGHT

        for target_group, indices in groups.items():
            synapse_obj = targets[indices[0]].parent.celltype.synapses[synapse_type]
            bc          = self._get_brian_connection(source_group, target_group, synapse_obj, units, homogeneous)
            padding     = (int(source.parent.first_id), int(targets[indices[0]].parent.first_id))
            src         = int(source) - padding[0]
            mytargets   = numpy.array(targets, int)[indices] - padding[1]
            bc.W.rows[src] = mytargets
            bc.W.data[src] = weights[indices]
            if not homogeneous:
                bc.delayvec.rows[src] = mytargets
                bc.delayvec.data[src] = delays[indices]
            else:
                bc.delay = int(delays[0] / bc.source.clock.dt)
            key = (source_group, target_group, synapse_obj)
            self._n[key] += len(mytargets)

            pop_sources = self._populations[0]
            if len(pop_sources) is 0:
                pop_sources[source_group] = 0
            elif not pop_sources.has_key(source_group):
                pop_sources[source_group] = numpy.sum([len(item) for item in pop_sources.keys()])
            pop_targets = self._populations[1]
            if len(pop_targets) is 0:
                pop_targets[target_group] = 0
            elif not pop_targets.has_key(target_group):
                pop_targets[target_group] = numpy.sum([len(item) for item in pop_targets.keys()])

    def saveConnections(self, file, gather=True, compatible_output=True):
        """
        Save connections to file in a format suitable for reading in with a
        FromFileConnector.
        """
        lines   = numpy.empty((len(self), 4))
        padding = 0
        for key in self._brian_connections.keys():
            bc   = self._brian_connections[key]
            size = bc.W.getnnz()
            lines[padding:padding+size,0], lines[padding:padding+size,1] = self._indices[key]
            lines[padding:padding+size,2] = bc.W.alldata / bc.weight_units
            if isinstance(bc, brian.DelayConnection):
                lines[padding:padding+size,3] = bc.delay.alldata / ms
            else:
                lines[padding:padding+size,3] = bc.delay * bc.source.clock.dt / ms
            padding += size

        logger.debug("--- Projection[%s].__saveConnections__() ---" % self.label)

        if isinstance(file, basestring):
            file = files.StandardTextFile(file, mode='w')

        file.write(lines, {'pre' : self.pre.label, 'post' : self.post.label})
        file.close()

    def set(self, name, value):
        """
        Set connection attributes for all connections in this Projection.

        `name`  -- attribute name
        `value` -- the attribute numeric value, or a list/1D array of such
                   values of the same length as the number of local connections,
                   or a 2D array with the same dimensions as the connectivity
                   matrix (as returned by `get(format='array')`).
        """
        for key in self._brian_connections.keys():
            bc = self._brian_connections[key]
            padding = 0
            if name == 'weight':
                M = bc.W
                units = bc.weight_units
            elif name == 'delay':
                M = bc.delay
                units = ms
            else:
                raise Exception("Setting parameters other than weight and delay not yet supported.")
            value = value*units
            if numpy.isscalar(value):
                if (name == 'weight') or (name == 'delay' and isinstance(bc, brian.DelayConnection)):
                    for row in xrange(M.shape[0]):
                        M.set_row(row, value)
                elif (name == 'delay' and isinstance(bc, brian.Connection)):
                    bc.delay = int(value / bc.source.clock.dt)
                else:
                    raise Exception("Setting a non appropriate parameter")
            elif isinstance(value, numpy.ndarray) and len(value.shape) == 2:
                if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                    raise Exception("FastConnector have been used, and only fixed homogeneous delays are allowed")
                address_gen = ((i, j) for i,row in enumerate(bc.W.rows) for j in row)
                for (i,j) in address_gen:
                    M[i,j] = value[i,j]
            elif core.is_listlike(value):
                N = M.getnnz()
                assert len(value[padding:padding+N]) == N
                if (name == 'delay') and not isinstance(bc, brian.DelayConnection):
                    raise Exception("FastConnector have been used: only fixed homogeneous delays are allowed")
                M.alldata = value
            else:
                raise Exception("Values must be scalars or lists/arrays")
            padding += M.getnnz()

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
        if such a connection exists. If there are no such connections, X_ij will
        be NaN.
        """
        values = numpy.array([])
        for key in self._brian_connections.keys():
            bc = self._brian_connections[key]
            if parameter_name == "weight":
                values = numpy.concatenate((values, bc.W.alldata / bc.weight_units))
            elif parameter_name == 'delay':
                if isinstance(bc, brian.DelayConnection):
                    values = numpy.concatenate((values, bc.delay.alldata / ms))
                else:
                    data   = bc.delay * bc.source.clock.dt * numpy.ones(bc.W.getnnz()) /ms
                    values = numpy.concatenate((values, data))
            else:
                raise Exception("Getting parameters other than weight and delay not yet supported.")

        if format == 'list':
            values = values.tolist()
        elif format == 'array':
            values_arr = numpy.nan * numpy.ones((self.parent.pre.size, self.parent.post.size))
            sources, targets = self._indices
            values_arr[sources, targets] = values
            values = values_arr
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
