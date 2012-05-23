# -*- coding: utf-8 -*-
"""
NEST v2 implementation of the PyNN API.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id$
"""

import numpy
import nest
from pyNN.nest import simulator
from pyNN import common, recording, errors, space, __doc__

try:
    nest.GetStatus([numpy.int64(0)])
except NESTError:
    raise Exception("NEST built without NumPy support. Try rebuilding NEST after installing NumPy.")

if recording.MPI and (nest.Rank() != recording.mpi_comm.rank):
    raise Exception("MPI not working properly. Please make sure you import pyNN.nest before pyNN.random.")

import shutil
import logging
from pyNN.nest.cells import NativeCellType, native_cell_type
from pyNN.nest.synapses import NativeSynapseDynamics, NativeSynapseMechanism
from pyNN.nest.standardmodels.cells import *
from pyNN.nest.connectors import *
from pyNN.nest.standardmodels.synapses import *
from pyNN.nest.standardmodels.electrodes import *
from pyNN.nest.recording import *
from pyNN.random import NumpyRNG
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import Sequence

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
    
    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.
    
    NEST-specific extra_params:
    
    `spike_precision`:
        should be "on_grid" (default) or "off_grid"
    `verbosity`:
        INSERT DESCRIPTION OF POSSIBLE VALUES
    `recording_precision`:
        number of decimal places (OR SIGNIFICANT FIGURES?) in recorded data
    `threads`:
        number of threads to use
    `rng_seeds`:
        a list of seeds, one for each thread on each MPI process
    `rng_seeds_seed`:
        a single seed that will be used to generate random values for `rng_seeds`
    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    for key in ("verbosity", "spike_precision", "recording_precision",
                "threads"):
        if key in extra_params:
            setattr(simulator.state, key, extra_params[key])
    # set kernel RNG seeds
    simulator.state.num_threads = extra_params.get('threads') or 1
    if 'rng_seeds' in extra_params:
        simulator.state.rng_seeds = extra_params['rng_seeds']
    else:
        rng = NumpyRNG(extra_params.get('rng_seeds_seed', 42))
        n = simulator.state.num_processes * simulator.state.threads
        simulator.state.rng_seeds = rng.next(n, 'randint', (100000,)).tolist()
    # set resolution
    simulator.state.dt = timestep
    # Set min_delay and max_delay for all synapse models
    simulator.state.set_delays(min_delay, max_delay)
    return rank()


def end():
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = recording.get_io(filename)
        population.write_data(io, variables)
    for tempdir in simulator.state.tempdirs:
        shutil.rmtree(tempdir)
    simulator.state.tempdirs = []
    simulator.state.write_on_end = []

run = common.build_run(simulator)

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

class PopulationMixin(object):

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _set_parameters(self, parameter_space):
        """
        parameter_space should contain native parameters
        """
        param_dict = _build_params(parameter_space, numpy.where(self._mask_local)[0])
        ids = self.local_cells.tolist()
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = [id.source for id in ids]
        nest.SetStatus(ids, param_dict)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        ids = self.local_cells.tolist()
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            ids = [id.source for id in ids]
        parameter_array = numpy.array(nest.GetStatus(ids, names))
        parameter_dict = dict((name, parameter_array[:, col])
                              for col, name in enumerate(names))
        if "spike_times" in parameter_dict: # hack
            parameter_dict["spike_times"] = [Sequence(value) for value in parameter_dict["spike_times"]]
        return ParameterSpace(parameter_dict, size=self.size)


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView, PopulationMixin):
    _simulator = simulator
    _assembly_class = Assembly


def _build_params(parameter_space, mask_local, size=None, extra_parameters=None):
    """
    Return either a single parameter dict or a list of dicts, suitable for use
    in Create or SetStatus.
    """
    if size:
        parameter_space.size = size
    if parameter_space.is_homogeneous:
        parameter_space.evaluate(simplify=True)
        cell_parameters = parameter_space.as_dict()
        if extra_parameters:
            cell_parameters.update(extra_parameters)
        for name, val in cell_parameters.items():
            if isinstance(val, Sequence):
                cell_parameters[name] = val.value
    else:
        parameter_space.evaluate(mask=mask_local)
        cell_parameters = list(parameter_space) # may not be the most efficient way. Might be best to set homogeneous parameters on creation, then inhomogeneous ones using SetStatus. Need some timings.
        for D in cell_parameters:
            for name, val in D.items():
                if isinstance(val, Sequence):
                    D[name] = val.value
            if extra_parameters:
                D.update(extra_parameters)
    return cell_parameters


class Population(common.Population, PopulationMixin):
    """
    An array of neurons all of the same type. `Population' is used as a generic
    term intended to include layers, columns, nuclei, etc., of cells.
    """
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        super(Population, self).__init__(size, cellclass, cellparams, structure, initial_values, label)
        self._simulator.state.populations.append(self)

    def _create_cells(self):
        """
        Create cells in NEST using the celltype of the current Population.
        """
        # this method should never be called more than once
        # perhaps should check for that
        nest_model = self.celltype.nest_name[simulator.state.spike_precision]
        if isinstance(self.celltype, StandardCellType):
            params = _build_params(self.celltype.translated_parameters,
                                   None,
                                   size=self.size,
                                   extra_parameters=self.celltype.extra_parameters)
        else:
            params = _build_params(self.celltype.parameter_space,
                                   None,
                                   size=self.size)
        try:
            self.all_cells = nest.Create(nest_model, self.size, params=params)
        except nest.NESTError, err:
            if "UnknownModelName" in err.message and "cond" in err.message:
                raise errors.InvalidModelError("%s Have you compiled NEST with the GSL (Gnu Scientific Library)?" % err)
            raise errors.InvalidModelError(err)
        # create parrot neurons if necessary
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            self.all_cells_source = numpy.array(self.all_cells)  # we put the parrots into all_cells, since this will
            self.all_cells = nest.Create("parrot_neuron", self.size)     # be used for connections and recording. all_cells_source
            nest.Connect(self.all_cells_source, self.all_cells)  # should be used for setting parameters
        self.first_id = self.all_cells[0]
        self.last_id = self.all_cells[-1]
        self._mask_local = numpy.array(nest.GetStatus(self.all_cells, 'local'))
        self.all_cells = numpy.array([simulator.ID(gid) for gid in self.all_cells], simulator.ID)
        for gid in self.all_cells:
            gid.parent = self
        if hasattr(self.celltype, "uses_parrot") and self.celltype.uses_parrot:
            for gid, source in zip(self.all_cells, self.all_cells_source):
                gid.source = source

    def _set_initial_value_array(self, variable, value):
        variable = STATE_VARIABLE_MAP.get(variable, variable)
        value = value.evaluate(simplify=True)
        try:
            nest.SetStatus(self.local_cells.tolist(), variable, value)
        except nest.NESTError, e:
            if "Unused dictionary items" in e.message:
                logger.warning("NEST does not allow setting an initial value for %s" % variable)
                # should perhaps check whether value-to-be-set is the same as current value,
                # and raise an Exception if not, rather than just emit a warning.
            else:
                raise


class Projection(common.Projection):
    """
    A container for all the connections of a given type (same synapse type and
    plasticity mechanisms) between two populations, together with methods to set
    parameters of those connections, including of plasticity mechanisms.
    """
    _simulator = simulator

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
        synapse_model = synapse_dynamics._get_nest_synapse_model("projection_%d" % Projection._nProj)
        if synapse_model is None:
            self.synapse_model = 'static_synapse_%s' % id(self)
            nest.CopyModel('static_synapse', self.synapse_model)
        else:
            self.synapse_model = synapse_model
        self._sources = []
        self._connections = None

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
            file = recording.files.StandardTextFile(file, mode='w')

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

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)

# ==============================================================================
