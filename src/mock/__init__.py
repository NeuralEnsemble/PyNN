"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import numpy
from pyNN import common, recording
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from pyNN.standardmodels import SynapseDynamics, STDPMechanism
from pyNN.connectors import *
from .standardmodels import *
from itertools import repeat


logger = logging.getLogger("PyNN")


class MockSimulator(object):
    class ID(int, common.IDMixin):
        def __init__(self, n):
            """Create an ID object with numerical value `n`."""
            int.__init__(n)
            common.IDMixin.__init__(self)
    class State(common.control.BaseState):
        def __init__(self):
            common.control.BaseState.__init__(self)
            self.mpi_rank = 0
            self.num_processes = 1
            self.clear()
        def run(self, simtime):
            self.t += simtime
            self.running = True
        def clear(self):
            self.recorders = set([])
            self.id_counter = 42
            self.reset()
        def reset(self):
            """Reset the state of the current network to time t = 0."""
            self.running = False
            self.t = 0
            self.t_start = 0

    state = State()

simulator = MockSimulator()

class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids):
        pass

    def _get_spiketimes(self, id):
        return numpy.array([id, id+5], dtype=float) % get_current_time()

    def _get_all_signals(self, variable, ids):
        # assuming not using cvode, otherwise need to get times as well and use IrregularlySampledAnalogSignal
        n_samples = int(round(self._simulator.state.t/self._simulator.state.dt)) + 1
        return numpy.vstack((numpy.random.uniform(size=n_samples) for id in ids)).T

    @staticmethod
    def find_units(variable):
        if variable in recording.UNITS_MAP:
            return recording.UNITS_MAP[variable]
        else:
            raise Exception("units unknown")

    def _local_count(self, variable, filter_ids=None):
        N = {}
        if variable == 'spikes':
            for id in self.filter_recorded(variable, filter_ids):
                N[int(id)] = 2
        else:
            raise Exception("Only implemented for spikes")
        return N


def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if 'rank' in extra_params:
        simulator.state.mpi_rank = extra_params['rank']
    if 'num_processes' in extra_params:
        simulator.state.num_processes = extra_params['num_processes']
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    # should have common implementation of end()

run = common.build_run(simulator)

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
                    num_processes, rank = common.build_state_queries(simulator)


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
            if isinstance(value, numpy.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, size=self.size) # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        ps = self.parent._get_parameters(*self.celltype.get_translated_names())
        for name, value in parameter_space.items():
            ps[name][self.mask] = value.evaluate(simplify=True)
        ps.evaluate(simplify=True)
        self.parent._parameters = ps.as_dict()

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def _create_cells(self):
        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.translated_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.size = self.size
        parameter_space.evaluate(simplify=True)
        self._parameters = parameter_space.as_dict()
        id_range = numpy.arange(simulator.state.id_counter,
                                simulator.state.id_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)
        def is_local(id):
            return (id % simulator.state.num_processes) == simulator.state.mpi_rank
        self._mask_local = is_local(self.all_cells)
        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = self._parameters[name]
        return ParameterSpace(parameter_dict, size=self.size)

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        ps = self._get_parameters(*self.celltype.get_translated_names())
        ps.update(**parameter_space)
        ps.evaluate(simplify=True)
        self._parameters = ps.as_dict()


class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, source, target, weight, delay, **other_attributes):
        self.source = source
        self.target = target
        self.weight = weight
        self.delay = delay
        for name, value in other_attributes.items():
            setattr(self, name, value)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        attributes = []
        for name in attribute_names:
            if name == "weights":
                name = "weight"
            elif name == "delays":
                name = "delay"
            attributes.append(getattr(self, name))
        return tuple(attributes)


class Projection(common.Projection):
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None, synapse_dynamics=None, label=None,
                 rng=None):
        common.Projection.__init__(self, presynaptic_population,
                                   postsynaptic_population, method, source,
                                   target, synapse_dynamics, label, rng)

        ## Deal with synaptic plasticity
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                pass
            if self.synapse_dynamics.slow:
                pass

        ## Create connections
        self.connections = []
        method.connect(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        pass

    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `sources`  -- a 1D array of pre-synaptic cell IDs
        `target`   -- the ID of the post-synaptic cell.
        `weight`   -- a 1D array of connection weights, of the same length as
                      `sources`, or a single weight value.
        `delays`   -- a 1D array of connection delays, of the same length as
                      `sources`, or a single delay value.
        """

        if isinstance(weights, float):
            weights = repeat(weights)
        if isinstance(delays, float):
            delays = repeat(delays)
        for source, weight, delay in zip(sources, weights, delays):
            self.connections.append(
                Connection(source, target, weight=weight, delay=delay)
            )

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

#set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)
