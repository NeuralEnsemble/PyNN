# encoding: utf-8
"""
MOOSE implementation of the PyNN API

Authors: Subhasis Ray and Andrew Davison

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import moose
import numpy
import shutil
import os.path
from pyNN.moose import simulator
from pyNN import common, recording, core

from pyNN.connectors import FixedProbabilityConnector, AllToAllConnector, OneToOneConnector
from pyNN.moose.standardmodels.cells import SpikeSourcePoisson, SpikeSourceArray, HH_cond_exp, IF_cond_exp, IF_cond_alpha
from pyNN.moose.cells import temporary_directory
from pyNN.moose.recording import Recorder
from pyNN import standardmodels

import logging
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================


def setup(timestep=0.1, min_delay=0.1, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    max_delay = extra_params.get('max_delay', 10.0)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.dt = timestep
    if not os.path.exists(temporary_directory):
        os.mkdir(temporary_directory)
    return 0


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    simulator.recorder_list = []
    shutil.rmtree(temporary_directory, ignore_errors=True)
    moose.PyMooseBase.endSimulation()

run, run_until = common.build_run(simulator)
run_for = run

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
    _assembly_class = Assembly

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


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

    def _create_cells(self, cellclass, cellparams, n):
        """
        Create cells in MOOSE.

        `cellclass`  -- a PyNN standard cell or a native MOOSE model.
        `cellparams` -- a dictionary of cell parameters.
        `n`          -- the number of cells to create
        """
        assert n > 0, 'n must be a positive integer'
        if isinstance(cellclass, type) and issubclass(cellclass, standardmodels.StandardCellType):
            celltype = cellclass(cellparams)
        else:
            print(cellclass)
            raise Exception("Only standard cells currently supported.")
        self.first_id = simulator.state.gid_counter
        self.last_id = simulator.state.gid_counter + n - 1
        self.all_cells = numpy.array([simulator.ID(id)
                                      for id in range(self.first_id, self.last_id + 1)],
                                     dtype=simulator.ID)

        # mask_local is used to extract those elements from arrays that apply to the cells on the current node
        # round-robin distribution of cells between nodes
        self._mask_local = self.all_cells % simulator.state.num_processes == simulator.state.mpi_rank

        for id in self.all_cells:
            id.parent = self
            id._build_cell(celltype.model, celltype.parameters)
        simulator.state.gid_counter += n


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
        assert synapse_dynamics is None, "don't yet handle synapse dynamics"
        self.synapse_model = None
        self.connections = []

        # Create connections
        method.connect(self)

    def _divergent_connect(self, source, targets, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `source`  -- the ID of the pre-synaptic cell.
        `targets` -- a list/1D array of post-synaptic cell IDs, or a single ID.
        `weight`  -- a list/1D array of connection weights, or a single weight.
                     Must have the same length as `targets`.
        `delays`  -- a list/1D array of connection delays, or a single delay.
                     Must have the same length as `targets`.
        """
        if not isinstance(source, int) or source > simulator.state.gid_counter or source < 0:
            errmsg = "Invalid source ID: %s (gid_counter=%d)" % (source, simulator.state.gid_counter)
            raise errors.ConnectionError(errmsg)
        if not core.is_listlike(targets):
            targets = [targets]

        weights = weights * 1000.0  # scale units
        if isinstance(weights, float):
            weights = [weights]
        if isinstance(delays, float):
            delays = [delays]
        assert len(targets) > 0
        # need to scale weights for appropriate units
        for target, weight, delay in zip(targets, weights, delays):
            if target.local:
                if not isinstance(target, common.IDMixin):
                    raise errors.ConnectionError("Invalid target ID: %s" % target)
                if self.synapse_type == "excitatory":
                    synapse_object = target._cell.esyn
                elif self.synapse_type == "inhibitory":
                    synapse_object = target._cell.isyn
                else:
                    synapse_object = getattr(target._cell, self.synapse_type)
                source._cell.source.connect('event', synapse_object, 'synapse')
                synapse_object.n_incoming_connections += 1
                index = synapse_object.n_incoming_connections - 1
                synapse_object.setWeight(index, weight)
                synapse_object.setDelay(index, delay)
                self.connections.append((source, target, index))

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
