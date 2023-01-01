"""
Arbor implementation of the PyNN API, for testing and documentation purposes.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from pyNN.standardmodels import StandardCellType
from pyNN.recording import get_io
from pyNN.arborproto import simulator
from pyNN.arborproto.standardmodels.cells import *
from pyNN.arborproto.connectors import *
from pyNN.arborproto.standardmodels.synapses import *
from pyNN.arborproto.standardmodels.electrodes import *
from pyNN.arborproto.standardmodels.ion_channels import *
from pyNN.arborproto.standardmodels.ionic_species import *
from pyNN.arborproto.populations import Population, PopulationView, Assembly
from pyNN.arborproto.projections import Projection
from pyNN.arborproto.connectors import *
# from pyNN.arbor.cells import NativeCellType, IntFire1, IntFire2, IntFire4
import pyNN.neuroml as neuroml

try:
    import pyNN.nineml as nineml
except ImportError:
    pass  # nineml is an optional dependency

import logging

logger = logging.getLogger("PyNN")


# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    # StandardCellType is part of the API therefore this assumes that it is implemented here as backend-arbor.
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY, **extra_params):
    """
    Called at the very beginning of a script.

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.

    <<TODO: Redocument for Arbor>>
    NEURON specific extra_params:

    use_cvode - use the NEURON cvode solver. Defaults to False.
      Optional cvode Parameters:
      -> rtol - specify relative error tolerance
      -> atol - specify absolute error tolerance

    native_rng_baseseed - added to MPI.rank to form seed for SpikeSourcePoisson, etc.
    default_maxstep - TODO

    returns: MPI rank

    """
    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = extra_params.get('rank', 0)
    simulator.state.num_processes = extra_params.get('num_processes', 1)
    # for MultiCompartmentNeuron
    # if "mcclass_attr" in extra_params:
    #     mc_attributes = ["ion_channels", "ionic_species", "post_synaptic_entities"]
    #     [setattr(MultiCompartmentNeuron, keyword, extra_params[keyword])
    #      for keyword in mc_attributes if keyword in extra_params]
    # >> > setattr(sim.MultiCompartmentNeuron, "ion_channels", expar["ion_channels"])
    # >> > sim.MultiCompartmentNeuron.ion_channels.keys()
    # dict_keys(['pas', 'na', 'kdr'])
    # >> > cc = sim.MultiCompartmentNeuron
    # >> > cc.ion_channels.keys()
    # dict_keys(['pas', 'na', 'kdr'])
    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    # should have common implementation of end()


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
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

# create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)


# record = common.build_record(simulator)

def record(self, **parameters):
    pass


def record_v(source, filename): return record(['v'], source, filename)


def record_gsyn(source, filename): return record(['gsyn_exc', 'gsyn_inh'], source, filename)
