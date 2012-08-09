"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2012 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from pyNN import common
from pyNN.standardmodels import SynapseDynamics, STDPMechanism
from pyNN.connectors import *
from . import simulator
from .standardmodels import *
from .populations import Population, PopulationView, Assembly
from .projections import Projection


logger = logging.getLogger("PyNN")


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

#            )

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector)

#set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)
