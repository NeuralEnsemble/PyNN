"""
Brian 2 implementation of simulation control functions

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import brian2
from ..common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from .. import common
from ..recording import get_io
from . import simulator
from .standardmodels.electrodes import update_currents


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """

    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = 0
    simulator.state.num_processes = 1

    simulator.state.network.add(
        brian2.NetworkOperation(update_currents, when="start", clock=simulator.state.network.clock)
    )
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

get_current_time, get_time_step, get_min_delay, get_max_delay, \
    num_processes, rank = common.build_state_queries(simulator)
