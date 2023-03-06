"""
NEURON implementation of functions for simulation set-up and control

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from ..common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from .. import common
from ..recording import get_io
from . import simulator

logger = logging.getLogger("PyNN")


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY, **extra_params):
    """
    Should be called at the very beginning of a script.

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.

    NEURON specific extra_params:

    use_cvode - use the NEURON cvode solver. Defaults to False.
      Optional cvode Parameters:
      -> rtol - specify relative error tolerance
      -> atol - specify absolute error tolerance

    native_rng_baseseed - added to MPI.rank to form seed for SpikeSourcePoisson, etc.
    default_maxstep - TODO

    returns: MPI rank

    """
    common.setup(timestep, min_delay, **extra_params)
    simulator.initializer.clear()
    simulator.state.clear()
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    if 'use_cvode' in extra_params:
        simulator.state.record_sample_times = extra_params['use_cvode']
        simulator.state.cvode.active(int(extra_params['use_cvode']))
        if 'rtol' in extra_params:
            simulator.state.cvode.rtol(float(extra_params['rtol']))
        if 'atol' in extra_params:
            simulator.state.cvode.atol(float(extra_params['atol']))
    if 'native_rng_baseseed' in extra_params:
        simulator.state.native_rng_baseseed = int(extra_params['native_rng_baseseed'])
    if 'default_maxstep' in extra_params:
        simulator.state.default_maxstep = float(extra_params['default_maxstep'])
    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    # simulator.state.finalize()


run, run_until = common.build_run(simulator)
run_for = run

reset = common.build_reset(simulator)

initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
    num_processes, rank = common.build_state_queries(simulator)
