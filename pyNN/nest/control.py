"""
NEST implementation of functions for simulation set-up and control

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import warnings
import shutil
import nest
from ..common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from .. import common
from ..recording import get_io
from . import simulator

logger = logging.getLogger("PyNN")


# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):
    """
    Should be called at the very beginning of a script.

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.

    NEST-specific extra_params:

    `spike_precision`:
        should be "off_grid" (default) or "on_grid"
    `verbosity`:
        one of: "all", "info", "deprecated", "warning", "error", "fatal"
    `recording_precision`:
        number of decimal places (OR SIGNIFICANT FIGURES?) in recorded data
    `threads`:
        number of threads to use
    `rng_seed`:
        seed for the NEST random number generator
    `rng_type`:
        type of the NEST random number generator
        (see https://nest-simulator.rtfd.io/en/stable/guides/random_numbers.html#seed-the-random-number-generator)  # noqa:E501
    `t_flush`:
        extra time to run the simulation after using reset() to ensure
        the previous run does not influence the new one
    """
    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    for key in ("threads", "verbosity", "spike_precision", "recording_precision"):
        if key in extra_params:
            setattr(simulator.state, key, extra_params[key])
    # set kernel RNG seeds
    simulator.state.num_threads = extra_params.get('threads') or 1
    if 'grng_seed' in extra_params:
        warnings.warn("The setup argument 'grng_seed' is now 'rng_seed'")
        simulator.state.rng_seed = extra_params['grng_seed']
    if 'rng_seeds' in extra_params:
        warnings.warn("The setup argument 'rng_seeds' is no longer available. "
                      "Taking the first value for the global seed.")
        simulator.state.rng_seed = extra_params['rng_seeds'][0]
    if 'rng_seeds_seed' in extra_params:
        warnings.warn("The setup argument 'rng_seeds_seed' is now 'rng_seed'")
        simulator.state.rng_seed = extra_params['rng_seeds_seed']
    else:
        simulator.state.rng_seed = extra_params.get('rng_seed', 42)
    if "rng_type" in extra_params:
        nest.rng_type = extra_params["rng_type"]
    if "t_flush" in extra_params:
        # see https://github.com/nest/nest-simulator/issues/1618
        simulator.state.t_flush = extra_params["t_flush"]
    # set resolution
    simulator.state.dt = timestep
    # Set min_delay and max_delay
    simulator.state.set_delays(min_delay, max_delay)
    nest.SetDefaults('spike_generator', {'precise_times': True})
    return rank()


def end():
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        logger.debug("%s%s --> %s" % (population.label, variables, filename))
        io = get_io(filename)
        population.write_data(io, variables)
    for tempdir in simulator.state.tempdirs:
        shutil.rmtree(tempdir)
    simulator.state.tempdirs = []
    simulator.state.write_on_end = []


run, run_until = common.build_run(simulator)
run_for = run

reset = common.build_reset(simulator)

initialize = common.initialize

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time, get_time_step, get_min_delay, get_max_delay, \
    num_processes, rank = common.build_state_queries(simulator)
