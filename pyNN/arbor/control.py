"""


"""

from ..common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from .. import common
from ..recording import get_io
from . import simulator


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):
    """
    Should be called at the very beginning of a script.

    `timestep`, `min_delay` and `max_delay` should all be in milliseconds.

    Arbor requires every synaptic delay to be at least twice the time step, so
    the default synaptic delay is the larger of `min_delay` and twice `timestep`
    (or twice `timestep` if `min_delay` is "auto", the default).

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.

    Arbor-specific extra_params:

    `threads`:
        number of threads to use (default 1)
    `rng_seed`:
        seed for Arbor's random number generator, used for example by
        SpikeSourcePoisson (default 42)
    """
    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    simulator.state.set_num_threads(extra_params.get("threads") or 1)
    simulator.state.rng_seed = extra_params.get("rng_seed", 42)
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
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
