"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from pyNN import common
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from pyNN.connectors import *
from pyNN.recording import *
from . import simulator
from .standardmodels import *
from .populations import Population, PopulationView, Assembly
from .projections import Projection
from neo.io import get_io


logger = logging.getLogger("PyNN")


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if isinstance(obj, type) and issubclass(obj, StandardCellType)]


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          **extra_params):

    max_delay = extra_params.get('max_delay', DEFAULT_MAX_DELAY)
    common.setup(timestep, min_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep  # move to common.setup?
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.mpi_rank = extra_params.get('rank', 0)
    simulator.state.num_processes = extra_params.get('num_processes', 1)
    #
    if "mcclass" in extra_params:
        mc_attribute = ["label", "ion_channels", "post_synaptic_entities"]
        _preset_mc_property(mc_attribute, extra_params)
        _preset_mc_property_value(mc_attribute, extra_params)
    #
    return rank()

def _preset_mc_property(mc_attribute, params):
    [setattr(params["mcclass"], keyword, property(params[keyword]))
     for keyword in mc_attribute if keyword in params]

def _preset_mc_property_value(mc_attribute, params):
    [setattr(params["mcclass"], keyword, params[keyword])
     for keyword in mc_attribute if keyword in params]


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

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)


record = common.build_record(simulator)


def record_v(source, filename): return record(['v'], source, filename)


def record_gsyn(source, filename): return record(['gsyn_exc', 'gsyn_inh'], source, filename)

