# encoding: utf-8
"""
nrnpython implementation of the PyNN API.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import warnings
try:
    from mpi4py import MPI
except ImportError:
    warnings.warn("mpi4py not available")
from pyNN.random import NumpyRNG, GSLRNG
from pyNN import common, core, space, __doc__
from pyNN.common.control import DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY
from pyNN.standardmodels import StandardCellType
from pyNN.recording import get_io
from pyNN.space import Space
from pyNN.neuron import simulator
from pyNN.neuron.random import NativeRNG
from pyNN.neuron.standardmodels.cells import *
from pyNN.neuron.connectors import *
from pyNN.neuron.standardmodels.synapses import *
from pyNN.neuron.standardmodels.electrodes import *
from pyNN.neuron.populations import Population, PopulationView, Assembly
from pyNN.neuron.projections import Projection
from pyNN.neuron.cells import NativeCellType
import numpy

import logging
from neuron import h
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Utility functions
# ==============================================================================

def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values() if (isinstance(obj, type) and 
                                                           issubclass(obj, StandardCellType) and
                                                           obj is not StandardCellType)]

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          max_delay=DEFAULT_MAX_DELAY, **extra_params):
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
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.initializer.clear()
    simulator.state.clear()
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    if 'use_cvode' in extra_params:
        simulator.state.cvode.active(int(extra_params['use_cvode']))
        if 'rtol' in extra_params:
            simulator.state.cvode.rtol(float(extra_params['rtol']))
        if 'atol' in extra_params:
            simulator.state.cvode.atol(float(extra_params['atol']))
    if 'native_rng_baseseed' in extra_params:
        simulator.state.native_rng_baseseed = int(extra_params['native_rng_baseseed'])
    if 'default_maxstep' in extra_params:
        simulator.state.default_maxstep=float(extra_params['default_maxstep'])
    return rank()

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []
    #simulator.state.finalize()

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

create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)

set = common.set

record = common.build_record(simulator)

record_v = lambda source, filename: record(['v'], source, filename)

record_gsyn = lambda source, filename: record(['gsyn_exc', 'gsyn_inh'], source, filename)

# ==============================================================================
