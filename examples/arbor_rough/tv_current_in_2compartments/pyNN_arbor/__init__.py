# ~mc/pyNN/arbor/__init__.py
# encoding: utf-8
"""
Arbor implementation of the PyNN API.
:copyright: Copyright 200x-20xx by the PyNN team, see AUTHORS.
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
from pyNN_arbor import simulator #from pyNN.neuron import simulator
#from pyNN.neuron.random import NativeRNG
from pyNN_arbor.standardmodels.cells import * #from pyNN.neuron.standardmodels.cells import *
#from pyNN.neuron.connectors import *
#from pyNN.neuron.standardmodels.synapses import *
#from pyNN.neuron.standardmodels.electrodes import *
from pyNN_arbor.standardmodels.ion_channels import * #from pyNN.neuron.standardmodels.ion_channels import *
#from pyNN.neuron.populations import Population, PopulationView, Assembly
#from pyNN.neuron.projections import Projection
#from pyNN.neuron.cells import NativeCellType
try:
    from . import nineml
except ImportError:
    pass  # nineml is an optional dependency

import logging
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

#sim.setup(timestep=0.025) #*ms)
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
    simulator.state.dt = timestep
    #return rank()

# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

#get_current_time, get_time_step, get_min_delay, get_max_delay, \
#            num_processes, rank = common.build_state_queries(simulator)