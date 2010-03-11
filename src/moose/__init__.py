# encoding: utf-8
"""
MOOSE implementation of the PyNN API

Authors: Subhasis Ray and Andrew Davison

$Id:$
"""

import moose
from pyNN.moose import simulator
from pyNN import common, recording
common.simulator = simulator
recording.simulator = simulator

from pyNN.moose.cells import *
from pyNN.moose.recording import *

import logging
logger = logging.getLogger("PyNN")

# ==============================================================================
#   Functions for simulation set-up and control
# ==============================================================================

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.dt = timestep
    return 0

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for recorder in simulator.recorder_list:
        recorder.write(gather=True, compatible_output=compatible_output)
    moose.PyMooseBase.endSimulation()

def run(simtime):
	"""Run the simulation for simtime"""
	simulator.run(simtime)


# ==============================================================================
#   Functions returning information about the simulation state
# ==============================================================================

get_current_time = common.get_current_time
get_time_step = common.get_time_step
get_min_delay = common.get_min_delay
get_max_delay = common.get_max_delay
num_processes = common.num_processes
rank = common.rank   

# ==============================================================================
#   Low-level API for creating, connecting and recording from individual neurons
# ==============================================================================

create = common.create

#connect = common.connect

#set = common.set

record = common.build_record('spikes', simulator)

record_v = common.build_record('v', simulator)

record_gsyn = common.build_record('gsyn', simulator)
	

