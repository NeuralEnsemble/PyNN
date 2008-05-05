"""Interfacing MOOSE to PyNN"""
import moose
from pyNN import common

def setup(timestep=0.1, min_delay=0.1, max_delay=10.0, debug=False, **extra_params):
    """
    Should be called at the very beginning of a script.
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    common.setup(timestep, min_delay, max_delay, debug, **extra_params)
    ctx = moose.PyMooseBase.getContext()
    ctx.setClock(0, timestep, 0)
    return 0

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    moose.PyMooseBase.endSimulation()
 

