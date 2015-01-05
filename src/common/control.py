# encoding: utf-8
"""
Common implementation of functions for simulation set-up and control

This module contains:
  * partial implementations of API functions which can be reused by
    backend-specific implementations (in some cases only the docstring
    is intended to be reused)
  * function factories for generating backend-specific API functions.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

DEFAULT_MAX_DELAY = 10.0
DEFAULT_TIMESTEP = 0.1
DEFAULT_MIN_DELAY = 'auto'

##if not 'simulator' in locals():
##    simulator = None  # should be set by simulator-specific modules
assert 'simulator' not in locals() ##


class BaseState(object):
    """Base class for simulator _State classes."""
    
    def __init__(self):
        """Initialize the simulator."""
        self.running = False
        self.t_start = 0
        self.write_on_end = [] # a list of (population, variable, filename) combinations that should be written to file on end()
        self.recorders = set([])


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          max_delay=DEFAULT_MAX_DELAY, **extra_params):
    """
    Initialises/reinitialises the simulator. Any existing network structure is
    destroyed.
    
    `timestep`, `min_delay` and `max_delay` should all be in milliseconds.

    `extra_params` contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    invalid_extra_params = ('mindelay', 'maxdelay', 'dt', 'time_step')
    for param in invalid_extra_params:
        if param in extra_params:
            raise Exception("%s is not a valid argument for setup()" % param)
    if min_delay != 'auto':
        if min_delay > max_delay:
            raise Exception("min_delay has to be less than or equal to max_delay.")
        if min_delay < timestep:
            raise Exception("min_delay (%g) must be greater than timestep (%g)" % (min_delay, timestep))


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    raise NotImplementedError


def build_run(simulator):
    def run_until(time_point, callbacks=None):
        """
        Run the simulation until `time_point` (in ms).
        
        `callbacks` is an optional list of callables, each of which should
        accept the current time as an argument, and return the next time it
        wishes to be called.
        """
        now = simulator.state.t
        if time_point - now < -simulator.state.dt/2.0:  # allow for floating point error
            raise ValueError("Time %g is in the past (current time %g)" % (time_point, now))
        if callbacks:
            callback_events = [(callback(simulator.state.t), callback)
                               for callback in callbacks]
            while simulator.state.t + 1e-9 < time_point:
                callback_events.sort(key=lambda cbe: cbe[0], reverse=True)
                next, callback = callback_events.pop()
                # collapse multiple events that happen within the same timestep
                active_callbacks = [callback]
                while len(callback_events) > 0 and\
                        abs(next - callback_events[-1][0]) < simulator.state.dt:
                    active_callbacks.append(callback_events.pop()[1])

                next = min(next, time_point)
                simulator.state.run_until(next)
                callback_events.extend((callback(simulator.state.t), callback)
                        for callback in active_callbacks)
        else:
            simulator.state.run_until(time_point)
        return simulator.state.t
    def run(simtime, callbacks=None):
        """
        Run the simulation for `simtime` ms.
        
        `callbacks` is an optional list of callables, each of which should
        accept the current time as an argument, and return the next time it
        wishes to be called.
        """
        return run_until(simulator.state.t + simtime, callbacks)
    return run, run_until


def build_reset(simulator):
    def reset(annotations={}):
        """
        Reset the time to zero, neuron membrane potentials and synaptic weights to
        their initial values, and delete any recorded data. The network structure
        is not changed, nor is the specification of which neurons to record from.
        """
        for recorder in simulator.state.recorders:
            recorder.store_to_cache(annotations)
        simulator.state.reset()
    return reset


def build_state_queries(simulator):
    def get_current_time():
        """Return the current time in the simulation (in milliseconds)."""
        return simulator.state.t

    def get_time_step():
        """Return the integration time step (in milliseconds)."""
        return simulator.state.dt

    def get_min_delay():
        """Return the minimum allowed synaptic delay (in milliseconds)."""
        return simulator.state.min_delay

    def get_max_delay():
        """Return the maximum allowed synaptic delay (in milliseconds)."""
        return simulator.state.max_delay

    def num_processes():
        """Return the number of MPI processes."""
        return simulator.state.num_processes

    def rank():
        """Return the MPI rank of the current node."""
        return simulator.state.mpi_rank

    return get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank
