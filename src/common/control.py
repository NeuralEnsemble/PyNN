"""
Functions for simulation set-up and control

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

DEFAULT_MAX_DELAY = 10.0
DEFAULT_TIMESTEP = 0.1
DEFAULT_MIN_DELAY = DEFAULT_TIMESTEP

if not 'simulator' in locals():
    simulator = None  # should be set by simulator-specific modules


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          max_delay=DEFAULT_MAX_DELAY, **extra_params):
    """
    Initialises/reinitialises the simulator. Any existing network structure is
    destroyed.
    
    extra_params contains any keyword arguments that are required by a given
    simulator but not by others.
    """
    invalid_extra_params = ('mindelay', 'maxdelay', 'dt')
    for param in invalid_extra_params:
        if param in extra_params:
            raise Exception("%s is not a valid argument for setup()" % param)
    if min_delay > max_delay:
        raise Exception("min_delay has to be less than or equal to max_delay.")
    if min_delay < timestep:
        raise Exception("min_delay (%g) must be greater than timestep (%g)" % (min_delay, timestep))

def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    raise NotImplementedError

def run(simtime):
    """Run the simulation for simtime ms."""
    raise NotImplementedError

def build_reset(simulator):
    def reset():
        """
        Reset the time to zero, neuron membrane potentials and synaptic weights to
        their initial values, and delete any recorded data. The network structure
        is not changed, nor is the specification of which neurons to record from.
        """
        simulator.reset()
    return reset

def build_state_queries(simulator):
    def get_current_time():
        """Return the current time in the simulation."""
        return simulator.state.t
    
    def get_time_step():
        """Return the integration time step."""
        return simulator.state.dt
    
    def get_min_delay():
        """Return the minimum allowed synaptic delay."""
        return simulator.state.min_delay
    
    def get_max_delay():
        """Return the maximum allowed synaptic delay."""
        return simulator.state.max_delay
    
    def num_processes():
        """Return the number of MPI processes."""
        return simulator.state.num_processes
    
    def rank():
        """Return the MPI rank of the current node."""
        return simulator.state.mpi_rank
    
    return get_current_time, get_time_step, get_min_delay, get_max_delay, num_processes, rank
