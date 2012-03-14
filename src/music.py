"""
This module supports running simulations with multiple, interacting simulators
at once, provided those simulators support the MUSIC communication interface.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

simulator_configurations = {}


class Config(object):
    """Store configuration information for a MUSIC-capable application."""
    
    def __init__(self, name, np, binary=None, args=None):
        self.name = name
        self.num_nodes = np
        self.executable_path = binary
        self.args = args


def setup(*configurations):
    """
    Specify PyNN module ranks and external MUSIC programs to be launched.
    """
    # Need Ctypes wrapper to libmusic pre-MPI-init rank assigner
    for config in configurations:
        assert isinstance(config, Config)
    simulator_configurations[config.name] = config
    # now do MUSIC launch phase with delayed port setup
    # this should set the "simulator" attribute of each Config to either the
    # real simulator backend module (e.g. pyNN.nest) or a ProxySimulator
    # instance
    raise NotImplementedError


def get_simulators(*simulator_names):
    """
    Return either the real PyNN module for each requested backend simulator (if
    that simulator is running on the current MPI node) or a `ProxySimulator`
    object (if the requested simulator is not running on the current node).
    """
    return (simulator_configurations[name].simulator for name in simulator_names)


def run(simtime):
    """Run the simulation for simtime ms."""
    #  finally setup delayed MUSIC port setup,
    # work through back-end specific eventport setup,
    # call run for backend assigned to this rank.
    raise NotImplementedError


class Projection(object): # may wish to inherit from common.projections.Projection
    """
    docstring goes here
    """
    #queues projections and port configs to be evaluated in music.run
    
    def __init__(self, presynaptic_neurons, postsynaptic_neurons, method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        raise NotImplementedError

    def _divergent_connect(self, source, targets, weights, delays):
        raise NotImplementedError
        #meop = pynest.Create("music_event_out_proxy")
        #meip = pynest.Create("music_event_in_proxy")
        #nest.SetStatus (meop, {'port_name' : 'spikes_out'})
        #nest.SetStatus (meip, {'port_name' : 'spikes_in', "music_channel": 0})
        #nest.DivergentConnect(meop, targets, {"music_channel": 0})
        #nest.Connect(source, meip)


class Port(object): # aka Pipe aka DataLink # other name suggestions welcome
    """
    Representation of a data connection from a PyNN Population to an external
    MUSIC program (data consumer) such as a visualization tool.
    """

    def __init__(self, population, variable, external_tool):
        raise NotImplementedError


from pyNN.common.control import DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY, DEFAULT_MAX_DELAY
from pyNN.space import Space

class ProxySimulator(object):
    """
    A proxy for a real simulator backend, which has the same API but doesn't
    actually run simulations.
    """
    # this may be of use outside of pyNN.music, but for simplicity I suggest
    # we develop it here for now and then think about moving/generalizing it
    # later    
    def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
              max_delay=DEFAULT_MAX_DELAY, **extra_params):
        pass
    
    def Population(self, size, cellclass, cellparams=None, structure=None, label=None):
        pass
    
    def DistanceDependentProbabilityConnector(self, d_expression,
        allow_self_connections=True, weights=0.0, delays=None, space=Space()):
        pass
    
    def Assembly(self, *populations, **kwargs):
        pass
    