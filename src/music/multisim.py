"""
This module supports running simulations with multiple, interacting simulators
at once, provided those simulators support the MUSIC communication interface.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import music
import warnings


# This is the map between simulator(proxies) and music Application instances
application_map = {}

# Simulator/proxy for this rank
this_simulator = None

# Application object for this rank
this_music_app = None

# This is the map between simulator name and the name of the
# corresponding PyNN backend.
backends = { 'nest' : 'nest',
             'neuron' : 'neuron' }


def getBackend(name):
    exec('import pyNN.%s' % backends[name])
    return eval('pyNN.%s' % backends[name])


class Config(object):
    """Store configuration information for a MUSIC-capable application."""
    
    def __init__(self, name=None, np=1, binary=None, args=None):
        self.name = name
        self.num_nodes = np
        self.executable_path = binary
        self.args = args


def setup(*configurations):
    """
    Specify PyNN module ranks and external MUSIC programs to be launched.

    Return either the real PyNN module for each requested backend simulator (if
    that simulator is running on the current MPI node) or a `ProxySimulator`
    object (if the requested simulator is not running on the current node).
    """
    global this_simulator, this_music_app
    
    # Parameter checking
    for config in configurations:
        assert isinstance(config, Config)

    
    # now do MUSIC launch phase with delayed port setup
    # this should set the "simulator" attribute of each Config to either the
    # real simulator backend module (e.g. pyNN.nest) or a ProxySimulator
    # instance
    simulators = []
    
    for config in configurations:
        application = music.Application(name = config.name, 
                                        np = config.num_nodes,
                                        binary = config.executable_path,
                                        args = config.args)
        if config.name in backends:
            if application.this:
                # Tell the MUSIC library to postpone setup until first
                # port creation. Must make this call after
                # specifications of Applications.
                music.postponeSetup()
                simulator = getBackend(config.name)
            else:
                simulator = ProxySimulator()
        else:
            # This seems to be an external application
            simulator = ExternalApplication (config)

        application_map[simulator] = application
        if application.this:
            this_simulator = simulator
            this_music_app = application

        simulators.append(simulator)

    return simulators


# List of MUSIC projections
projections = []
            
def run(simtime):
    """Run the simulation for simtime ms."""
    #  finally setup delayed MUSIC port setup,
    # work through back-end specific eventport setup,
    # call run for backend assigned to this rank.
    music.define ('stoptime', simtime / 1000.0) # convert to s
    music.configure()
    
    for projection in projections:
        projection.pending_action()
        
    if isinstance(this_simulator, ExternalApplication):
        # Let MUSIC launch the application
        music.launch()
        # Will never get here

    this_simulator.run(simtime)


def end():
    this_simulator.end()


class Projection(object): # may wish to inherit from common.projections.Projection
    """
    docstring goes here
    """
    #queues projections and port configs to be evaluated in music.run
    
    def __init__(self, presynaptic_neurons, postsynaptic_neurons, method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        # Queue this projection
        projections.append (this)

    def pending_action ():
        """
        Execute queued action
        """
        # Call backends here
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


def connectPorts(fromSim, fromPortName, toSim, toPortName, width = None):
    fromApp = application_map[fromSim]
    toApp = application_map[toSim]
    music.connect(fromApp, fromPortName, toApp, toPortName, width)


from pyNN.common.control import DEFAULT_TIMESTEP, DEFAULT_MIN_DELAY, DEFAULT_MAX_DELAY
from pyNN.space import Space

class ProxySimulator(object):
    """
    A proxy for a real simulator backend, which has the same API but doesn't
    actually run simulations.
    """
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        warnings.warn ("returning None for " + name)
        return None
    
    # this may be of use outside of pyNN.music, but for simplicity I suggest
    # we develop it here for now and then think about moving/generalizing it
    # later    
    def setup(self, timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
              max_delay=DEFAULT_MAX_DELAY, **extra_params):
        pass
    
    def Population(self, size, cellclass, cellparams=None, structure=None, label=None):
        return ProxyPopulation()
    
    def Projection(self, presynaptic_neurons, postsynaptic_neurons, method,
                   source=None, target=None, synapse_dynamics=None,
                   label=None, rng=None):
        return ProxyProjection()

    def AllToAllConnector (self):
        return None
        
    def DistanceDependentProbabilityConnector(self, d_expression,
        allow_self_connections=True, weights=0.0, delays=None, space=Space()):
        pass
    
    def Assembly(self, *populations, **kwargs):
        pass

    def run(self, simtime):
        pass

    def end(self):
        pass


class ProxyPopulation(object):
    """
    """
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()


class ProxyProjection(object):
    """
    """
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()


class ProxyMethod(object):
    """
    """
    def __call__ (self, *args):
        return None


class ExternalApplication(object):
    """
    Represents an application external to PyNN
    """
    def __init__ (self, config):
        pass
