"""
This module supports running simulations with multiple, interacting simulators
at once, provided those simulators support the MUSIC communication interface.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import music
#import warnings


# This is the map between simulator(proxies) and music Application instances
application_map = {}

# Simulator/proxy for this rank
this_simulator = None

# Application object for this rank
this_music_app = None

# The name of the backend in use on this rank, if applicable
this_backend = None

# This is the map between simulator name and the name of the
# corresponding PyNN backend.
backends = { 'nest' : 'nest',
             'neuron' : 'neuron' }


def getBackend(name):
    exec('import pyNN.%s' % backends[name])
    return eval('pyNN.%s' % backends[name])


def local_backend ():
    return this_backend


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
    global this_backend, this_simulator, this_music_app
    
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
                this_backend = config.name
                simulator.name = this_backend
            else:
                simulator = ProxySimulator()
        else:
            # This seems to be an external application
            simulator = ExternalApplication ()

        application_map[simulator] = application
        if application.this:
            simulator.local = True
            this_simulator = simulator
            this_music_app = application
        else:
            simulator.local = False

        simulators.append(simulator)

    return simulators


# List of pending actions
pending_actions = []
            
def run(simtime):
    """Run the simulation for simtime ms."""
    #  finally setup delayed MUSIC port setup,
    # work through back-end specific eventport setup,
    # call run for backend assigned to this rank.
    music.define ('stoptime', simtime / 1000.0) # convert to s
    music.configure()
    
    for actor in pending_actions:
        actor.pending_action()
        
    if isinstance(this_simulator, ExternalApplication):
        # Let MUSIC launch the application
        music.launch()
        # Will never get here

    this_simulator.run(simtime)


def end():
    this_simulator.end()


projection_number = 0

class Projection(object): # may wish to inherit from common.projections.Projection
    """
    music.Projection objects must be created in the same order on ALL ranks
    """
    #queues projections and port configs to be evaluated in music.run
    
    def __init__(self, presynaptic_neurons, postsynaptic_neurons, method,
                 source=None, target=None, synapse_dynamics=None,
                 label=None, rng=None):
        global projection_number

        # record parameters
        self.presynaptic_neurons=presynaptic_neurons
        self.postsynaptic_neurons=postsynaptic_neurons
        self.method=method
        self.source=source
        self.target=target
        self.synapse_dynamics=synapse_dynamics
        self.label=label
        self.rng=rng
        
        # number unique within local process and consistent with
        # corresponding projection in all other ranks (even those not
        # affected by this projection)
        self.number = projection_number
        projection_number += 1

        # inform MUSIC library
        self.output_port = output_port_name (presynaptic_neurons, self.number)
        self.input_port = input_port_name (presynaptic_neurons, self.number)
        self.width = len (presynaptic_neurons)
        connectPorts (sim_from_pop (presynaptic_neurons),
                      self.output_port,
                      sim_from_pop (postsynaptic_neurons),
                      self.input_port,
                      width=self.width)

        # Do we have to care at all?
        if isinstance(presynaptic_neurons, ProxyPopulation) \
           and isinstance(postsynaptic_neurons, ProxyPopulation):
                # Do nothing further
                return
            
        # Check that this_simulator supports music
        if not this_simulator.music_support:
            raise RuntimeError, 'Either pyNN.' + this_backend + """ doesn\'t yet support MUSIC
              or the simulator isn\'t installed with an enabled MUSIC interface"""
            
        # Queue this projection
        pending_actions.append (self)

    def pending_action (self):
        """
        Execute queued action
        """
        if isinstance(self.postsynaptic_neurons, ProxyPopulation):
            # Make backend create an EventOutputPort and map
            # presynaptic_neurons to that port.
            this_simulator.music_export (self.presynaptic_neurons,
                                         self.output_port)
            self.projection = ProxyProjection()
        else:
            self.projection = this_simulator.MusicProjection \
                        (self.input_port, self.width,
                         self.postsynaptic_neurons, self.method,
                         source=self.source, target=self.target,
                         synapse_dynamics=self.synapse_dynamics,
                         label=self.label, rng=self.rng)


    def __getattr__ (self, name):
        # Queued dispatch to self.projection
        # (Maybe this is actually just confusing to the user.)
        return QueuingCallable (lambda: self.projection.__getattribute__(name))

    
    #def _divergent_connect(self, source, targets, weights, delays):
        #raise NotImplementedError
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


output_ports = {}

def output_port_name(population, unique_number):
    if population in output_ports:
        return output_ports[population]
    else:
        name = 'out' + str (unique_number)
        output_ports[population] = name
        return name


def input_port_name(population, unique_number):
    return 'in' + str (unique_number)


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
        #warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()
    
    # this may be of use outside of pyNN.music, but for simplicity I suggest
    # we develop it here for now and then think about moving/generalizing it
    # later    
    def setup(self, timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
              max_delay=DEFAULT_MAX_DELAY, **extra_params):
        pass
    
    def Population(self, size, cellclass, cellparams=None, structure=None, label=None):
        return ProxyPopulation(self, size)
    
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
    def __init__(self, simulator, size):
        self.simulator = simulator
        self.size = size

    def __len__(self):
        return self.size
        
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        #warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()

def sim_from_pop(population):
    if isinstance(population, ProxyPopulation):
        return population.simulator
    else:
        return this_simulator
    

class ProxyProjection(object):
    """
    """
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        #warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()

    def __getattribute__ (self, name):
        return ProxyMethod()


class ProxyMethod(object):
    """
    """
    def __getattr__ (self, name):
        # Return None if we don't know what the remote simulator would
        # have returned.  For now, warn about it:
        #warnings.warn ("returning ProxyMethod for " + name)
        return ProxyMethod()

    def __call__ (self, *args):
        return ProxyMethod()


class QueuingCallable(object):
    """
    """
    def __init__ (self, callable):
        self.callable = callable

    def __call__ (self, *args):
        pending_actions.append (QueuedCall (self.callable, args))


class QueuedCall(object):
    """
    """
    def __init__ (self, callable, args):
        self.callable = callable
        self.args = args

    def pending_action (self):
        (self.callable)()(*self.args)


class ExternalApplication(object):
    """
    Represents an application external to PyNN
    """
    pass
