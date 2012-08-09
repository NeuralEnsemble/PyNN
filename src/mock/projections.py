from itertools import repeat
from pyNN import common
from . import simulator

class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, source, target, weight, delay, **other_attributes):
        self.source = source
        self.target = target
        self.weight = weight
        self.delay = delay
        for name, value in other_attributes.items():
            setattr(self, name, value)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        attributes = []
        for name in attribute_names:
            if name == "weights":
                name = "weight"
            elif name == "delays":
                name = "delay"
            attributes.append(getattr(self, name))
        return tuple(attributes)


class Projection(common.Projection):
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population, method,
                 source=None, target=None, synapse_dynamics=None, label=None,
                 rng=None):
        common.Projection.__init__(self, presynaptic_population,
                                   postsynaptic_population, method, source,
                                   target, synapse_dynamics, label, rng)

        ## Deal with synaptic plasticity
        if self.synapse_dynamics:
            if self.synapse_dynamics.fast:
                pass
            if self.synapse_dynamics.slow:
                pass

        ## Create connections
        self.connections = []
        method.connect(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        pass

    def _convergent_connect(self, sources, target, weights, delays):
        """
        Connect a neuron to one or more other neurons with a static connection.

        `sources`  -- a 1D array of pre-synaptic cell IDs
        `target`   -- the ID of the post-synaptic cell.
        `weight`   -- a 1D array of connection weights, of the same length as
                      `sources`, or a single weight value.
        `delays`   -- a 1D array of connection delays, of the same length as
                      `sources`, or a single delay value.
        """

        if isinstance(weights, float):
            weights = repeat(weights)
        if isinstance(delays, float):
            delays = repeat(delays)
        for source, weight, delay in zip(sources, weights, delays):
            self.connections.append(
                Connection(source, target, weight=weight, delay=delay)
            )
