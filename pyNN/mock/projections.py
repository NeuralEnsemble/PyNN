from itertools import repeat
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
from pyNN import common
from pyNN.core import ezip
from pyNN.parameters import ParameterSpace
from pyNN.space import Space
from . import simulator


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        for name, value in attributes.items():
            setattr(self, name, value)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        return tuple([getattr(self, name) for name in attribute_names])


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)

        ## Create connections
        self.connections = []
        connector.connect(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        #parameter_space = ParameterSpace
        raise NotImplementedError

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        for pre_idx, other in ezip(presynaptic_indices, *connection_parameters.values()):
            other_attributes = dict(zip(connection_parameters.keys(), other))
            self.connections.append(
                Connection(pre_idx, postsynaptic_index, **other_attributes)
            )