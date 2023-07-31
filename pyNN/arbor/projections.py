"""

"""

from collections import defaultdict
from itertools import repeat

import arbor

from .. import common
from ..core import ezip
from ..space import Space
from . import simulator


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, target, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.target = target
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

        #  Create connections
        self.connections = defaultdict(list)
        connector.connect(self)
        self._simulator.state.network.add_projection(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        raise NotImplementedError

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            location_selector=None,
                            **connection_parameters):
        if location_selector is None:
            target = self.receptor_type
        else:
            raise NotImplementedError()
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        for pre_idx, other in ezip(presynaptic_indices, *connection_parameters.values()):
            other_attributes = dict(zip(connection_parameters.keys(), other))
            self.connections[postsynaptic_index].append(
                Connection(pre_idx, postsynaptic_index, target, **other_attributes)
            )

    def arbor_connections(self, gid):
        """Return a list of incoming connections to the cell with the given gid"""
        try:
            postsynaptic_index = self.post.id_to_index(gid)
        except IndexError:
            return []
        else:
            return [
                arbor.connection(
                    (self.pre[c.presynaptic_index], "detector"),
                    c.target,
                    c.weight,
                    c.delay
                ) for c in self.connections[postsynaptic_index]
            ]
