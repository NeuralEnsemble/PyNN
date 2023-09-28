"""

"""

from collections import defaultdict
from itertools import repeat

import arbor

from .. import common
from ..core import ezip
from ..space import Space
from . import simulator


class ConnectionGroup:
    """

    """

    def __init__(self, pre, post, receptor_type, location_selector, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        self.receptor_type = receptor_type
        self.location_selector = location_selector
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
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        for pre_idx, other in ezip(presynaptic_indices, *connection_parameters.values()):
            other_attributes = dict(zip(connection_parameters.keys(), other))

            self.connections[postsynaptic_index].append(
                ConnectionGroup(pre_idx, postsynaptic_index, self.receptor_type, location_selector, **other_attributes)
            )

    def arbor_connections(self, gid):
        """Return a list of incoming connections to the cell with the given gid"""
        try:
            postsynaptic_index = self.post.id_to_index(gid)
        except IndexError:
            return []
        else:
            if self.pre.celltype.injectable:
                source = "detector"
            else:
                source = "spike-source"

            connections = []
            all_labels = list(self.post._arbor_cell_description[postsynaptic_index]["labels"])
            for cg in self.connections[postsynaptic_index]:
                if cg.location_selector in (None, "all"):
                    target_labels = [lbl for lbl in all_labels if lbl.startswith(cg.receptor_type)]
                    for target in target_labels:
                        connections.append(
                            arbor.connection(
                                (self.pre[cg.presynaptic_index], source),
                                target,
                                cg.weight,
                                cg.delay
                            )
                    )
                else:
                    raise NotImplementedError()
            return connections
