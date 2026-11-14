"""

"""

from collections import defaultdict
from itertools import repeat

import arbor
from arbor import units as U

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

    def _lif_post_cm_pF(self, gid):
        """The C_m (in pF) of the postsynaptic native lif cell with the given gid.

        The native C_m is stored in nF (its PyNN unit); the delta-weight charge
        relation ΔV[mV] = weight / C_m[pF] needs it in pF.
        """
        post_pop = self.post.parent if hasattr(self.post, "parent") else self.post
        native = post_pop._arbor_cell_description
        if not native._evaluated:
            native.evaluate()
        cm_nF = float(list(native)[post_pop.id_to_index(gid)]["C_m"])
        return 1000.0 * cm_nF

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
            is_lif_post = self.post.celltype.arbor_cell_kind == arbor.cell_kind.lif
            if is_lif_post:
                # A lif_cell's delta synapse adds weight/C_m [mV] to V_m, i.e. the
                # weight is a charge. PyNN's IF_curr_delta weight is a voltage step
                # (mV), so scale by the post cell's C_m [pF] to recover it.
                weight_scale = self._lif_post_cm_pF(gid)
            else:
                weight_scale = 1.0
                all_labels = list(self.post._arbor_cell_description[postsynaptic_index]["labels"])
            for cg in self.connections[postsynaptic_index]:
                if cg.location_selector in (None, "all"):
                    if is_lif_post:
                        # A native lif_cell has a single built-in delta synapse;
                        # excitatory vs inhibitory is set by the sign of the weight.
                        target_labels = ["syn"]
                    else:
                        target_labels = [lbl for lbl in all_labels if lbl.startswith(cg.receptor_type)]
                    for target in target_labels:
                        connections.append(
                            arbor.connection(
                                (self.pre[cg.presynaptic_index], source),
                                target,
                                cg.weight * weight_scale,
                                cg.delay * U.ms
                            )
                        )
                else:
                    raise NotImplementedError()
            return connections
