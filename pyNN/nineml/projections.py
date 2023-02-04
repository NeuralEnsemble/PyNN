# encoding: utf-8
"""
Export of PyNN scripts as NineML.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nineml.user as nineml

from pyNN import common
from pyNN.space import Space
from . import simulator
from .utility import catalog_url, build_parameter_set


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)
        self._simulator.state.net.projections.append(self)

    def __len__(self):
        return 0

    def to_nineml(self):
        safe_label = self.label.replace(u"→", "---")
        connection_rule = self._connector.to_nineml(safe_label)
        connection_type = nineml.ConnectionType(
            name="connection type for projection %s" % safe_label,
            definition=nineml.Definition("%s/connectiontypes/static_synapse.xml" % catalog_url,
                                         "dynamics"),
            parameters=build_parameter_set(self.synapse_type.native_parameters, self.shape))
        synaptic_responses = self.post.get_synaptic_response_components(self.receptor_type)
        synaptic_response, = synaptic_responses
        projection = nineml.Projection(
            name=safe_label,
            source=self.pre.to_nineml(),  # or just pass ref, and then resolve later?
            target=self.post.to_nineml(),
            rule=connection_rule,
            synaptic_response=synaptic_response,
            connection_type=connection_type)
        return projection
