"""
Export of PyNN scripts as NineML.

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nineml.user as nineml

from pyNN import connectors
from utility import build_parameter_set, catalog_url


class ConnectorMixin(object):

    def to_nineml(self, label):
        connector_parameters = {}
        for name in self.__class__.parameter_names:
            connector_parameters[name] = getattr(self, name)
        connection_rule = nineml.ConnectionRuleComponent(
                                    name="connection rule for projection %s" % label,
                                    definition=nineml.Definition(self.definition_url,
                                                                 "connection_generator"),
                                    parameters=build_parameter_set(connector_parameters))
        return connection_rule


class FixedProbabilityConnector(ConnectorMixin, connectors.FixedProbabilityConnector):
    definition_url = "%s/connectionrules/random_fixed_probability.xml" % catalog_url 
    parameter_names = ('p_connect', 'allow_self_connections')


class DistanceDependentProbabilityConnector(ConnectorMixin, connectors.DistanceDependentProbabilityConnector):
    definition_url = "%s/connectionrules/distance_dependent_probability.xml" % catalog_url
    parameter_names = ('d_expression', 'allow_self_connections')  # space


def list_connectors():
    return [FixedProbabilityConnector, DistanceDependentProbabilityConnector]
