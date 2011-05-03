"""
:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import connectors
from utility import build_parameter_set, catalog_url
import nineml.user_layer as nineml
    
class ConnectorMixin(object):
    
    def to_nineml(self, label):
        connector_parameters = {}
        for name in self.__class__.parameter_names:
            connector_parameters[name] = getattr(self, name)
        connection_rule = nineml.ConnectionRule(
                                    name="connection rule for projection %s" % label,
                                    definition=nineml.Definition(self.definition_url),
                                    parameters=build_parameter_set(connector_parameters))
        return connection_rule


class FixedProbabilityConnector(connectors.FixedProbabilityConnector, ConnectorMixin):
    definition_url = "%s/connectionrules/fixed_probability.xml" % catalog_url 
    parameter_names = ('p_connect', 'allow_self_connections')
   
   
class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector, ConnectorMixin):
    definition_url = "%s/connectionrules/distance_dependent_probability.xml" % catalog_url
    parameter_names = ('d_expression', 'allow_self_connections') # space


def list_connectors():
    return [FixedProbabilityConnector, DistanceDependentProbabilityConnector]