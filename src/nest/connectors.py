"""
Connection method classes for nest

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import random, core, errors
from pyNN.connectors import Connector, \
                            DistanceDependentProbabilityConnector, \
                            DisplacementDependentProbabilityConnector, \
                            IndexBasedProbabilityConnector, \
                            SmallWorldConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            CSAConnector, \
                            CloneConnector, \
                            ArrayConnector

class FixedProbabilityConnector() :
    def __init__(self, p_connect, allow_self_connections=True, with_replacement=True,
                 rng=None, safe=True, callback=None): 
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        
        self.p_connect = float(p_connect)
#        self.rng = _get_rng(rng)


    def connect(self, projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                       'rule' : 'pairwise_bernoulli',
                       'p' : self.p_connect}
        projection._connect(rule_params, syn_params)

class AllToAllConnector() :
    def __init__(self, allow_self_connections=True, with_replacement=True, safe=True,
                 callback=None):
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        
    def connect(self,projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                      'rule' : 'all_to_all'}

        projection._connect(rule_params, syn_params)

class OneToOneConnector() :
    def __init__(self, allow_self_connections=True, with_replacement=True, safe=True,
                 callback=None):
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
    
    def connect(self,projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                       'rule' : 'one_to_one'}

        projection._connect(rule_params, syn_params)

class FixedNumberPreConnector() :
    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
                 callback=None):
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        self.n = n

    def connect(self,projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                       'rule' : 'fixed_indegree',
                       'indegree' : self.n }

        projection._connect(rule_params, syn_params)

class FixedNumberPostConnector() :
    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
                 callback=None):
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        self.n = n

    def connect(self,projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                       'rule' : 'fixed_outdegree',
                       'outdegree' : self.n }

        projection._connect(rule_params, syn_params)

class FixedTotalNumberConnector() :
    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
                 callback=None):
        self.allow_self_connections = allow_self_connections
        self.with_replacement = with_replacement
        self.n = n

    def connect(self,projection) :
        syn_params = projection.synapse_parameters()
        rule_params = {'autapses' : self.allow_self_connections,
                       'multapses' : self.with_replacement,
                       'rule' : 'fixed_total_number',
                       'N' : self.n
                   }

        projection._connect(rule_params, syn_params)
