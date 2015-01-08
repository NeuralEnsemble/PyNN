"""
Connection method classes for nest

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import logging
import nest
try:
    import csa
    haveCSA = True
except ImportError:
    haveCSA = False
from pyNN import random, core, errors
from pyNN.connectors import (Connector,
                             AllToAllConnector,
                             FixedProbabilityConnector,
                             OneToOneConnector,
                             FixedNumberPreConnector,
                             FixedNumberPostConnector,
                             DistanceDependentProbabilityConnector,
                             DisplacementDependentProbabilityConnector,
                             IndexBasedProbabilityConnector,
                             SmallWorldConnector,
                             FromListConnector,
                             FromFileConnector,
                             CloneConnector,
                             ArrayConnector)

from .random import NativeRNG, NEST_RDEV_TYPES


logger = logging.getLogger("PyNN")


if not nest.sli_func("statusdict/have_libneurosim ::"):

    print(("CSAConnector: libneurosim support not available in NEST.\n" +
           "Falling back on PyNN's default CSAConnector.\n" +
           "Please re-compile NEST using --with-libneurosim=PATH"))

    from pyNN.connectors import CSAConnector

else:

    class CSAConnector(Connector):
        """
        Use the Connection-Set Algebra (Djurfeldt, 2012) to connect
        cells. This is an optimized variant of CSAConnector, which
        iterates the connection-set on the C++ level in NEST.

        See Djurfeldt et al. (2014) doi:10.3389/fninf.2014.00043 for
        more details about the new interface and a comparison between
        this and PyNN's native CSAConnector.

        Takes any of the standard :class:`Connector` optional
        arguments and, in addition:

            `cset`:
                a connection set object.
        """
        parameter_names = ('cset',)

        if haveCSA:
            def __init__(self, cset, safe=True, callback=None):
                """
                """
                Connector.__init__(self, safe=safe, callback=callback)
                self.cset = cset
                arity = csa.arity(cset)
                assert arity in (0, 2), 'must specify mask or connection-set with arity 0 or 2'
        else:
            def __init__(self, cset, safe=True, callback=None):
                raise RuntimeError("CSAConnector not available---couldn't import csa module")

        def connect(self, projection):
            """Connect-up a Projection."""

            presynaptic_cells = projection.pre.all_cells.astype('int64')
            postsynaptic_cells = projection.post.all_cells.astype('int64')

            if csa.arity(self.cset) == 2:
                param_map = {'weight': 0, 'delay': 1}
                nest.CGConnect(presynaptic_cells, postsynaptic_cells, self.cset,
                               param_map, projection.nest_synapse_model)
            else:
                nest.CGConnect(presynaptic_cells, postsynaptic_cells, self.cset,
                               model=projection.nest_synapse_model)

            projection._connections = None  # reset the caching of the connection list, since this will have to be recalculated
            projection._sources.extend(presynaptic_cells)


class NESTConnectorMixin(object):

    def synapse_parameters(self, projection):
        params = {'model': projection.nest_synapse_model}
        parameter_space = self._parameters_from_synapse_type(projection, distance_map=None)
        for name, value in parameter_space.items():
            if name in ('tau_minus', 'dendritic_delay_fraction', 'w_min_always_zero_in_NEST'):
                continue
            if isinstance(value.base_value, random.RandomDistribution):     # Random Distribution specified
                if isinstance(value.base_value.rng, NativeRNG):
                    logger.warning("Random values will be created inside NEST with NEST's own RNGs")
                    params[name] = value.evaluate().repr()
                else:
                    value.shape = (projection.pre.size, projection.post.size)
                    params[name] = value.evaluate()
            else:                                             # explicit values given
                if value.is_homogeneous:
                    params[name] = value.evaluate(simplify=True)
                elif value.shape:
                    params[name] = value.evaluate().flatten()    # If parameter is given as an array or function
                else:
                    value.shape = (1, 1)
                    params[name] = float(value.evaluate())  # If parameter is given as a single number. Checking of the dimensions should be done in NEST
                if name == "weight" and projection.receptor_type == 'inhibitory' and self.post.conductance_based:
                    params[name] *= -1  # NEST wants negative values for inhibitory weights, even if these are conductances
        return params


class FixedProbabilityConnector(FixedProbabilityConnector, NESTConnectorMixin):

    def connect(self, projection):
        if projection.synapse_type.native_parameters.has_native_rngs or isinstance(self.rng, NativeRNG):
            return self.native_connect(projection)
        else:
            return super(FixedProbabilityConnector, self).connect(projection)

    def native_connect(self, projection):
        syn_params = self.synapse_parameters(projection)
        rule_params = {'autapses': self.allow_self_connections,
                       'multapses': False,
                       'rule': 'pairwise_bernoulli',
                       'p': self.p_connect}
        projection._connect(rule_params, syn_params)


class AllToAllConnector(AllToAllConnector, NESTConnectorMixin):

    def connect(self, projection):
        if projection.synapse_type.native_parameters.has_native_rngs:  # or projection.synapse_type.native_parameters.non_random:  TODO
            return self.native_connect(projection)
        else:
            return super(AllToAllConnector, self).connect(projection)

    def native_connect(self, projection):
        syn_params = self.synapse_parameters(projection)
        rule_params = {'autapses': self.allow_self_connections,
                       'multapses': False,
                       'rule': 'all_to_all'}
        projection._connect(rule_params, syn_params)


#class OneToOneConnector():
#
#    def __init__(self, allow_self_connections=True, with_replacement=True, safe=True,
#                 callback=None):
#        self.allow_self_connections = allow_self_connections
#        self.with_replacement = with_replacement
#
#    def connect(self, projection):
#        syn_params = projection.synapse_parameters()
#        rule_params = {'autapses': self.allow_self_connections,
#                       'multapses': self.with_replacement,
#                       'rule': 'one_to_one'}
#
#        projection._connect(rule_params, syn_params)
#
#
#class FixedNumberPreConnector():
#
#    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
#                 callback=None, rng=None):
#        self.allow_self_connections = allow_self_connections
#        self.with_replacement = with_replacement
#        self.n = n
#
#    def connect(self, projection):
#        syn_params = projection.synapse_parameters()
#        rule_params = {'autapses': self.allow_self_connections,
#                       'multapses': self.with_replacement,
#                       'rule': 'fixed_indegree',
#                       'indegree': self.n }
#
#        projection._connect(rule_params, syn_params)
#
#
#class FixedNumberPostConnector():
#
#    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
#                 callback=None, rng=None):
#        self.allow_self_connections = allow_self_connections
#        self.with_replacement = with_replacement
#        self.n = n
#
#    def connect(self, projection):
#        syn_params = projection.synapse_parameters()
#        rule_params = {'autapses': self.allow_self_connections,
#                       'multapses': self.with_replacement,
#                       'rule': 'fixed_outdegree',
#                       'outdegree': self.n }
#
#        projection._connect(rule_params, syn_params)
#
#
#class FixedTotalNumberConnector():
#
#    def __init__(self, n, allow_self_connections=True, with_replacement=True, safe=True,
#                 callback=None):
#        self.allow_self_connections = allow_self_connections
#        self.with_replacement = with_replacement
#        self.n = n
#
#    def connect(self, projection):
#        syn_params = projection.synapse_parameters()
#        rule_params = {'autapses': self.allow_self_connections,
#                       'multapses': self.with_replacement,
#                       'rule': 'fixed_total_number',
#                       'N': self.n
#                   }
#        projection._connect(rule_params, syn_params)
