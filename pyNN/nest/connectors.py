# -*- coding: utf-8 -*-
"""
Connection method classes for NEST.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
from warnings import warn
import nest
try:
    import csa
    haveCSA = True
except ImportError:
    haveCSA = False

from .. import random
from ..connectors import (                      # noqa: F401
    Connector,
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
    ArrayConnector,
    FixedTotalNumberConnector,
    CSAConnector as DefaultCSAConnector)
from .random import NativeRNG


logger = logging.getLogger("PyNN")


class CSAConnector(DefaultCSAConnector):
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

    def connect(self, projection):
        if nest.ll_api.sli_func("statusdict/have_libneurosim ::"):
            return self.cg_connect(projection)
        else:
            warn("Note: using the default CSAConnector. To use the accelerated version for NEST,\n"
                 "Please re-compile NEST using --with-libneurosim=PATH")
            return super(CSAConnector, self).connect(projection)

    def cg_connect(self, projection):
        """Connect-up a Projection using the Connection Generator interface"""

        presynaptic_cells = projection.pre.all_cells.astype('int64')
        postsynaptic_cells = projection.post.all_cells.astype('int64')

        if csa.arity(self.cset) == 2:
            param_map = {'weight': 0, 'delay': 1}
            nest.CGConnect(presynaptic_cells, postsynaptic_cells, self.cset,
                           param_map, projection.nest_synapse_model)
        else:
            nest.CGConnect(presynaptic_cells, postsynaptic_cells, self.cset,
                           model=projection.nest_synapse_model)

        # reset the caching of the connection list, since this will have to be recalculated
        projection._connections = None
        projection._sources.extend(presynaptic_cells)


class NESTConnectorMixin(object):

    def synapse_parameters(self, projection):
        params = {'synapse_model': projection.nest_synapse_model}
        parameter_space = self._parameters_from_synapse_type(projection, distance_map=None)
        for name, value in parameter_space.items():
            if name in ('tau_minus', 'dendritic_delay_fraction', 'w_min_always_zero_in_NEST'):
                continue
            if isinstance(value.base_value, random.RandomDistribution):
                # Random Distribution specified
                if isinstance(value.base_value.rng, NativeRNG):
                    logger.warning(
                        "Random values will be created inside NEST with NEST's own RNGs")
                    # todo: re-enable support for clipped and clipped_to_boundary
                    params[name] = value.evaluate().as_nest_object()
                else:
                    value.shape = (projection.pre.size, projection.post.size)
                    params[name] = value.evaluate()
            else:
                # explicit values given
                if value.is_homogeneous:
                    params[name] = value.evaluate(simplify=True)
                elif value.shape:
                    # If parameter is given as an array or function
                    params[name] = value.evaluate().flatten()
                else:
                    value.shape = (1, 1)
                    # If parameter is given as a single number.
                    # Checking of the dimensions should be done in NEST
                    params[name] = float(value.evaluate())
                if (
                    name == "weight"
                    and projection.receptor_type == 'inhibitory'
                    and self.post.conductance_based
                ):
                    # NEST wants negative values for inhibitory weights,
                    # even if these are conductances
                    params[name] *= -1
        return params


class FixedProbabilityConnector(FixedProbabilityConnector, NESTConnectorMixin):

    def connect(self, projection):
        if (
            projection.synapse_type.native_parameters.has_native_rngs
            or isinstance(self.rng, NativeRNG)
        ):
            return self.native_connect(projection)
        else:
            return super(FixedProbabilityConnector, self).connect(projection)

    def native_connect(self, projection):
        syn_params = self.synapse_parameters(projection)
        rule_params = {'allow_autapses': self.allow_self_connections,
                       'allow_multapses': False,
                       'rule': 'pairwise_bernoulli',
                       'p': self.p_connect}
        projection._connect(rule_params, syn_params)


class AllToAllConnector(AllToAllConnector, NESTConnectorMixin):

    def connect(self, projection):
        # or projection.synapse_type.native_parameters.non_random:  TODO
        if projection.synapse_type.native_parameters.has_native_rngs:
            return self.native_connect(projection)
        else:
            return super(AllToAllConnector, self).connect(projection)

    def native_connect(self, projection):
        syn_params = self.synapse_parameters(projection)
        rule_params = {'allow_autapses': self.allow_self_connections,
                       'allow_multapses': False,
                       'rule': 'all_to_all'}
        projection._connect(rule_params, syn_params)


# class OneToOneConnector():
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
# class FixedNumberPreConnector():
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
# class FixedNumberPostConnector():
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
# class FixedTotalNumberConnector():
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
