"""
Connection method classes for nest

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN import random, core, errors
from pyNN.connectors import Connector, \
                            AllToAllConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            DisplacementDependentProbabilityConnector, \
                            IndexBasedProbabilityConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector, \
                            OneToOneConnector, \
                            SmallWorldConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            CloneConnector, \
                            ArrayConnector

import nest

try:
    import csa
    haveCSA = True
except ImportError:
    haveCSA = False

class CSAConnector(Connector):
    """
    Use the Connection Set Algebra (Djurfeldt, 2012) to connect cells.

    Takes any of the standard :class:`Connector` optional arguments and, in
    addition:

        `cset`:
            a connection set object.
    """
    parameter_names = ('cset',)

    if haveCSA:
        def __init__ (self, cset, safe=True, callback=None):
            """
            """
            Connector.__init__(self, safe=safe, callback=callback)
            self.cset = cset
            arity = csa.arity(cset)
            assert arity in (0, 2), 'must specify mask or connection-set with arity 0 or 2'
    else:
        def __init__ (self, cset, safe=True, callback=None):
            raise RuntimeError, "CSAConnector not available---couldn't import csa module"

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
