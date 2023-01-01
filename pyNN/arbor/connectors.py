"""
Connection method classes for the Arbor module

:copyright: Copyright 2006-2020 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pyNN.arbor import simulator
from pyNN.connectors import AllToAllConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            DisplacementDependentProbabilityConnector, \
                            IndexBasedProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector, \
                            SmallWorldConnector, \
                            CSAConnector, \
                            CloneConnector, \
                            ArrayConnector, \
                            FixedTotalNumberConnector
