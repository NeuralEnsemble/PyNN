"""
Connection method classes for the nemo module

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: connectors.py 833 2010-11-26 11:56:38Z pierre $
"""
from pyNN.space import Space
from pyNN.random import RandomDistribution
import numpy
from pyNN import random, common, core
from pyNN.connectors import AllToAllConnector, \
                            ProbabilisticConnector, \
                            OneToOneConnector, \
                            FixedProbabilityConnector, \
                            DistanceDependentProbabilityConnector, \
                            FromListConnector, \
                            FromFileConnector, \
                            FixedNumberPreConnector, \
                            FixedNumberPostConnector, \
                            SmallWorldConnector, \
                            CSAConnector
