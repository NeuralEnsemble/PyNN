"""
Connection method classes for the neuron module

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

# flake8: noqa

from . import simulator
from ..connectors import (
    AllToAllConnector,
    OneToOneConnector,
    FixedProbabilityConnector,
    DistanceDependentProbabilityConnector,
    DisplacementDependentProbabilityConnector,
    IndexBasedProbabilityConnector,
    FromListConnector,
    FromFileConnector,
    FixedNumberPreConnector,
    FixedNumberPostConnector,
    SmallWorldConnector,
    CSAConnector,
    CloneConnector,
    ArrayConnector,
    FixedTotalNumberConnector,
)
