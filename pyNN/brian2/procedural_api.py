"""
Brian 2 implementation of the "procedural" API

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from .. import common
from ..connectors import FixedProbabilityConnector
from .populations import Population
from .projections import Projection
from .standardmodels.synapses import StaticSynapse
from . import simulator


create = common.build_create(Population)


connect = common.build_connect(Projection, FixedProbabilityConnector, StaticSynapse)


set = common.set


record = common.build_record(simulator)


def record_v(source, filename):
    return record(['v'], source, filename)


def record_gsyn(source, filename):
    return record(['gsyn_exc', 'gsyn_inh'], source, filename)
